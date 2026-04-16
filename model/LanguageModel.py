import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from fast_transformers.transformers import *
import numpy as np

from jukebox.jukebox.transformer.ops import Conv1D
from model.autoregressive import ConditionalAutoregressive2D

def nucleus(probs, p=0.5):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word

def make_local_mask(seq_length, width, device):
    mask = torch.zeros(seq_length, seq_length)
    for i in range(seq_length):
        for j in range(seq_length):
            if j < i + width and j > i - width:
                mask[i][j] = 1
    return mask.bool().to(device)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1)]
        return self.dropout(x)

from jukebox.jukebox.hparams import setup_hparams
from jukebox.jukebox.transformer.ops import Conv1D, LayerNorm


class JukeTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = getattr(args, 'num_classes', None)
        self.make_juke_prior(args)
        self.use_tokens = args.name != 'lm9' and args.name != 'lm10'
        # print(f'use tokens:{self.use_tokens}')
        self.prime_state_proj = Conv1D(args.d_model, args.d_model)
        self.prime_state_ln = LayerNorm(args.d_model)
        self.binfo_type = args.binfo_type

        if self.num_classes:
            self.class_emb = nn.Embedding(self.num_classes, args.d_model)

        # token_weight_path = getattr(args, 'token_weight_path', None)
        # if token_weight_path is not None:
        #     weights = np.load(token_weight_path).astype(np.float32)
        #     self.register_buffer('token_weights', torch.from_numpy(weights))
        # else:
        #     self.token_weights = None

        if self.binfo_type == 'low':
            self.bact_state_proj = Conv1D(50, args.d_model) # changed from 50 to 1 for pansori beat
        elif self.binfo_type == 'mid':
            self.onset_emb = nn.Embedding(2, args.d_model)
        elif self.binfo_type == 'high':
            self.beat_emb = nn.Embedding(3, args.d_model)
        elif self.binfo_type == 'dbeats':
            self.beat_emb = nn.Embedding(3, args.d_model)
        elif self.binfo_type == 'dphase':
            # Projects 2-channel sin/cos phase encoding to d_model.
            # Sin/cos makes the representation cyclic: phase 0 and 1 both map
            # to [sin=0, cos=1], so the model sees no jump at the downbeat.
            self.phase_proj = nn.Linear(2, args.d_model)
        elif self.binfo_type is None:
            pass
        else:
            raise RuntimeError('No matchedbeat information type')

    def get_prime_loss(self, encoder_kv, prime_t):
        if self.use_tokens:
            encoder_kv = encoder_kv.float()
            encoder_kv = self.prime_x_out(encoder_kv)
            prime_loss = nn.functional.cross_entropy(encoder_kv.view(-1, 512), prime_t.view(-1)) / np.log(2.)
        else:
            prime_loss = torch.tensor(0.0, device='cuda')
        return prime_loss

    def get_encoder_kv(self, prime, binfo, fp16=False):
        if self.use_tokens:
            N = prime.shape[0]
            # print(f"prime min={prime.min()}, max={prime.max()}, shape={prime.shape}")
            # print(f"prime dtype={prime.dtype}")
            prime_acts = self.prime_prior(prime, binfo, None, None, fp16=fp16, isroll=False)
        
            assert prime_acts.dtype == torch.float, f'Expected torch.float, got {prime_acts.dtype}'
            encoder_kv = self.prime_state_ln(self.prime_state_proj(prime_acts))
            assert encoder_kv.dtype == torch.float, f'Expected torch.float, got {encoder_kv.dtype}'
        else:
            encoder_kv = None
        return encoder_kv
    
    def binfo_conditioner(self, binfo):
        if self.binfo_type == 'low':
            binfo = F.interpolate(binfo.unsqueeze(1), size=(self.prior.encoder_dims, 50)).squeeze(1)
            binfo = self.bact_state_proj(binfo)
        elif self.binfo_type == 'mid':
            binfo = self.onset_emb(binfo.long())
        elif self.binfo_type == 'high':
            binfo = binfo.double()
            binfo = torch.where(binfo > 1, 2., binfo)
            binfo = self.beat_emb(binfo.long())
        elif self.binfo_type == 'dbeats':
            binfo = binfo.double()
            binfo = torch.where(binfo > 1, 2., binfo)
            binfo = self.beat_emb(binfo.long())
        elif self.binfo_type == 'dphase':
            # binfo: (B, T) float in [0, 1] — sawtooth phase between downbeats
            # Encode as sin/cos so phase 0 ≡ phase 1 (both are the downbeat)
            phase = binfo.float() * (2 * math.pi)          # (B, T)
            phase_enc = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)  # (B, T, 2)
            binfo = self.phase_proj(phase_enc)              # (B, T, d_model)
        elif self.binfo_type is None:
            binfo = None

        return binfo

    def _get_y_cond(self, class_id):
        if self.num_classes and class_id is not None:
            return self.class_emb(class_id).unsqueeze(1)  # (N, 1, d_model)
        return None

    def forward(self, tgz, otz, binfo=None, class_id=None):
        binfo = self.binfo_conditioner(binfo)
        encoder_kv = self.get_encoder_kv(otz, binfo)
        y_cond = self._get_y_cond(class_id)
        loss, pred = self.prior(tgz, x_cond=binfo, y_cond=y_cond, encoder_kv=encoder_kv, fp16=False, loss_full=False,
                    encode=False, get_preds=True, get_acts=False, get_sep_loss=False)

        return loss, pred

    def sample(self, n_samples, otz, binfo, vqvae, temp=1.0, top_k=0, top_p=0.0, class_id=None):
        self.eval()
        with torch.no_grad():
            binfo = self.binfo_conditioner(binfo)
            encoder_kv = self.get_encoder_kv(otz, binfo)
            y_cond = self._get_y_cond(class_id)
            pred = self.prior.sample(n_samples, x_cond=binfo, y_cond=y_cond,
                encoder_kv=encoder_kv, fp16=False, temp=temp, top_k=top_k, top_p=top_p,
                get_preds=False, sample_tokens=None, device=otz.device
            )
            pred = vqvae.decode(pred)
        return pred

    # primed sampling with a given initial sequence (otz) and beat info (binfo)
    def primed_sample(
        self,
        n_samples,
        otz,
        binfo,
        vqvae,
        target_prefix=None,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        class_id=None
    ):
        self.eval()
        with torch.no_grad():
            binfo = self.binfo_conditioner(binfo)
            encoder_kv = self.get_encoder_kv(otz, binfo)
            y_cond = self._get_y_cond(class_id)

            # If no target prefix is supplied, do normal sampling
            sample_tokens = None
            if target_prefix is not None:
                sample_tokens = target_prefix.long()

            pred = self.prior.sample(
                n_samples,
                x_cond=binfo,
                y_cond=y_cond,
                encoder_kv=encoder_kv,
                fp16=False,
                temp=temp,
                top_k=top_k,
                top_p=top_p,
                get_preds=False,
                sample_tokens=sample_tokens,
                device=otz.device
            )

            pred = vqvae.decode(pred)

        return pred
        
    def make_juke_prior(self, args):
        sequence_length = 4096 // np.prod(args.upsample_ratios)
        # sequence_length = 32768 // np.prod(args.upsample_ratios)

        hps = setup_hparams('small_sep_enc_dec_prior', dict())
        hps['prior_depth'] = args.enc_layers
        hps['n_ctx'] = sequence_length
        hps['blocks'] = args.blocks
        hps['prior_width'] = args.d_model
        hps['attn_dropout'] = 0.3
        hps['resid_dropout']= 0.3 
        hps['emb_dropout'] = args.dropout
        hps['m_mlp'] = 1
        hps['heads'] = args.heads
        hps['attn_order'] = 8 if args.name != 'lm9' and args.name != 'lm10' else 2
        hps['prime_width']= args.d_model
        hps['prime_depth']=9
        hps['prime_heads']=2
        hps['prime_attn_order']=2
        hps['prime_blocks']= args.blocks
        hps['n_vocab'] = 1024
        hps['prime_attn_dropout'] = hps['attn_dropout']
        hps['prime_resid_dropout']= hps['resid_dropout'] 
        hps['prime_emb_dropout'] = 0
        hps['c_res'] = 0
        prior_kwargs = dict(input_shape=(hps.n_ctx,), bins=args.codebook_size,
                                width=hps.prior_width, depth=hps.prior_depth, heads=hps.heads,
                                attn_order=hps.attn_order, blocks=hps.blocks, spread=hps.spread,
                                attn_dropout=hps.attn_dropout, resid_dropout=hps.resid_dropout, emb_dropout=hps.emb_dropout,
                                zero_out=hps.zero_out, res_scale=hps.res_scale, pos_init=hps.pos_init,
                                init_scale=hps.init_scale,
                                m_attn=hps.m_attn, m_mlp=hps.m_mlp,
                                checkpoint_res=hps.c_res if hps.train else 0, checkpoint_attn=hps.c_attn if hps.train else 0, checkpoint_mlp=hps.c_mlp if hps.train else 0)
        
        prime_kwargs = dict(input_shape=(hps.n_ctx,), bins=hps.n_vocab,
                                width=hps.prime_width, depth=hps.prime_depth, heads=hps.prime_heads,
                                attn_order=hps.prime_attn_order, blocks=hps.prime_blocks, spread=hps.prime_spread,
                                attn_dropout=hps.prime_attn_dropout, resid_dropout=hps.prime_resid_dropout,
                                emb_dropout=hps.prime_emb_dropout,
                                zero_out=hps.prime_zero_out, res_scale=hps.prime_res_scale,
                                pos_init=hps.prime_pos_init, init_scale=hps.prime_init_scale,
                                m_attn=hps.prime_m_attn, m_mlp=hps.prime_m_mlp,
                                checkpoint_res=hps.prime_c_res if hps.train else 0, checkpoint_attn=hps.prime_c_attn if hps.train else 0,
                                checkpoint_mlp=hps.prime_c_mlp if hps.train else 0)
        
        self.hps = hps
        use_y_cond = self.num_classes is not None and self.num_classes > 0
        # y_cond = None
        self.prior = ConditionalAutoregressive2D(x_cond=args.binfo_type is not None, y_cond=use_y_cond, encoder_dims = sequence_length,
                                                    pos_emb=None, **prior_kwargs)
        self.prime_prior = ConditionalAutoregressive2D(x_cond=args.binfo_type is not None, y_cond=False, only_encode=True,
                                                    mask=False, pos_emb=None, **prime_kwargs)