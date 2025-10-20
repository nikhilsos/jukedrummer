import os
import sys
import argparse
import pickle
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid

from model.LanguageModel import JukeTransformer
from model.vqvae import VQVAE, Sampler
from dataset import BeatInfoPairedDataset
from jukebox.jukebox.train import get_optimizer
from hparams import OPT, MODEL_LIST, setup_lm_hparams


# ------------------------------
# Collate & tensor helpers
# ------------------------------
def _to_tensor(x, dtype=None):
    if isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x)
        t = torch.from_numpy(np.array(x, copy=True))
    elif isinstance(x, torch.Tensor):
        t = x.clone().contiguous()
    else:
        t = torch.tensor(x)
    return t.to(dtype) if dtype is not None else t

def _pad_crop_1d(x: torch.Tensor, T: int, pad_val: int):
    L = x.size(0)
    if L == T: return x
    if L > T:
        start = (L - T) // 2
        return x[start:start+T]
    out = x.new_full((T,), pad_val)
    out[:L] = x
    return out

def _pad_crop_2d(x: torch.Tensor, T: int, pad_val: float):
    # expect (L, D)
    if x.dim() != 2:
        raise ValueError(f"binfo must be 2D (L,D); got {tuple(x.shape)}")
    L, D = x.size(0), x.size(1)
    if L == T: return x
    if L > T:
        start = (L - T) // 2
        return x[start:start+T, :]
    out = x.new_full((T, D), pad_val)
    out[:L, :] = x
    return out

def _expected_binfo_dim(binfo_type: str) -> int:
    t = (binfo_type or "").lower()
    if t in ("low", "lowlevel", "low_level"): return 50
    if t in ("mid", "midlevel", "mid_level"): return 3
    if t in ("high", "highlevel", "high_level"): return 3
    raise ValueError(f"Unknown binfo_type: {binfo_type}")

def collate_lm(batch, LM_T: int = 1024, pad_tok: int = 0, pad_binfo: float = 0.0, binfo_dim: int = 50):
    """
    batch: list of (tgz, otz, ot_binfo)
    - tgz: (L,)
    - otz: (L,)
    - ot_binfo: (L, D) with D == binfo_dim
    """
    tgz_list, otz_list, binfo_list = zip(*batch)
    tgz_t, otz_t, binfo_t = [], [], []
    for t, o, b in zip(tgz_list, otz_list, binfo_list):
        t = _to_tensor(t, torch.long)
        o = _to_tensor(o, torch.long)
        b = _to_tensor(b, torch.float32)

        # Make sure binfo is (L, D). If user stored (D, L), transpose.
        if b.dim() == 2 and b.shape[0] == binfo_dim and b.shape[1] != binfo_dim:
            # probably (D, L) -> (L, D)
            b = b.transpose(0, 1)
        if b.dim() != 2 or b.shape[-1] != binfo_dim:
            raise ValueError(f"binfo last dim mismatch: got {tuple(b.shape)}, expected D={binfo_dim}")

        t = _pad_crop_1d(t, LM_T, pad_tok)
        o = _pad_crop_1d(o, LM_T, pad_tok)
        b = _pad_crop_2d(b, LM_T, pad_binfo)

        tgz_t.append(t)
        otz_t.append(o)
        binfo_t.append(b)

    tgz   = torch.stack(tgz_t,   dim=0)  # (B, T)
    otz   = torch.stack(otz_t,   dim=0)  # (B, T)
    binfo = torch.stack(binfo_t, dim=0)  # (B, T, D)
    return tgz, otz, binfo


# ------------------------------
# Dataset loader with checks
# ------------------------------
def load_id_splits(hps):
    ds_path = os.path.join(hps.path, "dataset.pkl")
    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"Dataset file not found at: {ds_path}")

    with open(ds_path, "rb") as f:
        dataset = pickle.load(f)

    if not isinstance(dataset, (list, tuple)) or len(dataset) != 2:
        raise ValueError(f"Invalid dataset format in {ds_path}. Expected [train_ids, valid_ids].")
    tr_ids, va_ids = dataset
    if not tr_ids or not va_ids:
        raise ValueError(f"Empty splits in {ds_path}. train={len(tr_ids)} valid={len(va_ids)}")
    return tr_ids, va_ids


def get_dataloaders(hps, binfo_dim, lm_T):
    tr_ids, va_ids = load_id_splits(hps)
    # print(f"number of training data: {len(tr_ids)}")
    # print(f"number of validation data: {len(va_ids)}")

    tr_dataset = BeatInfoPairedDataset(tr_ids, hps)
    va_dataset = BeatInfoPairedDataset(va_ids, hps)
    if len(tr_dataset) == 0 or len(va_dataset) == 0:
        raise ValueError("Training or validation dataset is empty after instantiation.")

    collate_fn = lambda batch: collate_lm(
        batch,
        LM_T=lm_T,
        pad_tok=0,
        pad_binfo=0.0,
        binfo_dim=binfo_dim
    )

    tr_loader = DataLoader(
        dataset=tr_dataset,
        batch_size=hps.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    va_loader = DataLoader(
        dataset=va_dataset,
        batch_size=hps.batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if len(tr_loader) == 0 or len(va_loader) == 0:
        raise ValueError("Dataloaders are empty. Check dataset paths/contents.")
    return tr_loader, va_loader


# ------------------------------
# Solver
# ------------------------------
class Solver:
    def __init__(self, model, vqvae, device, use_wandb=False):
        self.device = device
        self.model = model
        self.vqvae = vqvae
        self.criterion = nn.CrossEntropyLoss()  # model is expected to return loss already; keep for consistency
        self.opt, self.shd, _ = get_optimizer(self.model, OPT)
        self.use_wandb = use_wandb

    def run_batch(self, data, summary, training=True, make_sample=False):
        # Expect (tgz, otz, ot_binfo)
        tgz, otz, ot_binfo = data
        tgz = tgz.long().to(self.device)        # (B, T)
        otz = otz.long().to(self.device)        # (B, T)
        ot_binfo = ot_binfo.float().to(self.device)  # (B, T, D)

        if training:
            self.model.train()
            self.opt.zero_grad()
        else:
            self.model.eval()

        # Defensive shape/dtype checks before forward
        B, T = tgz.shape
        assert otz.shape == (B, T), f"otz shape {otz.shape} != {(B,T)}"
        assert ot_binfo.shape[0] == B and ot_binfo.shape[1] == T, f"binfo shape {ot_binfo.shape} must be (B,T,D)"

        loss, pred = self.model(tgz, otz, ot_binfo)  # model returns (loss, logits)
        key = 'train loss' if training else 'valid loss'
        summary[key] = float(loss.item())

        if training:
            loss.backward()
            self.opt.step()
            self.shd.step()

        if make_sample and self.use_wandb:
            import wandb
            with torch.no_grad():
                pred_ids = torch.argmax(pred, dim=-1)      # (B, T)
                pred_mel = self.vqvae.decode(pred_ids)     # expects token IDs
                real_mel = self.vqvae.decode(tgz)

                pred_img = make_grid(pred_mel[:4].unsqueeze(1), nrow=1).detach().cpu().numpy().transpose(1,2,0)
                real_img = make_grid(real_mel[:4].unsqueeze(1), nrow=1).detach().cpu().numpy().transpose(1,2,0)
                summary[f'{"train" if training else "valid"} pred mel'] = wandb.Image(pred_img)
                summary[f'{"train" if training else "valid"} real mel'] = wandb.Image(real_img)

        return summary


# ------------------------------
# Main
# ------------------------------
def main():
    # Args
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', type=int, default=None)
    ap.add_argument('--exp_idx', type=int, required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--wandb', action='store_true')
    ap.add_argument('--bs', type=int, default=None)
    ap.add_argument('--lm_t', type=int, default=1024, help="sequence length T for LM (after downsampling)")
    args = ap.parse_args()

    # Hparams
    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])
    if args.bs:
        hps.batch_size = args.bs
    binfo_dim = _expected_binfo_dim(getattr(hps, 'binfo_type', 'low'))

    # Device
    if torch.cuda.is_available() and args.cuda is not None:
        device = torch.device(f'cuda:{args.cuda}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"[INFO] Using device: {device}")

    # Model
    model = JukeTransformer(hps).to(device)
    if args.resume:
        os.makedirs(hps.ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(hps.ckpt_dir, f'exp{args.exp_idx}.pkl')
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            print(f"[INFO] Resumed from {ckpt_path}")
        else:
            print(f"[WARN] --resume set but no checkpoint at {ckpt_path}")

    # VQVAE (decoder used for logging only)
    vqvae = VQVAE(
        codebook_size=hps.codebook_size,
        encoder=Sampler(input_dim=80, output_dim=64, z_scale_factors=hps.downsample_ratios),
        decoder=Sampler(input_dim=64, output_dim=80, z_scale_factors=hps.upsample_ratios),
    )
    _mean, _std = vqvae.restore_from_ckpt(hps, device)

    # Data
    tr_loader, va_loader = get_dataloaders(hps, binfo_dim=binfo_dim, lm_T=args.lm_t)

    # WandB
    try:
        import wandb
        is_wandb = bool(args.wandb)
        if is_wandb:
            wandb.init(
                project='JukeDrummer Language model',
                config=dict(hps),
                dir='./wandb',
                name=f'exp{args.exp_idx}',
            )
    except ImportError:
        wandb = None
        is_wandb = False

    solver = Solver(model, vqvae, device, use_wandb=is_wandb)

    # Train loop with early stopping
    os.makedirs(hps.ckpt_dir, exist_ok=True)
    # patience = getattr(hps, 'early_stop_patience', 10)
    best_val = float('inf')
    no_improve = 0

    for epoch in range(OPT.epochs):
        summary = {}
        # ---- TRAIN ----
        bad_batches = 0
        pbar = tqdm(tr_loader, desc=f"train {epoch}")
        for bidx, batch in enumerate(pbar):
            try:
                make_sample = (bidx == len(tr_loader) - 1 and epoch % hps.sample_step == 0)
                summary = solver.run_batch(batch, summary, training=True, make_sample=make_sample)
                pbar.set_postfix(loss=f"{summary['train loss']:.4f}")
            except RuntimeError as e:
                # CUDA OOM or matmul/shape errors — skip batch, log once
                bad_batches += 1
                msg = str(e)
                print(f"[WARN][train][batch {bidx}] Skipping due to error: {msg}")
                if "out of memory" in msg.lower():
                    torch.cuda.empty_cache()
                continue
            except Exception as e:
                bad_batches += 1
                print(f"[WARN][train][batch {bidx}] Unexpected error, skipping: {e}")
                continue

        if bad_batches:
            print(f"[INFO] train epoch {epoch}: skipped {bad_batches} bad batches")

        # ---- VALID ----
        with torch.no_grad():
            vbad = 0
            vloss_acc = 0.0
            vcount = 0
            vpbar = tqdm(va_loader, desc=f"valid {epoch}")
            for bidx, batch in enumerate(vpbar):
                try:
                    make_sample = (bidx == len(va_loader) - 1 and epoch % hps.sample_step == 0)
                    summary = solver.run_batch(batch, summary, training=False, make_sample=make_sample)
                    vloss_acc += summary['valid loss']
                    vcount += 1
                    vpbar.set_postfix(vloss=f"{summary['valid loss']:.4f}")
                except Exception as e:
                    vbad += 1
                    print(f"[WARN][valid][batch {bidx}] Skipping due to error: {e}")
                    continue
            if vcount > 0:
                val_loss = vloss_acc / vcount
            else:
                val_loss = float('inf')
            print(f"[EPOCH {epoch}] train_loss={summary.get('train loss','?')} | valid_loss={val_loss:.6f}")

        # ---- Checkpoints ----
        # save current
        torch.save(
                {'model': model.state_dict(), 'hps': dict(hps)},
                os.path.join(hps.ckpt_dir, f'exp{args.exp_idx}.pkl'),
            )
        
        # save best & early stopping
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save(
            {'model': model.state_dict(), 'hps': dict(hps)},
            os.path.join(hps.ckpt_dir, f'exp{args.exp_idx}_best.pkl'),
        )
            print(f"[INFO] New best valid loss: {best_val:.6f} (saved)")
        # else:
        #     no_improve += 1
        #     print(f"[INFO] No improvement ({no_improve}/{patience})")
        #     if no_improve >= patience:
        #         print("[INFO] Early stopping.")
        #         break

        if is_wandb:
            if 'valid loss' in summary:
                summary['valid loss (avg)'] = val_loss
            wandb.log(summary, step=epoch)


if __name__ == '__main__':
    main()
