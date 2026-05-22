"""Generate demo samples for the README / GitHub page."""
import os, sys, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from hparams import setup_lm_hparams, MODEL_LIST, MEL
from model.LanguageModel import JukeTransformer
from model.vqvae import VQVAE, Sampler
from model.vocoder import HiFiVocoder
from dataset import BeatInfoPairedDataset
from utils.functions import get_vqvae, mel_gate

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EXP_IDX = 30
CKPT_PATH = 'ckpt/exp30_ce_best.pkl'
SR = 44100
OUT_DIR = 'demo_samples'
NUM_SAMPLES = 5

os.makedirs(OUT_DIR, exist_ok=True)

# Load models
print("Loading models...")
hps = setup_lm_hparams(MODEL_LIST[EXP_IDX])
hps.batch_size = 1
latent_dim = hps.get('latent_dim', 64)

vqvae = VQVAE(
    codebook_size=hps.codebook_size,
    encoder=Sampler(80, latent_dim, hps.downsample_ratios),
    decoder=Sampler(latent_dim, 80, hps.upsample_ratios),
    latent_dim=latent_dim,
).to(DEVICE)
vqvae.restore_from_ckpt(hps, DEVICE)
vqvae.eval()
vqvae.requires_grad_(False)

vq_tag = hps.get('vq_ckpt_tag', 'target')
vq_ckpt = torch.load(
    os.path.join(hps.ckpt_dir, f"{hps.vq_name}_{vq_tag}.pkl"),
    map_location=DEVICE, weights_only=False,
)
mean = torch.FloatTensor(vq_ckpt['mean']).to(DEVICE)
std = torch.FloatTensor(vq_ckpt['std']).to(DEVICE)

others_vqvae, _, others_mean, others_std = get_vqvae(
    vq_idx=int(hps.vq_name.strip('vq')),
    data_type='others', ckpt_dir=hps.ckpt_dir, device=DEVICE,
)
others_vqvae.eval()
others_vqvae.requires_grad_(False)

lm = JukeTransformer(hps).to(DEVICE)
lm_ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
lm.load_state_dict(lm_ckpt['model'], strict=True)
lm.eval()

vocoder = HiFiVocoder(
    ckpt_path='cp_hifigan/g_00000635',
    output_dir=OUT_DIR,
    device=DEVICE,
)

with open(os.path.join(hps.path, 'dataset.pkl'), 'rb') as f:
    _, va_ids = pickle.load(f)
dataset = BeatInfoPairedDataset(va_ids, hps, return_fn=True)

def normalize_audio(wav_np):
    peak = np.abs(wav_np).max()
    return wav_np / peak if peak > 0 else wav_np

def noise_gate(wav_np, threshold_db=-40, sr=44100):
    threshold = 10 ** (threshold_db / 20.0)
    window_len = max(int(sr * 5 / 1000), 1)
    kernel = np.ones(window_len) / window_len
    envelope = np.convolve(np.abs(wav_np), kernel, mode='same')
    gate = (envelope > threshold).astype(np.float32)
    release_samples = int(sr * 50 / 1000)
    if release_samples > 1:
        smooth_kernel = np.ones(release_samples) / release_samples
        gate = np.convolve(gate, smooth_kernel, mode='same')
        gate = np.clip(gate, 0, 1)
    return wav_np * gate

def to_wav_np(wav):
    if isinstance(wav, torch.Tensor):
        return wav.detach().cpu().numpy().squeeze()
    return np.array(wav).squeeze()


# Pick diverse samples (spread across dataset)
indices = np.linspace(0, len(dataset) - 1, NUM_SAMPLES, dtype=int)

print(f"Generating {NUM_SAMPLES} demo samples...")
for i, idx in enumerate(indices):
    idx = int(idx)
    sample = dataset[idx]
    tg_token, ot_token, binfo, fn = sample[0], sample[1], sample[2], sample[3]
    rest = list(sample[4:])
    class_id = rest[0] if len(rest) > 0 else None
    vocal_feat = rest[1] if len(rest) > 1 else None
    sample_name = fn.replace('.npy', '')
    print(f"  [{i+1}/{NUM_SAMPLES}] {fn} (idx={idx})")

    tgz = torch.from_numpy(np.array(tg_token)).long().unsqueeze(0).to(DEVICE)
    otz = torch.from_numpy(np.array(ot_token)).long().unsqueeze(0).to(DEVICE)
    binfo_t = torch.from_numpy(np.array(binfo)).float().unsqueeze(0).to(DEVICE)
    class_id_t = torch.tensor([class_id]).long().to(DEVICE) if class_id is not None else None
    vocal_feat_t = torch.from_numpy(np.array(vocal_feat)).float().unsqueeze(0).to(DEVICE) if vocal_feat is not None else None

    with torch.no_grad():
        gt_mel = vqvae.decode(tgz) * std + mean
        gen_mel = lm.primed_sample(
            n_samples=1, otz=otz, binfo=binfo_t, vqvae=vqvae,
            temp=0.9, top_p=0.95, class_id=class_id_t,
            rep_penalty=1.0, rep_window=16, vocal_feat=vocal_feat_t,
        )
        gen_mel = gen_mel * std + mean
        # Accompaniment (vocal) audio: reconstruct from the others VQ-VAE
        acc_mel = others_vqvae.decode(otz) * others_std + others_mean

    # Spectrograms
    raw_mel_path = os.path.join(hps.path, 'mel', 'target', fn)
    raw_mel_np = np.load(raw_mel_path) if os.path.exists(raw_mel_path) else gt_mel[0].cpu().numpy()
    gt_mel_np = gt_mel[0].cpu().numpy()
    gen_mel_np = gen_mel[0].cpu().numpy()

    all_vals = np.concatenate([raw_mel_np.flatten(), gt_mel_np.flatten(), gen_mel_np.flatten()])
    vmin, vmax = np.percentile(all_vals, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(18, 3.5))
    for ax, title, m in zip(axes,
        ['Original drums', 'VQ-VAE reconstruction', 'Generated drums'],
        [raw_mel_np, gt_mel_np, gen_mel_np]):
        im = ax.imshow(m, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Time frames')
        ax.set_ylabel('Mel bins')
        plt.colorbar(im, ax=ax)
    plt.suptitle(f'Sample: {sample_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'sample_{i+1}_spectrogram.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Audio
    gt_wav_np = noise_gate(normalize_audio(to_wav_np(vocoder(gt_mel))))
    gen_wav_np = noise_gate(normalize_audio(to_wav_np(vocoder(gen_mel))))

    sf.write(os.path.join(OUT_DIR, f'sample_{i+1}_original_drums.wav'), gt_wav_np, SR)
    sf.write(os.path.join(OUT_DIR, f'sample_{i+1}_generated_drums.wav'), gen_wav_np, SR)

    # Others (vocal) audio from VQ-VAE reconstruction + mix
    others_wav_np = normalize_audio(to_wav_np(vocoder(acc_mel)))

    min_len = min(len(gen_wav_np), len(others_wav_np), len(gt_wav_np))
    sf.write(os.path.join(OUT_DIR, f'sample_{i+1}_accompaniment.wav'), others_wav_np[:min_len], SR)

    mix_vol = 0.5
    mixed_gen = mix_vol * gen_wav_np[:min_len] + (1.0 - mix_vol) * others_wav_np[:min_len]
    peak = np.abs(mixed_gen).max()
    if peak > 0.95:
        mixed_gen *= 0.95 / peak
    sf.write(os.path.join(OUT_DIR, f'sample_{i+1}_mix_generated.wav'), mixed_gen, SR)

    mixed_gt = mix_vol * gt_wav_np[:min_len] + (1.0 - mix_vol) * others_wav_np[:min_len]
    peak = np.abs(mixed_gt).max()
    if peak > 0.95:
        mixed_gt *= 0.95 / peak
    sf.write(os.path.join(OUT_DIR, f'sample_{i+1}_mix_original.wav'), mixed_gt, SR)

print(f"\nDone! Samples saved to {OUT_DIR}/")
print("Files per sample: spectrogram.png, original_drums.wav, generated_drums.wav, accompaniment.wav, mix_generated.wav, mix_original.wav")
