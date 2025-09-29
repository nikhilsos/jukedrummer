import os
import torch
import numpy as np
import tqdm
from model.vqvae import VQVAE, Sampler
from hparams import setup_vq_hparams
from hifi_gan.meldataset import spectral_normalize_torch



# User parameters
TOKEN_DIR = '/home/nikhil/jukedrummer/data/token/target/vq1'
OUTPUT_DIR = '/home/nikhil/jukedrummer/data/audio/reconstructed_mel_recon_vq1'
VQ_IDX = 1
CHECKPOINT_PATH = '/home/nikhil/jukedrummer/ckpt/vq1_target.pkl'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load VQ-VAE model and hparams
hps = setup_vq_hparams(f'vq{VQ_IDX}')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
encoder = Sampler(input_dim=80, output_dim=64, z_scale_factors=hps['downsample_ratios'])
decoder = Sampler(input_dim=64, output_dim=80, z_scale_factors=hps['upsample_ratios'])
vqvae = VQVAE(
    codebook_size=hps['codebook_size'],
    encoder=encoder,
    decoder=decoder,
    device=device
).to(device)
vqvae.eval()

# Load checkpoint
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
if 'model' in ckpt:
    vqvae.load_state_dict(ckpt['model'])
elif 'state_dict' in ckpt:
    vqvae.load_state_dict(ckpt['state_dict'])
else:
    vqvae.load_state_dict(ckpt)

# Loop over token files and decode to mel
for fname in tqdm.tqdm(os.listdir(TOKEN_DIR), desc="Decoding tokens"):
    if not fname.endswith('.npy'):
        continue

    token_path = os.path.join(TOKEN_DIR, fname)
    tokens = np.load(token_path)

    # Ensure shape [1, T]
    if tokens.ndim == 1:
        tokens = tokens[None, :]

    tokens = torch.from_numpy(tokens).long().to(device)

    with torch.no_grad():
        recon_mel = vqvae.decode(tokens)   # [1, 80, T]
        recon_mel = recon_mel.squeeze(0) 
         # [80, T]

        # --- Apply the same normalization as MelDataset ---
        recon_mel = spectral_normalize_torch(recon_mel).cpu().numpy()
        print('shape of recon mel:', recon_mel.shape)

    out_path = os.path.join(OUTPUT_DIR, fname)
    np.save(out_path, recon_mel)

print(f"Reconstructed mels saved to {OUTPUT_DIR}")