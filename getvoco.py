import os
import torch
import librosa
import numpy as np
from model.vqvae import VQVAE, Sampler
from hparams import setup_vq_hparams

# Set parameters
audio_dir = "/home/nikhil/jukedrummer/data/audio/target"  # directory with .wav files
output_dir = "/home/nikhil/jukedrummer/data/forvocoder/target_specs"  # directory to save mel .npy files
os.makedirs(output_dir, exist_ok=True)

# Load VQ-VAE model
vq_idx = 1  # set your VQ-VAE index
hps = setup_vq_hparams(f'vq{vq_idx}')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
checkpoint = torch.load(
    "/home/nikhil/jukedrummer/ckpt/vq1_target.pkl",
    map_location=device,
    weights_only=False  # restore full state_dict
)
vqvae.load_state_dict(checkpoint['model'])

# Extraction loop
for fname in os.listdir(audio_dir):
    if not fname.endswith('.wav'):
        continue
    audio_path = os.path.join(audio_dir, fname)
    y, sr = librosa.load(audio_path, sr=44100)

    # Convert to mono if needed
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Torch tensor, shape (B, 1, T)
    y_tensor = torch.FloatTensor(y).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # Forward pass through VQ-VAE
        # NOTE: this assumes your VQVAE.forward returns reconstructed mel
        recon_mel = vqvae(y_tensor)

        # If forward() returns tuple (recon, other_stuff), unpack it:
        if isinstance(recon_mel, (tuple, list)):
            recon_mel = recon_mel[0]

        mel = recon_mel.squeeze().cpu().numpy()

    # Save mel spectrogram
    out_path = os.path.join(output_dir, fname.replace('.wav', '.npy'))
    np.save(out_path, mel.astype(np.float32))
    print(f"Saved mel for {fname} to {out_path}")
