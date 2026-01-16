import os
import numpy as np
import torch
import tqdm

from hparams import setup_vq_hparams
from utils.functions import get_vqvae

DEVICE = "cuda"
assert torch.cuda.is_available()
VQ_IDX = 3
TOKEN_DIR = f"/home/nikhil/jukedrummer/data/token/target/vq{VQ_IDX}"
OUTPUT_DIR = f"/home/nikhil/jukedrummer/data/mel_from_tokens_vq{VQ_IDX}"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# load hparams
hps = setup_vq_hparams(f"vq{VQ_IDX}")

# load VQ-VAE
vqvae, _, target_mean, target_std = get_vqvae(
    vq_idx=VQ_IDX,
    data_type="target",
    ckpt_dir=hps.ckpt_dir,
    device=DEVICE,
)
vqvae.eval()


# decode tokens -> mel
for fname in tqdm.tqdm(os.listdir(TOKEN_DIR)):
    if not fname.endswith(".npy"):
        continue

    tokens = np.load(os.path.join(TOKEN_DIR, fname))
    if tokens.ndim == 1:
        tokens = tokens[None, :]  # [1, T]

    tokens = torch.from_numpy(tokens).long().to(DEVICE)

    with torch.no_grad():
        # output is already full-resolution mel
        
        mel = vqvae.decode(tokens) # shape: [1, 80, 4096]

    mel = mel * target_std + target_mean  # denormed
    mel = mel.squeeze(0).cpu().numpy() # [80, 4096]
    
    np.save(os.path.join(OUTPUT_DIR, fname), mel)
