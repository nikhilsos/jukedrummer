import torch
import torch.nn as nn
import os
from tqdm import tqdm 
from torchvision.utils import make_grid

from dataset import *
from model.vqvae import VQVAE, Sampler
import argparse

from hparams import setup_vq_hparams, OPT
from jukebox.jukebox.train import get_optimizer
from tqdm import tqdm

def get_dataset(hps, data_type):
    with open(os.path.join(hps.path, 'dataset.pkl'), 'rb') as f:
        dataset = pickle.load(f)
    
    mean, std, non_nan = compute_mean_std(
        os.path.join(hps['path'], 'mel', args.data_type), dataset
    )

    tr_ids = dataset[0]
    va_ids = dataset[1]

    tr_dataset = MelDataset(tr_ids, hps, data_type)
    va_dataset = MelDataset(va_ids, hps, data_type)

    tr_dataloader = DataLoader(
        dataset=tr_dataset,
        batch_size=hps.batch_size,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    va_dataloader = DataLoader(
        dataset=va_dataset,
        batch_size=hps.batch_size,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    return tr_dataloader, va_dataloader, mean, std,


### Hyper Parameter Setting
parser = argparse.ArgumentParser()
parser.add_argument('--vq_idx', type=int)
parser.add_argument('--data_type', type=str, choices={'target', 'others'})
parser.add_argument('--cuda', type=int)
parser.add_argument('--wandb', action='store_true')
args = parser.parse_args()
hps = setup_vq_hparams('vq'+str(args.vq_idx))
sequence_length = 4096 // (np.prod(hps['upsample_ratios']))
device = torch.device(f'cuda:{args.cuda}')
if args.data_type == 'others':
    hps['codebook_size'] = 1024

## Model Setting
encoder = Sampler(input_dim=80, output_dim=64, z_scale_factors=hps['downsample_ratios']) 
decoder = Sampler(input_dim=64, output_dim=80, z_scale_factors=hps['upsample_ratios'])
model = VQVAE(
    codebook_size=hps['codebook_size'],
    encoder= encoder,
    decoder= decoder,
    device= device
    ).to(device)

### Data Setting

tr_dataloader, va_dataloader, mean, std = get_dataset(hps, data_type = args.data_type )

mean, std = mean.to(device), std.to(device)
### OPT
opt, shd, scalar = get_optimizer(model, OPT)
criterion = nn.MSELoss()

### WANDB
try:
    import wandb
    is_wandb = args.wandb
except ImportError:
    is_wandb = False

if is_wandb:
    run = wandb.init(
        project="JukeDrummer VQ-VAE",
        name=f"{hps.name}-{args.data_type}",
        config=dict(hps),
    )

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("valid/*", step_metric="epoch")
    wandb.define_metric("codebook/*", step_metric="epoch")


for epoch in tqdm(range(1001), desc="Epochs"):
    ############################
    # TRAIN
    ############################
    model.train()
    train_rec_losses = []
    train_commit_losses = []

    for step, mel in enumerate(tr_dataloader):
        opt.zero_grad()

        mel = mel.to(device)
        mel = (mel - mean) / std

        r_mel, commit_loss, metric = model(mel)
        reconstruct_loss = criterion(r_mel, mel)
        loss = reconstruct_loss + hps['commit_beta'] * commit_loss

        loss.backward()
        opt.step()
        shd.step()

        train_rec_losses.append(reconstruct_loss.item())
        train_commit_losses.append(commit_loss.item())

    ############################
    # VALID
    ############################
    model.eval()
    valid_rec_losses = []
    valid_commit_losses = []
    commit_ratios = []

    with torch.no_grad():
        for mel in va_dataloader:
            mel = mel.to(device)
            mel = (mel - mean) / std

            r_mel, commit_loss, metric = model(mel)
            reconstruct_loss = criterion(r_mel, mel)

            valid_rec_losses.append(reconstruct_loss.item())
            valid_commit_losses.append(commit_loss.item())
            commit_ratios.append(
                commit_loss.item() / (reconstruct_loss.item() + 1e-8)
            )

            usage = metric['usage'].item()

    ############################
    # LOGGING
    ############################
    summary = {
        "epoch": epoch,
        "train/reconstruct_loss": float(np.mean(train_rec_losses)),
        "train/commit_loss": float(np.mean(train_commit_losses)),
        "valid/reconstruct_loss": float(np.mean(valid_rec_losses)),
        "valid/commit_loss": float(np.mean(valid_commit_losses)),
        "valid/commit_ratio": float(np.mean(commit_ratios)),
        "codebook/usage": usage,
        "codebook/used_curr": metric['used_curr'].item(),
    }

    if is_wandb:
        wandb.log(summary, step=epoch)

        if epoch % 50 == 0:
            r_img = make_grid(r_mel[:4].unsqueeze(1), nrow=1)
            m_img = make_grid(mel[:4].unsqueeze(1), nrow=1)

            wandb.log({
                "reconstruct": wandb.Image(r_img),
                "real": wandb.Image(m_img),
            }, step=epoch)

            print('summary at epoch ', epoch, ': ', summary)

    ############################
    # CHECKPOINT
    ############################
    torch.save(
        {
            "model": model.state_dict(),
            "mean": mean.cpu().numpy(),
            "std": std.cpu().numpy(),
            "hps": dict(hps),
        },
        os.path.join(hps.ckpt_dir, f"{hps.name}_{args.data_type}.pkl")
    )