import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
import argparse
import pickle
import numpy as np
import sys
from dataset import MelDataset, compute_mean_std
from model.vqvae import VQVAE, Sampler
from hparams import setup_vq_hparams, OPT
from jukebox.jukebox.train import get_optimizer
from icecream import ic

import torch.nn.functional as F
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vq_idx', type=int, required=True, help="Index of the VQ-VAE model")
    parser.add_argument('--data_type', type=str, choices=['target', 'others'], required=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--wandb', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)

    return parser.parse_args()



def get_dataset(hps, data_type):
    dataset_path = os.path.join(hps['path'], 'dataset.pkl')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # ic(dataset)

    # verify dataset is not empty
    if dataset_path.endswith('.pkl'):
        if not isinstance(dataset, (list, tuple)) or len(dataset) != 2:
            raise ValueError(f"Invalid dataset format in {dataset_path}")
        if len(dataset[0]) == 0 or len(dataset[1]) == 0:
            raise ValueError(f"Dataset is empty in {dataset_path}")

    # print("dataset type:", type(dataset))
    # print("dataset length:", len(dataset))

    # if isinstance(dataset, (list, tuple)):
    #     print("dataset[0] (train):", dataset[0][:5] if len(dataset[0]) > 0 else "EMPTY")
    #     print("dataset[1] (valid):", dataset[1][:5] if len(dataset[1]) > 0 else "EMPTY")

    mean, std, _ = compute_mean_std(os.path.join(hps['path'], 'mel', data_type), dataset)
    tr_ids, va_ids = dataset[0], dataset[1]

    # print(f"Train IDs count: {len(tr_ids)}")
    # print(f"Valid IDs count: {len(va_ids)}")


    

    tr_dataset = MelDataset(tr_ids, hps, data_type)
    va_dataset = MelDataset(va_ids, hps, data_type)

    # ensure the dataset is not empty
    if len(tr_dataset) == 0 or len(va_dataset) == 0:
        raise ValueError("Training or validation dataset is empty. Please check the dataset paths and contents.")
    print(f"Train dataset size: {len(tr_dataset)}, Validation dataset size: {len(va_dataset)}")
    
    def custom_collate(batch):
        max_width = 8192
        processed = []

        for x in batch:
            # Convert to tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            # Ensure float32 dtype (optional, but common)
            x = x.float()

            # Pad or truncate
            if x.shape[1] < max_width:
                x = F.pad(x, (0, max_width - x.shape[1]))
            elif x.shape[1] > max_width:
                x = x[:, :max_width]

            processed.append(x)

        return torch.stack(processed)



    tr_dataloader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=hps['batch_size'],
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        collate_fn = custom_collate
    )

    va_dataloader = torch.utils.data.DataLoader(
        dataset=va_dataset,
        batch_size=hps['batch_size'],
        num_workers=1,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        collate_fn = custom_collate
    )
    # verify the dataloaders are not empty
    if len(tr_dataloader) == 0 or len(va_dataloader) == 0:
        raise ValueError("Dataloaders are empty. Please check the dataset paths and contents.")

    return tr_dataloader, va_dataloader, mean, std


def setup_model(hps, device):
    encoder = Sampler(input_dim=80, output_dim=64, z_scale_factors=hps['downsample_ratios'])
    decoder = Sampler(input_dim=64, output_dim=80, z_scale_factors=hps['upsample_ratios'])
    model = VQVAE(
        codebook_size=hps['codebook_size'],
        encoder=encoder,
        decoder=decoder,
        device=device
    ).to(device)
    return model


def main():
    args = parse_arguments()
    hps = setup_vq_hparams(f'vq{args.vq_idx}')
    if args.data_type == 'others':
        hps['codebook_size'] = 1024

    sequence_length = 4096 // np.prod(hps['upsample_ratios'])
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    tr_loader, va_loader, mean, std = get_dataset(hps, args.data_type)
    if len(tr_loader) == 0 or len(va_loader) == 0:
        raise ValueError("Training or validation dataset is empty. Please check the dataset paths and contents.")

    print(f"Train dataset size: {len(tr_loader.dataset)}, Validation dataset size: {len(va_loader.dataset)}")
    model = setup_model(hps, device)
    opt, scheduler, scaler = get_optimizer(model, OPT)
    criterion = nn.MSELoss()

    # Optional WandB logging
    is_wandb = False
    if args.wandb:
        try:
            import wandb
            is_wandb = True
            wandb.init(
                project='JukeDrummer VQ-VAE',
                config=hps,
                dir='./wandb',
                name=f'{hps["name"]}_{args.data_type}'
            )
        except ImportError:
            print("wandb not installed. Proceeding without logging.")

    mean, std = mean.to(device), std.to(device)

    # Early stopping parameters
    patience = 25
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(1001):
        print(f"\n--- Epoch {epoch} ---")

        model.train()
        summary = {}

        train_loop = tqdm(tr_loader, desc=f"Training Epoch {epoch}", leave=False)
        for mel in train_loop:
            opt.zero_grad()
            mel = mel.to(device)
            mel = (mel - mean) / std

            r_mel, commit_loss, metric = model(mel)
            recon_loss = criterion(r_mel, mel)
            loss = commit_loss * hps['commit_beta'] + recon_loss

            loss.backward()
            opt.step()
            scheduler.step()

            summary['train_reconstruct_loss'] = recon_loss.item()
            summary['train_commit_loss'] = commit_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            valid_loop = tqdm(va_loader, desc=f"Validation Epoch {epoch}", leave=False)
            for mel in valid_loop:
                mel = mel.to(device)
                mel = (mel - mean) / std

                r_mel, commit_loss, metric = model(mel)
                recon_loss = criterion(r_mel, mel)

                summary['valid_reconstruct_loss'] = recon_loss.item()
                summary['valid_commit_loss'] = commit_loss.item()

                val_loss += recon_loss.item()  # Accumulate validation loss

                if epoch % 50 == 0:
                    mel_img = make_grid(mel[:4].unsqueeze(1), nrow=1).cpu().numpy().transpose(1, 2, 0)
                    r_mel_img = make_grid(r_mel[:4].unsqueeze(1), nrow=1).cpu().numpy().transpose(1, 2, 0)
                    if is_wandb:
                        wandb.log({
                            'real': wandb.Image(mel_img),
                            'reconstruct': wandb.Image(r_mel_img)
                        })


                        if is_wandb:
                            wandb.log({
                                'real': wandb.Image(mel_img),
                                'reconstruct': wandb.Image(r_mel_img)
                            })

        # Average validation loss
        val_loss /= len(va_loader)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")
        os.makedirs('checkpoints', exist_ok=True)

        model_dict ={
            'model': model.state_dict(),
            'mean': mean.cpu().numpy(),
            'std': std.cpu().numpy(),
            'hps': dict(hps),
        }
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset counter if improvement
            print("Validation loss improved, saving best model...")
            torch.save(
            model_dict,
            os.path.join(hps.ckpt_dir, f'{hps.name}_{args.data_type}.pkl')
        )
                         # Save the best model
        else:
            counter += 1  # Increment counter if no improvement
            print(f"No improvement. Early stopping counter: {counter}/{patience}")

        # Save the current model inside checkpoints directory

        
        torch.save(
            model_dict,
            os.path.join(hps.ckpt_dir, f'vqvae_checkpoints/current_model_epoch_{epoch}_{args.data_type}.pkl')
        )

        # remove older checkpoints to save space
        for f in os.listdir(os.path.join(hps.ckpt_dir, 'vqvae_checkpoints')):
            if f.startswith('current_model_epoch_') and f.endswith(f'_{args.data_type}.pkl'):
                epoch_num = int(f.split('_')[3])
                if epoch_num < epoch - 1:  # Keep only the last two epochs
                    os.remove(os.path.join(hps.ckpt_dir, 'vqvae_checkpoints', f))
                    print(f"Removed old checkpoint: {f}")

        # Early stopping condition
        if counter >= patience:
            print("Early stopping triggered.")
            break

        print(f"Epoch {epoch} Summary:")
        print(summary)
        if 'metric' in locals():
            print(f"Usage: {metric['usage'].item()} | Used Curr: {metric['used_curr'].item()}")
        else:
            print("Metric not available for this epoch.")


        if is_wandb:
            wandb.log(summary, step=epoch)



if __name__ == '__main__':
    main()

'''
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.utils import make_grid
import argparse
import pickle
import numpy as np
import sys
from dataset import MelDataset, compute_mean_std
from model.vqvae import VQVAE, Sampler
from hparams import setup_vq_hparams, OPT
from jukebox.jukebox.train import get_optimizer
from icecream import ic
import torch.nn.functional as F
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vq_idx', type=int, required=True, help="Index of the VQ-VAE model")
    parser.add_argument('--data_type', type=str, choices=['target', 'others'], required=True)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--wandb', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        # sys.exit(1)

    return parser.parse_args()



def get_dataset(hps, data_type):
    dataset_path = os.path.join(hps['path'], 'dataset.pkl')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    # ic(dataset)

    # verify dataset is not empty
    if dataset_path.endswith('.pkl'):
        if not isinstance(dataset, (list, tuple)) or len(dataset) != 2:
            raise ValueError(f"Invalid dataset format in {dataset_path}")
        if len(dataset[0]) == 0 or len(dataset[1]) == 0:
            raise ValueError(f"Dataset is empty in {dataset_path}")

    # print("dataset type:", type(dataset))
    # print("dataset length:", len(dataset))

    if isinstance(dataset, (list, tuple)):
        print("dataset[0] (train):", dataset[0][:5] if len(dataset[0]) > 0 else "EMPTY")
        print("dataset[1] (valid):", dataset[1][:5] if len(dataset[1]) > 0 else "EMPTY")

    mean, std, _ = compute_mean_std(os.path.join(hps['path'], 'mel', data_type), dataset)
    tr_ids, va_ids = dataset[0], dataset[1]

    print(f"Train IDs count: {len(tr_ids)}")
    print(f"Valid IDs count: {len(va_ids)}")


    

    tr_dataset = MelDataset(tr_ids, hps, data_type)
    va_dataset = MelDataset(va_ids, hps, data_type)

    # ensure the dataset is not empty
    if len(tr_dataset) == 0 or len(va_dataset) == 0:
        raise ValueError("Training or validation dataset is empty. Please check the dataset paths and contents.")
    print(f"Train dataset size: {len(tr_dataset)}, Validation dataset size: {len(va_dataset)}")
    
    def custom_collate(batch):
        max_width = 8192
        processed = []

        for x in batch:
            # Convert to tensor if needed
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)

            # Ensure float32 dtype (optional, but common)
            x = x.float()

            # Pad or truncate
            if x.shape[1] < max_width:
                x = F.pad(x, (0, max_width - x.shape[1]))
            elif x.shape[1] > max_width:
                x = x[:, :max_width]

            processed.append(x)

        return torch.stack(processed)



    tr_dataloader = torch.utils.data.DataLoader(
        dataset=tr_dataset,
        batch_size=hps['batch_size'],
        num_workers=4,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        collate_fn = custom_collate
    )

    va_dataloader = torch.utils.data.DataLoader(
        dataset=va_dataset,
        batch_size=hps['batch_size'],
        num_workers=1,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        collate_fn = custom_collate
    )
    # verify the dataloaders are not empty
    if len(tr_dataloader) == 0 or len(va_dataloader) == 0:
        raise ValueError("Dataloaders are empty. Please check the dataset paths and contents.")

    return tr_dataloader, va_dataloader, mean, std


def setup_model(hps, device):
    encoder = Sampler(input_dim=80, output_dim=64, z_scale_factors=hps['downsample_ratios'])
    decoder = Sampler(input_dim=64, output_dim=80, z_scale_factors=hps['upsample_ratios'])
    model = VQVAE(
        codebook_size=hps['codebook_size'],
        encoder=encoder,
        decoder=decoder,
        device=device
    ).to(device)
    return model


def main():
    args = parse_arguments()
    hps = setup_vq_hparams(f'vq{args.vq_idx}')
    if args.data_type == 'others':
        hps['codebook_size'] = 1024

    sequence_length = 4096 // np.prod(hps['upsample_ratios'])
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    tr_loader, va_loader, mean, std = get_dataset(hps, args.data_type)
    if len(tr_loader) == 0 or len(va_loader) == 0:
        raise ValueError("Training or validation dataset is empty. Please check the dataset paths and contents.")

    print(f"Train dataset size: {len(tr_loader.dataset)}, Validation dataset size: {len(va_loader.dataset)}")
    model = setup_model(hps, device)
    opt, scheduler, scaler = get_optimizer(model, OPT)
    criterion = nn.MSELoss()

    # Optional WandB logging
    is_wandb = False
    if args.wandb:
        try:
            import wandb
            is_wandb = True
            wandb.init(
                project='JukeDrummer VQ-VAE',
                config=hps,
                dir='./wandb',
                name=f'{hps["name"]}_{args.data_type}'
            )
        except ImportError:
            print("wandb not installed. Proceeding without logging.")

    mean, std = mean.to(device), std.to(device)

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(1001):
        print(f"\n--- Epoch {epoch} ---")

        model.train()
        summary = {}

        train_loop = tqdm(tr_loader, desc=f"Training Epoch {epoch}", leave=False)
        for mel in train_loop:
            opt.zero_grad()
            mel = mel.to(device)
            mel = (mel - mean) / std

            r_mel, commit_loss, metric = model(mel)
            recon_loss = criterion(r_mel, mel)
            loss = commit_loss * hps['commit_beta'] + recon_loss

            loss.backward()
            opt.step()
            scheduler.step()

            summary['train_reconstruct_loss'] = recon_loss.item()
            summary['train_commit_loss'] = commit_loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            valid_loop = tqdm(va_loader, desc=f"Validation Epoch {epoch}", leave=False)
            for mel in valid_loop:
                mel = mel.to(device)
                mel = (mel - mean) / std

                r_mel, commit_loss, metric = model(mel)
                recon_loss = criterion(r_mel, mel)

                summary['valid_reconstruct_loss'] = recon_loss.item()
                summary['valid_commit_loss'] = commit_loss.item()

                val_loss += recon_loss.item()  # Accumulate validation loss

                if epoch % 50 == 0:
                    mel_img = make_grid(mel[:4].unsqueeze(1), nrow=1).cpu().numpy().transpose(1, 2, 0)
                    r_mel_img = make_grid(r_mel[:4].unsqueeze(1), nrow=1).cpu().numpy().transpose(1, 2, 0)
                    if is_wandb:
                        wandb.log({
                            'real': wandb.Image(mel_img),
                            'reconstruct': wandb.Image(r_mel_img)
                        })


                        if is_wandb:
                            wandb.log({
                                'real': wandb.Image(mel_img),
                                'reconstruct': wandb.Image(r_mel_img)
                            })

        # Average validation loss
        val_loss /= len(va_loader)
        print(f"Epoch {epoch}, Validation Loss: {val_loss}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset counter if improvement
            print("Validation loss improved, saving best model...")
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        else:
            counter += 1  # Increment counter if no improvement
            print(f"No improvement. Early stopping counter: {counter}/{patience}")

        # Save the current model
        torch.save(model.state_dict(), f'current_model_epoch_{epoch}.pth')

        # Early stopping condition
        if counter >= patience:
            print("Early stopping triggered.")
            break

        print(f"Epoch {epoch} Summary:")
        print(summary)
        if 'metric' in locals():
            print(f"Usage: {metric['usage'].item()} | Used Curr: {metric['used_curr'].item()}")
        else:
            print("Metric not available for this epoch.")


        if is_wandb:
            wandb.log(summary, step=epoch)


if __name__ == '__main__':
    main()


'''