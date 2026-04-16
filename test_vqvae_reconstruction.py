"""
Script to test VQVAE reconstruction by loading spectrograms and comparing 
original vs reconstructed spectrograms side-by-side.

Usage:
    python test_vqvae_reconstruction.py --vq_idx 1 --num_samples 5
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob

from model.vqvae import VQVAE, Sampler
from hparams import setup_vq_hparams


def load_vqvae_model(vq_idx, data_type, device):
    """
    Load VQVAE model from checkpoint.
    
    Args:
        vq_idx: VQ model index (1, 2, or 3)
        data_type: 'target' or 'others'
        device: torch device
    
    Returns:
        model: loaded VQVAE model
        mean: normalization mean
        std: normalization std
        hps: hyperparameters
    """
    # Load checkpoint first to get the correct hps
    ckpt_name = f'vq{vq_idx}_{data_type}.pkl'
    ckpt_path = os.path.join('ckpt', ckpt_name)
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    hps_loaded = ckpt.get('hps', setup_vq_hparams(f'vq{vq_idx}'))

    # Create model using hps from checkpoint
    encoder = Sampler(input_dim=80, output_dim=64, z_scale_factors=hps_loaded['downsample_ratios'])
    decoder = Sampler(input_dim=64, output_dim=80, z_scale_factors=hps_loaded['upsample_ratios'])
    model = VQVAE(
        codebook_size=hps_loaded['codebook_size'],
        encoder=encoder,
        decoder=decoder,
        device=device
    ).to(device)

    model.load_state_dict(ckpt['model'])
    mean = torch.FloatTensor(ckpt['mean']).to(device)
    std = torch.FloatTensor(ckpt['std']).to(device)

    print(f"Loaded {ckpt_name}: codebook_size={hps_loaded.get('codebook_size', 'N/A')}")
    
    model.eval()
    return model, mean, std, hps_loaded


def load_spectrograms(data_dir, num_samples=5, data_type='others'):
    """
    Load spectrograms from the data directory.
    
    Args:
        data_dir: directory containing mel spectrograms
        num_samples: number of samples to load
        data_type: 'target' or 'others'
    
    Returns:
        spectrograms: list of numpy arrays
        filenames: list of filenames
    """
    mel_dir = os.path.join(data_dir, 'mel', data_type)
    
    if not os.path.exists(mel_dir):
        raise FileNotFoundError(f"Mel directory not found: {mel_dir}")
    
    # Get list of npy files
    npy_files = sorted(glob(os.path.join(mel_dir, '*.npy')))
    if not npy_files:
        raise FileNotFoundError(f"No mel spectrograms found in {mel_dir}")
    
    # Take first num_samples
    npy_files = npy_files[:num_samples]
    
    spectrograms = []
    filenames = []
    
    for f in npy_files:
        mel = np.load(f)
        spectrograms.append(mel)
        filenames.append(os.path.basename(f))
    
    print(f"Loaded {len(spectrograms)} spectrograms from {mel_dir}")
    return spectrograms, filenames


def normalize_spectrogram(mel, mean, std):
    """
    Normalize spectrogram using mean and std.
    
    Args:
        mel: spectrogram array (80, T)
        mean: mean tensor from checkpoint (can be various shapes)
        std: std tensor from checkpoint (can be various shapes)
    
    Returns:
        normalized spectrogram tensor (1, 80, T)
    """
    # Convert to tensor and move to same device as mean/std
    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel)
    
    # Move to device before normalization
    mel = mel.to(mean.device)
    
    # Ensure correct shape (1, 80, T)
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    
    # Reshape mean and std to (1, 80, 1) for broadcasting
    # mean/std from checkpoint could be (80,) or (1, 80, 1)
    if mean.dim() == 1:
        mean = mean.view(1, -1, 1)  # (1, 80, 1)
    elif mean.dim() == 2:
        mean = mean.unsqueeze(0)   # (1, 80, 1)
    
    if std.dim() == 1:
        std = std.view(1, -1, 1)    # (1, 80, 1)
    elif std.dim() == 2:
        std = std.unsqueeze(0)     # (1, 80, 1)
    
    # Normalize
    mel = (mel - mean) / std
    
    return mel.float()


def denormalize_spectrogram(mel, mean, std):
    """
    Denormalize spectrogram using mean and std.
    
    Args:
        mel: normalized spectrogram tensor (B, 80, T)
        mean: mean tensor from checkpoint
        std: std tensor from checkpoint
    
    Returns:
        denormalized spectrogram numpy array (B, 80, T)
    """
    # Move mean/std to same device as mel (which may be on CPU after .cpu())
    mean = mean.to(mel.device)
    std = std.to(mel.device)
    
    # Reshape mean and std to (1, 80, 1) for broadcasting
    if mean.dim() == 1:
        mean = mean.view(1, -1, 1)  # (1, 80, 1)
    elif mean.dim() == 2:
        mean = mean.unsqueeze(0)   # (1, 80, 1)
    
    if std.dim() == 1:
        std = std.view(1, -1, 1)    # (1, 80, 1)
    elif std.dim() == 2:
        std = std.unsqueeze(0)     # (1, 80, 1)
    
    # Denormalize
    mel = mel * std + mean
    
    return mel


def reconstruct_spectrograms(model, spectrograms, mean, std, device):
    """
    Reconstruct spectrograms using VQVAE model.
    
    Args:
        model: VQVAE model
        spectrograms: list of spectrogram arrays
        mean: normalization mean
        std: normalization std
        device: torch device
    
    Returns:
        reconstructed: list of reconstructed spectrogram arrays
    """
    reconstructed = []
    
    with torch.no_grad():
        for mel in spectrograms:
            # Normalize
            mel_norm = normalize_spectrogram(mel, mean, std)
            mel_norm = mel_norm.to(device)
            
            # Reconstruct
            r_mel, commit_loss, metric = model(mel_norm)
            
            # Denormalize
            r_mel_denorm = denormalize_spectrogram(r_mel.cpu(), mean, std)
            
            reconstructed.append(r_mel_denorm.squeeze(0).numpy())
    
    return reconstructed


def plot_comparison(original, reconstructed, filenames, output_path, data_type):
    """
    Plot side-by-side comparison of original and reconstructed spectrograms.
    
    Args:
        original: list of original spectrogram arrays
        reconstructed: list of reconstructed spectrogram arrays
        filenames: list of filenames
        output_path: path to save the figure
        data_type: 'target' or 'others'
    """
    n_samples = len(original)
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Compute global vmin/vmax for consistent color scaling
    all_vals = np.concatenate([orig.flatten() for orig in original] + 
                               [recon.flatten() for recon in reconstructed])
    vmin, vmax = np.percentile(all_vals, [2, 98])
    
    for i, (orig, recon, fname) in enumerate(zip(original, reconstructed, filenames)):
        # Short filename for display
        short_fname = os.path.splitext(fname)[0][:30] + "..." if len(fname) > 30 else os.path.splitext(fname)[0]
        
        # Plot original
        im1 = axes[i, 0].imshow(orig, aspect='auto', origin='lower', 
                                 cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f'{short_fname}\nOriginal ({data_type})')
        axes[i, 0].set_xlabel('Time frames')
        axes[i, 0].set_ylabel('Mel bins')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Plot reconstructed
        im2 = axes[i, 1].imshow(recon, aspect='auto', origin='lower', 
                                 cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f'Reconstructed ({data_type})')
        axes[i, 1].set_xlabel('Time frames')
        axes[i, 1].set_ylabel('Mel bins')
        plt.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to {output_path}")
    plt.close()


def plot_triple_comparison(real_drums, vqvae_recon, generated, filenames, output_path):
    """
    Three-panel comparison per sample:
      Col 0: original target mel (raw, no VQ-VAE)
      Col 1: VQ-VAE reconstruction of target (encode → decode)
      Col 2: LM-generated drums from vocal conditioning

    Args:
        real_drums:   list of np arrays (80, T) — raw target mel
        vqvae_recon:  list of np arrays (80, T) — target VQ-VAE roundtrip
        generated:    list of np arrays (80, T) — LM output
        filenames:    list of str
        output_path:  str
    """
    n = len(real_drums)
    fig, axes = plt.subplots(n, 3, figsize=(18, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    all_vals = np.concatenate(
        [m.flatten() for m in real_drums + vqvae_recon + generated]
    )
    vmin, vmax = np.percentile(all_vals, [2, 98])

    titles = ['Original drums', 'VQ-VAE reconstruction', 'LM generated']
    panels = [real_drums, vqvae_recon, generated]

    for i, fname in enumerate(filenames):
        short = os.path.splitext(fname)[0][:30]
        for col, (title, panel) in enumerate(zip(titles, panels)):
            im = axes[i, col].imshow(
                panel[i], aspect='auto', origin='lower',
                cmap='viridis', vmin=vmin, vmax=vmax
            )
            axes[i, col].set_title(f'{short}\n{title}')
            axes[i, col].set_xlabel('Time frames')
            axes[i, col].set_ylabel('Mel bins')
            plt.colorbar(im, ax=axes[i, col])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved triple comparison to {output_path}")
    plt.close()


def plot_combined_comparison(target_data, others_data, output_path):
    """
    Plot combined comparison showing target and others results side by side.
    
    Args:
        target_data: dict with keys 'spectrograms', 'reconstructed', 'filenames'
        others_data: dict with keys 'spectrograms', 'reconstructed', 'filenames'
        output_path: path to save the figure
    """
    n_samples = min(len(target_data['spectrograms']), len(others_data['spectrograms']))
    fig, axes = plt.subplots(n_samples, 4, figsize=(24, 3 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        orig_t = target_data['spectrograms'][i]
        recon_t = target_data['reconstructed'][i]
        fname_t = target_data['filenames'][i]
        
        orig_o = others_data['spectrograms'][i]
        recon_o = others_data['reconstructed'][i]
        fname_o = others_data['filenames'][i]
        
        # Calculate MSE
        mse_t = np.mean((orig_t - recon_t) ** 2)
        mse_o = np.mean((orig_o - recon_o) ** 2)
        
        # Compute vmin/vmax for each pair
        vmin_t, vmax_t = np.percentile(np.concatenate([orig_t.flatten(), recon_t.flatten()]), [2, 98])
        vmin_o, vmax_o = np.percentile(np.concatenate([orig_o.flatten(), recon_o.flatten()]), [2, 98])
        
        # Plot target original
        short_fname_t = os.path.splitext(fname_t)[0][:20] + "..." if len(fname_t) > 20 else os.path.splitext(fname_t)[0]
        im1 = axes[i, 0].imshow(orig_t, aspect='auto', origin='lower', 
                                 cmap='viridis', vmin=vmin_t, vmax=vmax_t)
        axes[i, 0].set_title(f'Target: {short_fname_t}\nOriginal')
        axes[i, 0].set_xlabel('Time frames')
        axes[i, 0].set_ylabel('Mel bins')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # Plot target reconstructed
        im2 = axes[i, 1].imshow(recon_t, aspect='auto', origin='lower', 
                                 cmap='viridis', vmin=vmin_t, vmax=vmax_t)
        axes[i, 1].set_title(f'Target Model\nMSE: {mse_t:.4f}')
        axes[i, 1].set_xlabel('Time frames')
        axes[i, 1].set_ylabel('Mel bins')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # Plot others original
        short_fname_o = os.path.splitext(fname_o)[0][:20] + "..." if len(fname_o) > 20 else os.path.splitext(fname_o)[0]
        im3 = axes[i, 2].imshow(orig_o, aspect='auto', origin='lower', 
                                 cmap='viridis', vmin=vmin_o, vmax=vmax_o)
        axes[i, 2].set_title(f'Others: {short_fname_o}\nOriginal')
        axes[i, 2].set_xlabel('Time frames')
        axes[i, 2].set_ylabel('Mel bins')
        plt.colorbar(im3, ax=axes[i, 2])
        
        # Plot others reconstructed
        im4 = axes[i, 3].imshow(recon_o, aspect='auto', origin='lower', 
                                 cmap='viridis', vmin=vmin_o, vmax=vmax_o)
        axes[i, 3].set_title(f'Others Model\nMSE: {mse_o:.4f}')
        axes[i, 3].set_xlabel('Time frames')
        axes[i, 3].set_ylabel('Mel bins')
        plt.colorbar(im4, ax=axes[i, 3])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined comparison to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test VQVAE reconstruction')
    parser.add_argument('--vq_idx', type=int, default=1, choices=[1, 2, 3],
                        help='VQ model index (1, 2, or 3)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to test')
    parser.add_argument('--data_dir', type=str, default='data_jd',
                        help='Data directory containing mel spectrograms')
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device index')
    parser.add_argument('--combined', action='store_true',
                        help='Generate combined comparison image')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Store results for combined comparison
    all_results = {}
    
    # Test for both target and others
    for data_type in ['target', 'others']:
        print(f"\n{'='*50}")
        print(f"Testing VQ{args.vq_idx} with {data_type} checkpoint")
        print(f"{'='*50}")
        
        try:
            # Load model
            model, mean, std, hps = load_vqvae_model(args.vq_idx, data_type, device)
            
            # Load spectrograms from the appropriate folder
            # When testing target checkpoint, use target spectrograms
            # When testing others checkpoint, use others spectrograms
            spectrograms, filenames = load_spectrograms(
                args.data_dir, 
                num_samples=args.num_samples,
                data_type=data_type  # Use matching data type
            )
            
            # Reconstruct
            reconstructed = reconstruct_spectrograms(model, spectrograms, mean, std, device)
            
            # Store for potential combined comparison
            all_results[data_type] = {
                'spectrograms': spectrograms,
                'reconstructed': reconstructed,
                'filenames': filenames
            }
            
            # Compute MSE loss for each sample
            print("\nReconstruction quality:")
            for i, (orig, recon) in enumerate(zip(spectrograms, reconstructed)):
                mse = np.mean((orig - recon) ** 2)
                print(f"  {filenames[i]}: MSE = {mse:.4f}")
            
            # Plot comparison
            output_path = f'vqvae_reconstruction_vq{args.vq_idx}_{data_type}.png'
            plot_comparison(spectrograms, reconstructed, filenames, output_path, data_type)
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Skipping {data_type}...")
            continue
        except Exception as e:
            print(f"Error processing {data_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Generate combined comparison if requested and both are available
    if args.combined and 'target' in all_results and 'others' in all_results:
        print(f"\n{'='*50}")
        print("Generating combined comparison...")
        print(f"{'='*50}")
        
        output_path = f'vqvae_reconstruction_vq{args.vq_idx}_combined.png'
        plot_combined_comparison(
            all_results['target'],
            all_results['others'],
            output_path
        )
    
    print("\nDone!")


if __name__ == '__main__':
    main()
