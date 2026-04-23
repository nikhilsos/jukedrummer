"""
Verification script to compare LM predictions vs ground truth.

Tests:
1. Argmax (greedy) - best possible token prediction
2. Low temperature - near-greedy sampling
3. High temperature - random sampling

Shows token-level accuracy at each level.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse

from model.LanguageModel import JukeTransformer
from model.vqvae import VQVAE, Sampler
from dataset import *
from hparams import OPT, MODEL_LIST, setup_lm_hparams


def compute_token_accuracy(pred_tokens, target_tokens):
    """Token-level accuracy."""
    return (pred_tokens == target_tokens).float().mean().item()


def compute_topk_accuracy(pred_logits, target_tokens, k=5):
    """Top-k accuracy."""
    _, topk_preds = pred_logits.topk(k, dim=-1)
    target_expanded = target_tokens.unsqueeze(-1).expand_as(topk_preds)
    return (topk_preds == target_expanded).any(dim=-1).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--exp_idx", type=int, default=23)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda}")
    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])

    print(f"Testing exp{args.exp_idx}")
    print(f"perceptual_type: {getattr(hps, 'perceptual_type', 'standard')}")
    print(f"lambda_p: {getattr(hps, 'lambda_p', 0.0)}")
    print(f"tau: {getattr(hps, 'tau', 0.5)}")

    # VQVAE
    vqvae = VQVAE(
        codebook_size=hps.codebook_size,
        encoder=Sampler(80, 64, hps.downsample_ratios),
        decoder=Sampler(64, 80, hps.upsample_ratios),
    ).to(device)
    vqvae.restore_from_ckpt(hps, device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    # Language Model
    lm = JukeTransformer(hps).to(device)

    # Try to load checkpoint
    ckpt_path = os.path.join(hps.ckpt_dir, f"exp{args.exp_idx}_best.pkl")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(hps.ckpt_dir, f"exp{args.exp_idx}_best_fad.pkl")

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=lambda s, l: s)
        lm.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded: {ckpt_path}")
    else:
        print(f"No checkpoint found at {ckpt_path}")
        return

    # Dataset
    with open(os.path.join(hps.path, "dataset.pkl"), "rb") as f:
        tr_ids, va_ids = pickle.load(f)

    dataset = BeatInfoPairedDataset(va_ids[: args.n_samples], hps)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("\n" + "=" * 60)
    print("Token-level accuracy:")
    print("=" * 60)

    # Single pass for token metrics
    token_accs = []
    top5_accs = []
    top1_probs = []

    for i, data in enumerate(loader):
        tgz = data[0].long().to(device)
        otz = data[1].long().to(device)
        ot_binfo = data[2].float().to(device)
        j_info = data[3].long().to(device)

        with torch.no_grad():
            _, pred_logits = lm(tgz, otz, ot_binfo, class_id=j_info)

            pred_tokens = torch.argmax(pred_logits, dim=-1)
            token_accs.append(compute_token_accuracy(pred_tokens, tgz))
            top5_accs.append(compute_topk_accuracy(pred_logits, tgz, k=5))

            # Flatten for correct indexing
            probs_flat = pred_logits.view(-1, pred_logits.shape[-1])  # [B*T, C]
            targets_flat = tgz.view(-1)  # [B*T]
            probs_at_target = (
                probs_flat[torch.arange(len(targets_flat)).to(device), targets_flat]
                .mean()
                .item()
            )
            top1_probs.append(probs_at_target)

    print(f"Token Accuracy (argmax):  {np.mean(token_accs) * 100:.1f}%")
    print(f"Top-5 Accuracy:         {np.mean(top5_accs) * 100:.1f}%")
    print(f"Avg prob at target:     {np.mean(top1_probs) * 100:.1f}%")

    # Temperature doesn't affect argmax - only sampling
    print("\n" + "=" * 60)
    print("Mel-space reconstruction (this is what matters):")
    print("=" * 60)

    lm_mse_losses = []
    target_norms = []

    for i, data in enumerate(loader):
        tgz = data[0].long().to(device)
        otz = data[1].long().to(device)
        ot_binfo = data[2].float().to(device)
        j_info = data[3].long().to(device)

        with torch.no_grad():
            _, pred_logits = lm(tgz, otz, ot_binfo, class_id=j_info)
            pred_tokens = torch.argmax(pred_logits, dim=-1)

            target_mel = vqvae.decode(tgz)
            pred_mel = vqvae.decode(pred_tokens)

            mse = ((target_mel - pred_mel) ** 2).mean().item()
            lm_mse_losses.append(mse)
            target_norms.append((target_mel**2).mean().item())

    avg_mse = np.mean(lm_mse_losses)
    avg_target_var = np.mean(target_norms)
    print(f"LM → Target mel MSE: {avg_mse:.6f}")
    print(f"Target mel variance: {avg_target_var:.6f}")
    print(f"Normalized MSE: {avg_mse / avg_target_var:.3f}")

    # Test VQVAE baseline (decode target tokens vs decode same tokens = should be identical)
    print("\n" + "=" * 60)
    print("VQVAE deterministic decode baseline:")
    print("=" * 60)

    decode_diffs = []
    for i, data in enumerate(loader):
        tgz = data[0].long().to(device)
        with torch.no_grad():
            mel1 = vqvae.decode(tgz)
            mel2 = vqvae.decode(tgz)
            diff = ((mel1 - mel2) ** 2).mean().item()
            decode_diffs.append(diff)
    print(f"Decode determinism check: {np.mean(decode_diffs):.8f} (should be 0)")

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print("Key metrics:")
    print(f"  - Token accuracy (argmax): {np.mean(token_accs) * 100:.1f}%")
    print(f"  - Mel-space NMSE: {avg_mse / avg_target_var:.3f}")
    print("  - If NMSE > 0.1, there's significant mel-space deviation")


if __name__ == "__main__":
    main()
