"""
train_lm_ablation.py — Configurable loss ablation training for JukeDrummer LM.

Each loss can be toggled on/off via CLI flags. Checkpoints and output dirs
are auto-named based on which losses are active.

Examples:
  # CE only (baseline)
  python train_lm_ablation.py --cuda 0 --exp_idx 23 --wandb

  # Focal + perceptual
  python train_lm_ablation.py --cuda 0 --exp_idx 23 --wandb --focal --percep

  # All losses
  python train_lm_ablation.py --cuda 0 --exp_idx 23 --wandb --focal --percep --fad --onset

  # Custom weights
  python train_lm_ablation.py --cuda 0 --exp_idx 23 --wandb --focal --onset --lambda_onset 0.2
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle
import torch.nn.functional as F
from torchvision.utils import make_grid
from jukebox.jukebox.train import get_optimizer

from model.LanguageModel import JukeTransformer
from model.vqvae import VQVAE, Sampler
from dataset import *
from hparams import OPT, MODEL_LIST, setup_lm_hparams
from losses import (
    perceptual_loss_advanced,
    perceptual_loss_multiscale,
    fad_loss,
    onset_contour_loss,
    onset_weighted_ce,
)


def compute_token_weights(hps, alpha=0.5):
    """Compute inverse-frequency CE weights from training token distribution.

    weight[i] = (1 / freq[i])^alpha, normalized so mean weight = 1.
    alpha=0.5 is moderate rebalancing; alpha=1.0 is full inverse frequency.
    """
    token_dir = os.path.join(hps.path, "token", "target", hps.vq_name)
    counts = np.zeros(hps.codebook_size, dtype=np.float64)
    for fn in os.listdir(token_dir):
        if not fn.endswith(".npy"):
            continue
        tokens = np.load(os.path.join(token_dir, fn)).flatten()
        for t in tokens:
            counts[t] += 1

    # Avoid division by zero for unused tokens
    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    weights = (1.0 / freq) ** alpha
    weights = weights / weights.mean()  # normalize so mean weight = 1

    print(f"Token weight range: [{weights.min():.3f}, {weights.max():.3f}] (alpha={alpha})")
    top5 = np.argsort(-counts)[:5]
    for i in top5:
        print(f"  token {i:3d}: freq={100*freq[i]:.1f}%, weight={weights[i]:.3f}")

    return torch.FloatTensor(weights)


def get_dataset(hps):
    with open(os.path.join(hps.path, "dataset.pkl"), "rb") as f:
        tr_ids, va_ids = pickle.load(f)

    import re, random

    def song_name(fn):
        m = re.match(r"^(.*?)_\d+(?:\.npy)?$", os.path.basename(fn))
        return m.group(1) if m else fn

    all_ids = tr_ids + va_ids
    songs = list({song_name(f) for f in all_ids})
    random.seed(42)
    random.shuffle(songs)
    split = int(0.8 * len(songs))
    tr_songs = set(songs[:split])
    tr_ids = [f for f in all_ids if song_name(f) in tr_songs]
    va_ids = [f for f in all_ids if song_name(f) not in tr_songs]
    print(
        f"Split by song: {len(tr_songs)} train songs ({len(tr_ids)} chunks) | "
        f"{len(songs) - split} val songs ({len(va_ids)} chunks)"
    )

    tr_dataset = BeatInfoPairedDataset(tr_ids, hps)
    va_dataset = BeatInfoPairedDataset(va_ids, hps)

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=hps.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )
    va_loader = DataLoader(
        va_dataset,
        batch_size=hps.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
    )
    return tr_loader, va_loader


def build_loss_tag(args):
    """Build a short tag string from active losses, e.g. 'focal_percep_onset'."""
    parts = ["ce"]  # CE is always on
    if args.focal:
        parts[0] = "focal"
    if args.balanced:
        parts.append("bal")
    if args.percep:
        parts.append("percep")
    if args.fad:
        parts.append("fad")
    if args.onset:
        parts.append("onset")
    if args.hit_boost:
        parts.append("hitboost")
    return "_".join(parts)


class Solver:
    def __init__(self, model, vqvae, device, hps, loss_flags, token_weights=None):
        self.device = device
        self.model = model
        self.vqvae = vqvae
        self.loss_flags = loss_flags
        self.token_weights = token_weights  # (codebook_size,) or None

        self.tau = hps.get("tau", 0.5)

        # Focal
        self.focal_gamma = hps.get("focal_gamma", 2.0) if loss_flags["focal"] else 0.0

        # Perceptual
        self.lambda_p = hps.get("lambda_p", 0.2) if loss_flags["percep"] else 0.0
        self.perceptual_type = hps.get("perceptual_type", "standard")
        self.perceptual_alpha = hps.get("perceptual_alpha", 0.5)
        self.perceptual_taus = hps.get("perceptual_taus", [0.1, 0.5, 1.0])

        # FAD
        self.lambda_fad = hps.get("lambda_fad", 0.01) if loss_flags["fad"] else 0.0
        self.fad_diagonal = hps.get("fad_diagonal", True)

        # Onset contour
        self.lambda_onset = hps.get("lambda_onset", 0.1) if loss_flags["onset"] else 0.0
        self.onset_sigma = hps.get("onset_sigma", 3.0)

        # Onset-weighted CE (replaces standard CE when active)
        self.hit_boost = hps.get("hit_boost", 3.0) if loss_flags.get("hit_boost") else 0.0
        self.hit_sigma = hps.get("hit_sigma", 2.0)


        self.opt, self.shd, _ = get_optimizer(self.model, OPT)

    def active_losses_str(self):
        parts = []
        if self.focal_gamma > 0:
            parts.append(f"focal(g={self.focal_gamma})")
        else:
            parts.append("ce")
        if self.token_weights is not None:
            parts.append("balanced")
        if self.lambda_p > 0:
            parts.append(f"percep({self.perceptual_type},w={self.lambda_p})")
        if self.lambda_fad > 0:
            parts.append(f"fad(w={self.lambda_fad})")
        if self.lambda_onset > 0:
            parts.append(f"onset(w={self.lambda_onset},s={self.onset_sigma})")
        if self.hit_boost > 0:
            parts.append(f"hit_boost({self.hit_boost}x)")
        return " + ".join(parts)

    def run(self, data, training=True, make_sample=False, use_wandb=False):
        if training:
            self.model.train()
            self.opt.zero_grad()
        else:
            self.model.eval()

        tgz = data[0].long().to(self.device)
        otz = data[1].long().to(self.device)
        ot_binfo = data[2].float().to(self.device)
        j_info = data[3].long().to(self.device)
        vocal_feat = data[4].float().to(self.device) if len(data) > 4 else None

        _, pred = self.model(tgz, otz, ot_binfo, class_id=j_info, vocal_feat=vocal_feat)

        # --- CE / Focal / Onset-weighted CE ---
        codebook = self.vqvae.vq.k
        bins = pred.shape[-1]

        if self.hit_boost > 0:
            # Onset-weighted CE: normal weight on silence, boosted on drum hits
            loss = onset_weighted_ce(
                pred, tgz, codebook,
                hit_boost=self.hit_boost, sigma=self.hit_sigma,
            )
        else:
            ce_nats = F.cross_entropy(
                pred.reshape(-1, bins), tgz.reshape(-1),
                weight=self.token_weights, reduction="none"
            )
            if self.focal_gamma > 0:
                pt = torch.exp(-ce_nats)
                loss = ((1 - pt) ** self.focal_gamma * ce_nats).mean()
            else:
                loss = ce_nats.mean()
            loss = loss / np.log(2.0)

        logs = {"ce_loss": loss.item()}

        # --- Perceptual ---
        if self.lambda_p > 0:
            if self.perceptual_type == "advanced":
                p_loss = perceptual_loss_advanced(
                    pred, tgz, codebook,
                    tau=self.tau, alpha=self.perceptual_alpha, use_cosine=True,
                )
            elif self.perceptual_type == "multiscale":
                p_loss = perceptual_loss_multiscale(
                    pred, tgz, codebook, taus=self.perceptual_taus,
                )
            else:
                soft_w = torch.softmax(pred / self.tau, dim=-1)
                soft_emb = soft_w @ codebook
                target_emb = codebook[tgz]
                p_loss = F.mse_loss(soft_emb, target_emb)
            loss = loss + self.lambda_p * p_loss
            logs["percep_loss"] = p_loss.item()

        # --- FAD ---
        if self.lambda_fad > 0:
            f_loss = fad_loss(
                pred, tgz, codebook, tau=self.tau, diagonal=self.fad_diagonal,
            )
            loss = loss + self.lambda_fad * f_loss
            logs["fad_loss"] = f_loss.item()

        # --- Onset contour ---
        if self.lambda_onset > 0:
            o_loss = onset_contour_loss(
                pred, tgz, codebook, tau=self.tau, sigma=self.onset_sigma,
            )
            loss = loss + self.lambda_onset * o_loss
            logs["onset_loss"] = o_loss.item()

        if training:
            loss.backward()
            self.opt.step()
            if self.shd is not None:
                self.shd.step()

        logs["loss"] = loss.item()

        if make_sample and use_wandb:
            with torch.no_grad():
                pred_ids = torch.argmax(pred, dim=-1)
                pred_mel = self.vqvae.decode(pred_ids[:4])
                real_mel = self.vqvae.decode(tgz[:4])
                logs["pred_mel"] = wandb.Image(
                    make_grid(pred_mel.unsqueeze(1), nrow=1)
                    .cpu().numpy().transpose(1, 2, 0)
                )
                logs["real_mel"] = wandb.Image(
                    make_grid(real_mel.unsqueeze(1), nrow=1)
                    .cpu().numpy().transpose(1, 2, 0)
                )

        return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JukeDrummer LM training with configurable loss ablations"
    )
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--exp_idx", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--bs", type=int)

    # Loss toggles
    loss_group = parser.add_argument_group("loss selection (off by default, enable with flag)")
    loss_group.add_argument("--focal", action="store_true", help="use focal CE (gamma from hparams) instead of standard CE")
    loss_group.add_argument("--percep", action="store_true", help="enable perceptual loss")
    loss_group.add_argument("--fad", action="store_true", help="enable FAD loss")
    loss_group.add_argument("--onset", action="store_true", help="enable onset contour loss")
    loss_group.add_argument("--hit_boost", action="store_true", help="onset-weighted CE: boost drum hit timesteps (replaces focal/balanced)")
    loss_group.add_argument("--balanced", action="store_true", help="class-balanced CE (inverse-frequency token weighting)")
    loss_group.add_argument("--balance_alpha", type=float, default=0.5,
                            help="rebalancing strength: 0=uniform, 0.5=moderate, 1.0=full inverse (default: 0.5)")

    # Weight overrides
    weight_group = parser.add_argument_group("loss weight overrides")
    weight_group.add_argument("--focal_gamma", type=float)
    weight_group.add_argument("--lambda_p", type=float)
    weight_group.add_argument("--lambda_fad", type=float)
    weight_group.add_argument("--lambda_onset", type=float)
    weight_group.add_argument("--onset_sigma", type=float)
    weight_group.add_argument("--hit_boost_val", type=float, help="hit boost multiplier (default 3.0)")
    weight_group.add_argument("--hit_sigma", type=float, help="smoothing sigma for hit detection (default 2.0)")
    weight_group.add_argument("--tau", type=float)

    args = parser.parse_args()

    # --- Setup hparams ---
    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])
    if args.bs:
        hps.batch_size = args.bs

    # Apply weight overrides to hps
    # Map CLI names to hps keys
    if args.hit_boost_val is not None:
        hps["hit_boost"] = args.hit_boost_val
    if args.hit_sigma is not None:
        hps["hit_sigma"] = args.hit_sigma

    for key in ["focal_gamma", "lambda_p", "lambda_fad", "lambda_onset",
                "onset_sigma", "tau"]:
        val = getattr(args, key, None)
        if val is not None:
            hps[key] = val

    # Defaults for keys that lm11 may not define
    hps.setdefault("tau", 0.5)
    hps.setdefault("focal_gamma", 2.0)
    hps.setdefault("lambda_p", 0.2)
    hps.setdefault("lambda_fad", 0.01)
    hps.setdefault("lambda_onset", 0.1)
    hps.setdefault("onset_sigma", 3.0)
    hps.setdefault("fad_diagonal", True)
    hps.setdefault("perceptual_type", "advanced")
    hps.setdefault("perceptual_alpha", 0.5)
    hps.setdefault("perceptual_taus", [0.1, 0.5, 1.0])
    hps.setdefault("hit_boost", 3.0)
    hps.setdefault("hit_sigma", 2.0)

    # --- Loss flags ---
    loss_flags = {
        "focal": args.focal,
        "percep": args.percep,
        "fad": args.fad,
        "onset": args.onset,
        "balanced": args.balanced,
        "hit_boost": args.hit_boost,
    }

    # Compute class-balanced CE weights
    device = torch.device(f"cuda:{args.cuda}")
    token_weights = None
    if args.balanced:
        token_weights = compute_token_weights(hps, alpha=args.balance_alpha).to(device)

    loss_tag = build_loss_tag(args)
    ckpt_prefix = f"exp{args.exp_idx}_{loss_tag}"
    output_dir = f"/home/nikhil/jukedrummer/output_{loss_tag}"

    print(f"hps.binfo_type: {hps.binfo_type}")
    print(f"Loss config: {loss_tag}")
    print(f"Checkpoint prefix: {ckpt_prefix}")
    print(f"Output dir: {output_dir}")

    device = torch.device(f"cuda:{args.cuda}")

    model = JukeTransformer(hps).to(device)

    latent_dim = hps.get('latent_dim', 64)
    vqvae = VQVAE(
        codebook_size=hps.codebook_size,
        encoder=Sampler(80, latent_dim, hps.downsample_ratios),
        decoder=Sampler(latent_dim, 80, hps.upsample_ratios),
        latent_dim=latent_dim,
    ).to(device)
    vqvae.restore_from_ckpt(hps, device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    solver = Solver(model, vqvae, device, hps, loss_flags, token_weights=token_weights)
    print(f"Active losses: {solver.active_losses_str()}")

    if args.resume:
        ckpt_path = os.path.join(hps.ckpt_dir, f"{ckpt_prefix}_best.pkl")
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        solver.opt.load_state_dict(ckpt["opt"])
        if solver.shd and "shd" in ckpt:
            solver.shd.load_state_dict(ckpt["shd"])

    tr_loader, va_loader = get_dataset(hps)
    global_step = 0

    if args.wandb:
        import wandb

        wandb.init(
            project="JukeDrummer LM ablations",
            name=f"{ckpt_prefix}",
            config={**dict(hps), "loss_tag": loss_tag, "loss_flags": loss_flags},
            dir="./wandb",
        )
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("valid/*", step_metric="step")

    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 15

    for epoch in range(OPT.epochs):
        train_loss, val_loss = 0.0, 0.0

        # -------- TRAIN --------
        for data in tqdm(tr_loader, desc=f"train {loss_tag}"):
            logs = solver.run(data, training=True, use_wandb=args.wandb)
            train_loss += logs["loss"]

            if args.wandb:
                wandb_logs = {"step": global_step, "lr": solver.opt.param_groups[0]["lr"]}
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        wandb_logs[f"train/{k}"] = v
                wandb.log(wandb_logs)

            global_step += 1

        train_loss /= len(tr_loader)

        # -------- VALID --------
        for data in tqdm(va_loader, desc=f"valid {loss_tag}"):
            with torch.no_grad():
                logs = solver.run(data, training=False, use_wandb=args.wandb)
            val_loss += logs["loss"]

            if args.wandb:
                wandb_logs = {"step": global_step}
                for k, v in logs.items():
                    if isinstance(v, (int, float)):
                        wandb_logs[f"valid/{k}"] = v
                wandb.log(wandb_logs)

        val_loss /= len(va_loader)

        print(f"{epoch:04d} | train: {train_loss:.4f} | valid: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": solver.opt.state_dict(),
                    "shd": solver.shd.state_dict() if solver.shd else None,
                    "hps": dict(hps),
                    "loss_tag": loss_tag,
                },
                os.path.join(hps.ckpt_dir, f"{ckpt_prefix}_best.pkl"),
            )
            print(f"  -> best val loss {best_val_loss:.4f}, saved {ckpt_prefix}_best.pkl")
        else:
            patience_counter += 1
            print(f"  -> no improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

        if epoch % hps.sample_step == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": solver.opt.state_dict(),
                    "shd": solver.shd.state_dict() if solver.shd else None,
                    "hps": dict(hps),
                    "loss_tag": loss_tag,
                },
                os.path.join(hps.ckpt_dir, f"{ckpt_prefix}_periodic.pkl"),
            )
