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
  python train_lm_ablation.py --cuda 0 --exp_idx 23 --wandb --focal --percep --fad --onset --mel

  # Custom weights
  python train_lm_ablation.py --cuda 0 --exp_idx 23 --wandb --focal --onset --lambda_onset 0.2
"""

import os
import subprocess
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
    mel_detail_loss,
)


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
    if args.percep:
        parts.append("percep")
    if args.fad:
        parts.append("fad")
    if args.onset:
        parts.append("onset")
    if args.mel:
        parts.append("mel")
    return "_".join(parts)


class Solver:
    def __init__(self, model, vqvae, device, hps, loss_flags):
        self.device = device
        self.model = model
        self.vqvae = vqvae
        self.loss_flags = loss_flags

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

        # Onset
        self.lambda_onset = hps.get("lambda_onset", 0.1) if loss_flags["onset"] else 0.0
        self.onset_sigma = hps.get("onset_sigma", 3.0)

        # Mel detail
        self.lambda_mel = hps.get("lambda_mel", 0.1) if loss_flags["mel"] else 0.0
        self.mel_log_weight = hps.get("mel_log_weight", 0.5)

        self.opt, self.shd, _ = get_optimizer(self.model, OPT)

    def active_losses_str(self):
        parts = []
        if self.focal_gamma > 0:
            parts.append(f"focal(g={self.focal_gamma})")
        else:
            parts.append("ce")
        if self.lambda_p > 0:
            parts.append(f"percep({self.perceptual_type},w={self.lambda_p})")
        if self.lambda_fad > 0:
            parts.append(f"fad(w={self.lambda_fad})")
        if self.lambda_onset > 0:
            parts.append(f"onset(w={self.lambda_onset},s={self.onset_sigma})")
        if self.lambda_mel > 0:
            parts.append(f"mel(w={self.lambda_mel})")
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

        _, pred = self.model(tgz, otz, ot_binfo, class_id=j_info)

        # --- CE / Focal ---
        bins = pred.shape[-1]
        ce_nats = F.cross_entropy(
            pred.reshape(-1, bins), tgz.reshape(-1), reduction="none"
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
            codebook = self.vqvae.vq.k
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
            codebook = self.vqvae.vq.k
            f_loss = fad_loss(
                pred, tgz, codebook, tau=self.tau, diagonal=self.fad_diagonal,
            )
            loss = loss + self.lambda_fad * f_loss
            logs["fad_loss"] = f_loss.item()

        # --- Onset contour ---
        if self.lambda_onset > 0:
            codebook = self.vqvae.vq.k
            o_loss = onset_contour_loss(
                pred, tgz, codebook, tau=self.tau, sigma=self.onset_sigma,
            )
            loss = loss + self.lambda_onset * o_loss
            logs["onset_loss"] = o_loss.item()

        # --- Mel detail ---
        if self.lambda_mel > 0:
            codebook = self.vqvae.vq.k
            m_loss = mel_detail_loss(
                pred, tgz, codebook, self.vqvae.decoder,
                tau=self.tau, log_weight=self.mel_log_weight,
            )
            loss = loss + self.lambda_mel * m_loss
            logs["mel_loss"] = m_loss.item()

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
    loss_group.add_argument("--mel", action="store_true", help="enable mel detail loss")

    # Weight overrides
    weight_group = parser.add_argument_group("loss weight overrides")
    weight_group.add_argument("--focal_gamma", type=float)
    weight_group.add_argument("--lambda_p", type=float)
    weight_group.add_argument("--lambda_fad", type=float)
    weight_group.add_argument("--lambda_onset", type=float)
    weight_group.add_argument("--lambda_mel", type=float)
    weight_group.add_argument("--onset_sigma", type=float)
    weight_group.add_argument("--mel_log_weight", type=float)
    weight_group.add_argument("--tau", type=float)

    args = parser.parse_args()

    # --- Setup hparams ---
    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])
    if args.bs:
        hps.batch_size = args.bs

    # Apply weight overrides to hps
    for key in ["focal_gamma", "lambda_p", "lambda_fad", "lambda_onset",
                "lambda_mel", "onset_sigma", "mel_log_weight", "tau"]:
        val = getattr(args, key, None)
        if val is not None:
            hps[key] = val

    # Defaults for keys that lm11 may not define
    hps.setdefault("tau", 0.5)
    hps.setdefault("focal_gamma", 2.0)
    hps.setdefault("lambda_p", 0.2)
    hps.setdefault("lambda_fad", 0.01)
    hps.setdefault("lambda_onset", 0.1)
    hps.setdefault("lambda_mel", 0.1)
    hps.setdefault("onset_sigma", 3.0)
    hps.setdefault("mel_log_weight", 0.5)
    hps.setdefault("fad_diagonal", True)
    hps.setdefault("perceptual_type", "advanced")
    hps.setdefault("perceptual_alpha", 0.5)
    hps.setdefault("perceptual_taus", [0.1, 0.5, 1.0])

    # --- Loss flags ---
    loss_flags = {
        "focal": args.focal,
        "percep": args.percep,
        "fad": args.fad,
        "onset": args.onset,
        "mel": args.mel,
    }

    loss_tag = build_loss_tag(args)
    ckpt_prefix = f"exp{args.exp_idx}_{loss_tag}"
    output_dir = f"/home/nikhil/jukedrummer/output_{loss_tag}"

    print(f"hps.binfo_type: {hps.binfo_type}")
    print(f"Loss config: {loss_tag}")
    print(f"Checkpoint prefix: {ckpt_prefix}")
    print(f"Output dir: {output_dir}")

    device = torch.device(f"cuda:{args.cuda}")

    model = JukeTransformer(hps).to(device)

    vqvae = VQVAE(
        codebook_size=hps.codebook_size,
        encoder=Sampler(80, 64, hps.downsample_ratios),
        decoder=Sampler(64, 80, hps.upsample_ratios),
    ).to(device)
    vqvae.restore_from_ckpt(hps, device)
    vqvae.eval()
    vqvae.requires_grad_(False)

    solver = Solver(model, vqvae, device, hps, loss_flags)
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
            subprocess.Popen(
                [
                    "python", "inference.py",
                    "--exp_idx", str(args.exp_idx),
                    "--cuda", str(args.cuda),
                    "--input_dir", "/home/nikhil/jukedrummer/data_test/audio/others",
                    "--output_dir", output_dir,
                    "--sample_iters", "1",
                    "--temp", "0.9",
                    "--top_p", "0.9",
                ]
            )
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
