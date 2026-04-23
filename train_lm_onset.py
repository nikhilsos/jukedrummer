"""
train_lm_onset.py — Language model training with onset contour loss.

Identical to train_lm.py but adds onset_contour_loss to guide the model
toward placing drum hits at correct temporal positions.

New hparams (set in hparams.py or override via CLI):
  --lambda_onset  : weight for onset contour loss (default from hps or 0.1)
  --onset_sigma   : Gaussian smoothing width in frames (default from hps or 3.0)
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
    focal_loss,
    perceptual_loss,
    perceptual_loss_advanced,
    perceptual_loss_multiscale,
    fad_loss,
    onset_contour_loss,
    mel_detail_loss, 
)


def get_dataset(hps):
    with open(os.path.join(hps.path, "dataset.pkl"), "rb") as f:
        tr_ids, va_ids = pickle.load(f)

    # Re-split by song to prevent leakage (original split was by chunk)
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


class Solver:
    def __init__(self, model, vqvae, device, hps=None):
        self.device = device
        self.model = model
        self.vqvae = vqvae
        self.lambda_p = getattr(hps, "lambda_p", 0.0)
        self.tau = getattr(hps, "tau", 0.5)
        self.focal_gamma = getattr(hps, "focal_gamma", 0.0)

        self.perceptual_type = getattr(hps, "perceptual_type", "standard")
        self.perceptual_alpha = getattr(hps, "perceptual_alpha", 0.5)
        self.perceptual_taus = getattr(hps, "perceptual_taus", [0.1, 0.5, 1.0])

        self.lambda_fad = getattr(hps, "lambda_fad", 0.0)
        self.fad_diagonal = getattr(hps, "fad_diagonal", True)

        # Onset contour loss params
        self.lambda_onset = getattr(hps, "lambda_onset", 0.1)
        self.onset_sigma = getattr(hps, "onset_sigma", 3.0)

        # Mel detail loss params
        self.lambda_mel = getattr(hps, "lambda_mel", 0.1)
        self.mel_log_weight = getattr(hps, "mel_log_weight", 0.5)

        self.opt, self.shd, _ = get_optimizer(self.model, OPT)

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

        # Focal loss replaces the standard CE returned by the model.
        bins = pred.shape[-1]
        ce_nats = torch.nn.functional.cross_entropy(
            pred.reshape(-1, bins), tgz.reshape(-1), reduction="none"
        )
        if self.focal_gamma > 0:
            pt = torch.exp(-ce_nats)
            loss = ((1 - pt) ** self.focal_gamma * ce_nats).mean()
        else:
            loss = ce_nats.mean()
        loss = loss / np.log(2.0)  # convert nats -> bits

        if self.lambda_p > 0:
            codebook = self.vqvae.vq.k

            if self.perceptual_type == "standard":
                soft_w = torch.softmax(pred / self.tau, dim=-1)
                soft_emb = soft_w @ codebook
                target_emb = codebook[tgz]
                p_loss = F.mse_loss(soft_emb, target_emb)
            elif self.perceptual_type == "advanced":
                p_loss = perceptual_loss_advanced(
                    pred,
                    tgz,
                    codebook,
                    tau=self.tau,
                    alpha=self.perceptual_alpha,
                    use_cosine=True,
                )
            elif self.perceptual_type == "multiscale":
                p_loss = perceptual_loss_multiscale(
                    pred, tgz, codebook, taus=self.perceptual_taus
                )
            else:
                soft_w = torch.softmax(pred / self.tau, dim=-1)
                soft_emb = soft_w @ codebook
                target_emb = codebook[tgz]
                p_loss = F.mse_loss(soft_emb, target_emb)

            loss = loss + self.lambda_p * p_loss

        if self.lambda_fad > 0:
            codebook = self.vqvae.vq.k
            f_loss = fad_loss(
                pred, tgz, codebook, tau=self.tau, diagonal=self.fad_diagonal
            )
            loss = loss + self.lambda_fad * f_loss

        # Onset contour loss — temporal alignment of drum hits
        if self.lambda_onset > 0:
            codebook = self.vqvae.vq.k
            o_loss = onset_contour_loss(
                pred, tgz, codebook, tau=self.tau, sigma=self.onset_sigma
            )
            loss = loss + self.lambda_onset * o_loss

        # Mel detail loss — fine spectral reconstruction via frozen VQVAE decoder
        if self.lambda_mel > 0:
            codebook = self.vqvae.vq.k
            m_loss = mel_detail_loss(
                pred, tgz, codebook, self.vqvae.decoder,
                tau=self.tau, log_weight=self.mel_log_weight,
            )
            loss = loss + self.lambda_mel * m_loss

        if training:
            loss.backward()
            self.opt.step()
            if self.shd is not None:
                self.shd.step()

        logs = {"loss": loss.item()}

        if make_sample and use_wandb:
            with torch.no_grad():
                pred_ids = torch.argmax(pred, dim=-1)
                pred_mel = self.vqvae.decode(pred_ids[:4])
                real_mel = self.vqvae.decode(tgz[:4])

                logs["pred_mel"] = wandb.Image(
                    make_grid(pred_mel.unsqueeze(1), nrow=1)
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )
                logs["real_mel"] = wandb.Image(
                    make_grid(real_mel.unsqueeze(1), nrow=1)
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                )

        return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--exp_idx", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--bs", type=int)
    parser.add_argument("--lambda_onset", type=float, help="onset contour loss weight")
    parser.add_argument("--onset_sigma", type=float, help="Gaussian smoothing width in frames")
    parser.add_argument("--lambda_mel", type=float, help="mel detail loss weight")
    parser.add_argument("--mel_log_weight", type=float, help="log-mag L1 blend (0=linear, 1=log)")
    args = parser.parse_args()

    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])
    print(f"hps.binfo_type: {hps.binfo_type}")
    if args.bs:
        hps.batch_size = args.bs

    # Defaults for onset/mel losses (hparams may not define these)
    hps.setdefault("lambda_onset", 0.1)
    hps.setdefault("onset_sigma", 3.0)
    hps.setdefault("lambda_mel", 0.1)
    hps.setdefault("mel_log_weight", 0.5)

    # CLI overrides
    if args.lambda_onset is not None:
        hps.lambda_onset = args.lambda_onset
    if args.onset_sigma is not None:
        hps.onset_sigma = args.onset_sigma
    if args.lambda_mel is not None:
        hps.lambda_mel = args.lambda_mel
    if args.mel_log_weight is not None:
        hps.mel_log_weight = args.mel_log_weight

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

    solver = Solver(model, vqvae, device, hps=hps)

    if args.resume:
        print(
            f"You are training exp{args.exp_idx}, Resuming from checkpoint...{os.path.join(hps.ckpt_dir, f'exp{args.exp_idx}.pkl')}"
        )
        ckpt = torch.load(os.path.join(hps.ckpt_dir, f"exp{args.exp_idx}_best.pkl"))
        model.load_state_dict(ckpt["model"])
        solver.opt.load_state_dict(ckpt["opt"])
        if solver.shd and "shd" in ckpt:
            solver.shd.load_state_dict(ckpt["shd"])

    tr_loader, va_loader = get_dataset(hps)

    global_step = 0

    if args.wandb:
        import wandb

        wandb.init(
            project="JukeDrummer LM onset-aligned training",
            name=f"exp{args.exp_idx}_onset",
            config=dict(hps),
            dir="./wandb",
        )

        wandb.define_metric("step")
        wandb.define_metric("train/loss", step_metric="step")
        wandb.define_metric("valid/loss", step_metric="step")
        wandb.define_metric("lr", step_metric="step")

    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 15

    for epoch in range(OPT.epochs):
        train_loss, val_loss = 0.0, 0.0

        # -------- TRAIN --------
        for data in tqdm(tr_loader):
            logs = solver.run(
                data,
                training=True,
                make_sample=False,
                use_wandb=args.wandb,
            )

            loss_val = logs["loss"]
            train_loss += loss_val

            if args.wandb:
                wandb.log(
                    {
                        "train_loss": loss_val,
                        "lr": solver.opt.param_groups[0]["lr"],
                        "step": global_step,
                    }
                )

            global_step += 1

        train_loss /= len(tr_loader)

        # -------- VALID --------
        for data in tqdm(va_loader):
            with torch.no_grad():
                logs = solver.run(
                    data,
                    training=False,
                    make_sample=False,
                    use_wandb=args.wandb,
                )

            loss_val = logs["loss"]
            val_loss += loss_val

            if args.wandb:
                wandb.log(
                    {
                        "valid_loss": loss_val,
                        "step": global_step,
                    }
                )

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
                },
                os.path.join(hps.ckpt_dir, f"exp{args.exp_idx}_best_onset.pkl"),
            )
            print(f"  -> best val loss {best_val_loss:.4f}, saved checkpoint")
            subprocess.Popen(
                [
                    "python",
                    "inference.py",
                    "--exp_idx",
                    str(args.exp_idx),
                    "--cuda",
                    str(args.cuda),
                    "--input_dir",
                    "/home/nikhil/jukedrummer/data_test/audio/others",
                    "--output_dir",
                    "/home/nikhil/jukedrummer/output_onset",
                    "--sample_iters",
                    "1",
                    "--temp",
                    "0.9",
                    "--top_p",
                    "0.9",
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
                },
                os.path.join(hps.ckpt_dir, f"exp{args.exp_idx}_onset.pkl"),
            )
