import os
import argparse
import pickle

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from torchvision.utils import make_grid
from jukebox.jukebox.train import get_optimizer

from model.LanguageModel import JukeTransformer
from model.vqvae import VQVAE, Sampler
from dataset import *
from hparams import OPT, MODEL_LIST, setup_lm_hparams


# --------------------------------------------------
# Distributed init
# --------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)

    return device, local_rank


# --------------------------------------------------
# Dataset
# --------------------------------------------------

def get_dataset(hps, rank, world_size):

    with open(os.path.join(hps.path, "dataset.pkl"), "rb") as f:
        tr_ids, va_ids = pickle.load(f)

    tr_dataset = BeatInfoPairedDataset(tr_ids, hps)
    va_dataset = BeatInfoPairedDataset(va_ids, hps)

    tr_sampler = DistributedSampler(
        tr_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    va_sampler = DistributedSampler(
        va_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=hps.batch_size,
        sampler=tr_sampler,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    va_loader = DataLoader(
        va_dataset,
        batch_size=hps.batch_size,
        sampler=va_sampler,
        drop_last=False,
        num_workers=1,
        pin_memory=True,
    )

    return tr_loader, va_loader


# --------------------------------------------------
# Solver
# --------------------------------------------------

class Solver:

    def __init__(self, model, vqvae, device):

        self.device = device
        self.model = model
        self.vqvae = vqvae

        self.opt, self.shd, _ = get_optimizer(self.model, OPT)

    def run(self, data, training=True, make_sample=False, use_wandb=False):

        if training:
            self.model.train()
            self.opt.zero_grad()
        else:
            self.model.eval()

        tgz = data[0].long().to(self.device, non_blocking=True)
        otz = data[1].long().to(self.device, non_blocking=True)
        ot_binfo = data[2].float().to(self.device, non_blocking=True)

        loss, pred = self.model(tgz, otz, ot_binfo)

        if training:
            loss.backward()
            self.opt.step()

            if self.shd is not None:
                self.shd.step()

        logs = {"loss": loss.item()}

        return logs


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_idx", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--bs", type=int)

    args = parser.parse_args()

    # ---------------------------
    # DDP setup
    # ---------------------------

    device, local_rank = setup_ddp()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    is_main = rank == 0

    # ---------------------------
    # Hparams
    # ---------------------------

    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])

    if args.bs:
        hps.batch_size = args.bs

    # ---------------------------
    # Model
    # ---------------------------

    model = JukeTransformer(hps).to(device)

    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
    )

    # ---------------------------
    # VQVAE (frozen)
    # ---------------------------

    vqvae = VQVAE(
        codebook_size=hps.codebook_size,
        encoder=Sampler(80, 64, hps.downsample_ratios),
        decoder=Sampler(64, 80, hps.upsample_ratios),
    ).to(device)

    vqvae.restore_from_ckpt(hps, device)

    vqvae.eval()
    vqvae.requires_grad_(False)

    # ---------------------------
    # Solver
    # ---------------------------

    solver = Solver(model, vqvae, device)

    # ---------------------------
    # Resume
    # ---------------------------

    if args.resume and is_main:

        ckpt_path = os.path.join(
            hps.ckpt_dir,
            f"exp{args.exp_idx}.pkl"
        )

        print(f"Resuming from {ckpt_path}")

        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
        )

        model.module.load_state_dict(ckpt["model"])
        solver.opt.load_state_dict(ckpt["opt"])

        if solver.shd and "shd" in ckpt:
            solver.shd.load_state_dict(ckpt["shd"])

    dist.barrier()

    # ---------------------------
    # Dataset
    # ---------------------------

    tr_loader, va_loader = get_dataset(
        hps,
        rank,
        world_size,
    )

    # ---------------------------
    # WandB (rank0 only)
    # ---------------------------

    if args.wandb and is_main:

        import wandb

        wandb.init(
            project="JukeDrummer Language Model",
            name=f"exp{args.exp_idx}",
            config=dict(hps),
            dir="./wandb",
        )

        wandb.define_metric("step")
        wandb.define_metric("train_loss", step_metric="step")
        wandb.define_metric("valid_loss", step_metric="step")
        wandb.define_metric("lr", step_metric="step")

    global_step = 0

    # ---------------------------
    # Training loop
    # ---------------------------

    for epoch in range(OPT.epochs):

        tr_loader.sampler.set_epoch(epoch)

        train_loss = 0.0
        val_loss = 0.0

        # -------- TRAIN --------

        for data in tqdm(tr_loader, disable=not is_main):

            logs = solver.run(
                data,
                training=True,
            )

            loss_val = logs["loss"]

            train_loss += loss_val

            if args.wandb and is_main:

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

        for data in tqdm(va_loader, disable=not is_main):

            with torch.no_grad():

                logs = solver.run(
                    data,
                    training=False,
                )

            val_loss += logs["loss"]

        val_loss /= len(va_loader)

        # -------- LOG --------

        if is_main:

            print(
                f"{epoch:04d} | "
                f"train: {train_loss:.4f} | "
                f"valid: {val_loss:.4f}"
            )

            if args.wandb:

                wandb.log(
                    {
                        "valid_loss": val_loss,
                        "step": global_step,
                    }
                )

        # -------- SAVE --------

        if is_main and epoch % hps.sample_step == 0:

            save_path = os.path.join(
                hps.ckpt_dir,
                f"exp{args.exp_idx}.pkl"
            )

            torch.save(
                {
                    "model": model.module.state_dict(),
                    "opt": solver.opt.state_dict(),
                    "shd": solver.shd.state_dict() if solver.shd else None,
                    "hps": dict(hps),
                },
                save_path,
            )

    dist.destroy_process_group()


# --------------------------------------------------

if __name__ == "__main__":
    main()
