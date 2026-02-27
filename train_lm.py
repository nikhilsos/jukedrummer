import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pickle

from torchvision.utils import make_grid
from jukebox.jukebox.train import get_optimizer

from model.LanguageModel import JukeTransformer
from model.vqvae import VQVAE, Sampler
from dataset import *
from hparams import OPT, MODEL_LIST, setup_lm_hparams


def get_dataset(hps):
    with open(os.path.join(hps.path, 'dataset.pkl'), 'rb') as f:
        tr_ids, va_ids = pickle.load(f)

    tr_dataset = BeatInfoPairedDataset(tr_ids, hps)
    va_dataset = BeatInfoPairedDataset(va_ids, hps)

    tr_loader = DataLoader(
        tr_dataset,
        batch_size=hps.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    va_loader = DataLoader(
        va_dataset,
        batch_size=hps.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=True
    )

    return tr_loader, va_loader


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

        tgz = data[0].long().to(self.device)
        otz = data[1].long().to(self.device)
        ot_binfo = data[2].float().to(self.device)
        j_info = data[3].long().to(self.device)

        loss, pred = self.model(tgz, otz, ot_binfo, class_id=j_info) 

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
                    make_grid(pred_mel.unsqueeze(1), nrow=1).cpu().numpy().transpose(1, 2, 0)
                )
                logs["real_mel"] = wandb.Image(
                    make_grid(real_mel.unsqueeze(1), nrow=1).cpu().numpy().transpose(1, 2, 0)
                )

        return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, required=True)
    parser.add_argument("--exp_idx", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--bs", type=int)
    args = parser.parse_args()

    hps = setup_lm_hparams(MODEL_LIST[args.exp_idx])
    print(f"hps.binfo_type: {hps.binfo_type}") 
    if args.bs:
        hps.batch_size = args.bs

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

    solver = Solver(model, vqvae, device)

    if args.resume:
        print(f"You are training exp{args.exp_idx}, Resuming from checkpoint...{os.path.join(hps.ckpt_dir, f'exp{args.exp_idx}.pkl')}")
        ckpt = torch.load(os.path.join(hps.ckpt_dir, f"exp{args.exp_idx}.pkl"))
        model.load_state_dict(ckpt["model"])
        solver.opt.load_state_dict(ckpt["opt"])
        if solver.shd and "shd" in ckpt:
            solver.shd.load_state_dict(ckpt["shd"])

    tr_loader, va_loader = get_dataset(hps)

    if args.wandb:
        import wandb
        wandb.init(
            project="JukeDrummer Language Model beat and downbeat info paired training",
            name=f"exp{args.exp_idx}",
            config=dict(hps),
            dir="./wandb",
        )

        if args.wandb:
            wandb.define_metric("step")
            wandb.define_metric("train/loss", step_metric="step")
            wandb.define_metric("valid/loss", step_metric="step")
            wandb.define_metric("lr", step_metric="step")

        global_step = 0

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


        # if args.wandb:
        #     wandb.log(
        #         {
        #             "loss/train": float(train_loss),
        #             "loss/valid": float(val_loss),
        #             "lr": solver.opt.param_groups[0]["lr"],
        #         },
        #         step=epoch
        #     )

        if epoch % hps.sample_step == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": solver.opt.state_dict(),
                    "shd": solver.shd.state_dict() if solver.shd else None,
                    "hps": dict(hps),
                },
                os.path.join(hps.ckpt_dir, f"exp{args.exp_idx}_class_id.pkl"),
            )
