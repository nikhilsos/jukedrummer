#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import traceback
from functools import partial
from multiprocessing import Pool

import numpy as np
import librosa
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from librosa.filters import mel as librosa_mel_fn

import sys
sys.path.append(os.getcwd())
from hparams import MEL as MEL_HPARAMS


# ============================================================
# Audio2Mel Module
# ============================================================

class Audio2Mel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        window = torch.hann_window(hps.win_length).float()
        mel_basis = librosa_mel_fn(
            sr=hps.sampling_rate,
            n_fft=hps.n_fft,
            n_mels=hps.n_mel_channels,
            fmin=hps.mel_fmin,
            fmax=hps.mel_fmax,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)

        self.hop_length = hps.hop_length
        self.n_fft = hps.n_fft
        self.win_length = hps.win_length

    @torch.no_grad()
    def forward(self, audio: torch.Tensor):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        elif audio.dim() != 2:
            raise ValueError(f"Unexpected audio shape: {audio.shape}")

        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect")

        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=False,
        )
        real, imag = fft.unbind(-1)
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


# ============================================================
# Worker Logic
# ============================================================

_EXTRACTOR = None
_SR = None

def _init_extractor(hps):
    global _EXTRACTOR, _SR
    torch.set_num_threads(1)
    _EXTRACTOR = Audio2Mel(hps).eval()
    _SR = hps.sampling_rate


def _process_one(fn, *, mel_dir, audio_dir, data_type):
    global _EXTRACTOR, _SR
    assert _EXTRACTOR is not None and _SR is not None

    src_path = os.path.join(audio_dir, data_type, fn)
    out_dir = os.path.join(mel_dir, data_type)
    os.makedirs(out_dir, exist_ok=True)

    try:
        y, _ = librosa.load(src_path, sr=_SR, mono=True)
        if y.size == 0:
            print(f"[WARN] Empty audio: {src_path}")
            return fn, 0

        peak = np.abs(y).max()
        if peak > 1.0:
            y = y / peak

        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mel = _EXTRACTOR(y_t).cpu().numpy()[0].astype(np.float32)

        if np.any(np.isnan(mel)):
            print(f"[WARN] NaNs in mel for {src_path}")
            return fn, 0

        if np.sum(np.mean(mel, axis=0) < -8) > mel.shape[1] // 2:
            return fn, 0

        base, _ = os.path.splitext(fn)
        out_path = os.path.join(out_dir, base + ".npy")
        np.save(out_path, mel, allow_pickle=False)
        return fn, mel.shape[-1]

    except Exception as e:
        print(f"[ERROR] Failed on {src_path}: {e}")
        traceback.print_exc()
        return fn, 0


# ============================================================
# Inference Function
# ============================================================

def inference(fns, audio_dir, mel_dir, process_num=4, data_types=("target", "others"), hps=MEL_HPARAMS):
    """
    External entry point.
    Extract mel spectrograms for the provided filenames across data types.

    Args:
        fns: list of filenames (e.g., ['song1.wav', 'song2.wav'])
        audio_dir: base path containing subfolders for each data_type
        mel_dir: output directory for mel spectrograms
        process_num: number of worker processes
        data_types: iterable of subfolders to process
        hps: hyperparameters (MEL_HPARAMS)
    """
    for dt in data_types:
        os.makedirs(os.path.join(mel_dir, dt), exist_ok=True)

    stats = {dt: {"ok": 0, "fail": 0} for dt in data_types}

    worker = None
    with Pool(processes=process_num, initializer=_init_extractor, initargs=(hps,)) as pool:
        for dt in data_types:
            if not fns:
                print(f"[INFO] {dt}: no filenames given. Skipping.")
                continue

            print(f"[INFO] {dt}: processing {len(fns)} files")
            worker = partial(_process_one, mel_dir=mel_dir, audio_dir=audio_dir, data_type=dt)

            for fn, length in tqdm(pool.imap(worker, fns), total=len(fns), smoothing=0):
                if length == 0:
                    stats[dt]["fail"] += 1
                else:
                    stats[dt]["ok"] += 1

    return stats


# ============================================================
# Command-Line Entry Point
# ============================================================

def _collect_filenames(root_dir, extension):
    fns = [fn for fn in os.listdir(root_dir) if fn.endswith(extension)]
    fns.sort()
    return fns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Root audio directory with subfolders "target"/"others"')
    parser.add_argument('--mel_dir', type=str, required=True,
                        help='Output directory for .npy mel spectrograms')
    parser.add_argument('--process_num', type=int, default=8,
                        help='Number of worker processes')
    args = parser.parse_args()

    hps = MEL_HPARAMS
    fns = _collect_filenames(os.path.join(args.audio_dir, "target"), hps.extension)

    print(f"[INFO] Found {len(fns)} files to process.")
    stats = inference(fns, args.audio_dir, args.mel_dir, args.process_num, data_types=("target", "others"), hps=hps)
    print("[DONE] Inference complete:", stats)


if __name__ == "__main__":
    main()
