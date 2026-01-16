import os
import argparse
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from librosa.filters import mel as librosa_mel_fn

import sys
sys.path.append(os.getcwd())
from hparams import MEL as MEL_HPARAMS


# -------------------------
# Mel extractor
# -------------------------
class Audio2Mel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        window = torch.hann_window(hps.win_length)
        mel_basis = librosa_mel_fn(
            hps.sampling_rate,
            hps.n_fft,
            hps.n_mel_channels,
            hps.mel_fmin,
            hps.mel_fmax,
        )

        self.register_buffer("window", window)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

        self.n_fft = hps.n_fft
        self.hop_length = hps.hop_length
        self.win_length = hps.win_length

    def forward(self, audio):
        # audio: (B, 1, T)
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), mode="reflect").squeeze(1)

        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True,
        )

        mag = spec.abs()
        mel = torch.matmul(self.mel_basis, mag)
        mel = torch.log10(torch.clamp(mel, min=1e-5))
        return mel

# -------------------------
# Main
# -------------------------
def main(args):
    hps = MEL_HPARAMS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = Audio2Mel(hps).to(device)
    extractor.eval()

    os.makedirs(args.mel_dir, exist_ok=True)

    audio_files = sorted(
        fn for fn in os.listdir(args.audio_dir)
        if fn.endswith(hps.extension)
    )

    for fn in tqdm(audio_files, total=len(audio_files)):
        path = os.path.join(args.audio_dir, fn)

        try:
            y, sr = sf.read(path, dtype="float32")
            if sr != hps.sampling_rate:
                raise RuntimeError("Sample rate mismatch")

            peak = np.abs(y).max()
            if peak > 1.0:
                y /= peak

            y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                mel = extractor(y)[0].cpu().numpy().astype(np.float32)

            if np.any(np.isnan(mel)):
                continue

            out_path = os.path.join(
                args.mel_dir, fn.replace(".wav", ".npy")
            )
            np.save(out_path, mel, allow_pickle=False)

        except Exception as e:
            print(f"FAILED: {fn} | {e}")

def inference(fns, audio_dir, mel_dir):
    hps = MEL_HPARAMS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    extractor = Audio2Mel(hps).to(device)
    extractor.eval()

    with torch.no_grad():
        for split in ["target", "others"]:
            out_dir = os.path.join(mel_dir, split)
            os.makedirs(out_dir, exist_ok=True)

            for fn in tqdm(fns, total=len(fns), desc=f"Mel [{split}]"):
                try:
                    path = os.path.join(audio_dir, split, fn)

                    y, sr = sf.read(path)
                    if sr != hps.sampling_rate:
                        continue

                    y = y.astype(np.float32)
                    if y.ndim == 2:
                        y = y.mean(axis=1)

                    peak = np.abs(y).max()
                    if peak > 1.0:
                        y /= peak

                    y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).to(device)

                    mel = extractor(y)[0].cpu().numpy().astype(np.float32)
                    if np.any(np.isnan(mel)):
                        continue

                    base = os.path.splitext(fn)[0]
                    np.save(
                        os.path.join(out_dir, base + ".npy"),
                        mel,
                        allow_pickle=False,
                    )

                except Exception as e:
                    print(f"FAILED [{split}]:", fn, e)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True)
    parser.add_argument("--mel_dir", required=True)
    args = parser.parse_args()

    main(args)
