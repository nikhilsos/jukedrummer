import os
import math
import random
import torch
import torch.utils.data
import numpy as np
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(path):
    sr, audio = read(path)
    return audio, sr


class VQMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        audio_files,
        mel_dir,
        segment_size,
        hop_size,
        sampling_rate,
        split=True,
        shuffle=True,
    ):
        self.audio_files = audio_files
        self.mel_dir = mel_dir
        self.segment_size = segment_size
        self.hop_size = hop_size
        self.sampling_rate = sampling_rate
        self.split = split

        if shuffle:
            random.shuffle(self.audio_files)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        wav_path = self.audio_files[idx]
        fname = os.path.splitext(os.path.basename(wav_path))[0]

        # ---- load audio ----
        audio, sr = load_wav(wav_path)
        if sr != self.sampling_rate:
            raise ValueError(f"SR mismatch: {sr} != {self.sampling_rate}")

        audio = audio.astype(np.float32) / MAX_WAV_VALUE
        audio = torch.from_numpy(audio)

        # ---- load VQ-VAE mel ----
        mel = np.load(os.path.join(self.mel_dir, fname + ".npy"))
        mel = torch.from_numpy(mel).float()  # [80, T]

        if mel.dim() != 2:
            raise RuntimeError("Mel must be [80, T]")

        # ---- random segment ----
        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)

            if mel.size(1) > frames_per_seg:
                mel_start = random.randint(0, mel.size(1) - frames_per_seg)
                mel = mel[:, mel_start:mel_start + frames_per_seg]
                audio = audio[
                    mel_start * self.hop_size :
                    (mel_start + frames_per_seg) * self.hop_size
                ]
            else:
                mel = torch.nn.functional.pad(
                    mel, (0, frames_per_seg - mel.size(1))
                )
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.size(0))
                )

        return mel, audio, fname
