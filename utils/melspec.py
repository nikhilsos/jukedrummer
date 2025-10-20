# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

import pickle
import os
import librosa
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
from librosa.filters import mel as librosa_mel_fn
import torch
from torch import nn
from torch.nn import functional as F
import pickle
import argparse

import sys
sys.path.append(os.getcwd())
from hparams import MEL as MEL_HPARAMS
from functools import partial

'''
Modified from
https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py#L26
'''



class Audio2Mel(nn.Module):
    def __init__(self, hps):
        super().__init__()
        window = torch.hann_window(hps.win_length).float()
        mel_basis = librosa_mel_fn(
            hps.sampling_rate, hps.n_fft, hps.n_mel_channels, hps.mel_fmin, hps.mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.hop_length = hps.hop_length
        self.n_fft = hps.n_fft     
        self.win_length = hps.win_length
        self.window = window
    # def forward(self, audio):
    #     p = (self.n_fft - self.hop_length) // 2
    #     audio = F.pad(audio, (p, p), "reflect").squeeze(1)
    #     fft = torch.stft(
    #         audio,
    #         n_fft=self.n_fft,
    #         hop_length=self.hop_length,
    #         win_length=self.win_length,
    #         window=self.window,
    #         center=False,
    #         return_complex=False
    #     )
    #     real_part, imag_part = fft.unbind(-1)
    #     magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
    #     mel_output = torch.matmul(self.mel_basis, magnitude)
    #     log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
    #     return log_mel_spec
    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)

        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True
        )

        # Correct magnitude calculation
        magnitude = torch.abs(fft)

        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


def process_audios(fn, mel_dir, audio_dir, data_type, sample_rate, extract_func):
    ##### both others and target need to extract mel ######
    y, _ = librosa.load(os.path.join(audio_dir, data_type, fn), sr=sample_rate)
    peak = np.abs(y).max()
    if peak > 1.0:
        y /= peak
    y = torch.from_numpy(y)
    try:
        mel = extract_func(y[None, None])
        mel = mel.numpy()[0].astype(np.float32)
        if np.any(np.isnan(mel)) or np.sum(np.mean(mel, axis=0, keepdims=False) < -8) > mel.shape[1] // 2:
            return id, 0
        np.save(os.path.join(mel_dir, fn.replace('wav', 'npy')), mel, allow_pickle=False)
    except:
        print('error occur')
        return id, 0
    return fn, mel.shape[-1]

def inference(fns, audio_dir, mel_dir, process_num=4):
    print('step 2: extract Mel spectrogram')
    hps = MEL_HPARAMS
    extract_func = Audio2Mel(hps)
    sr = hps.sampling_rate
    pool = Pool(processes=process_num)

    for i, (fn, length) in enumerate(tqdm(pool.imap(
        partial(process_audios, 
                mel_dir=mel_dir,
                audio_dir=audio_dir,
                data_type='target',
                sample_rate=sr,
                extract_func=extract_func,
                ), fns)),1):
        if length == 0:
            print(f'Some error occurs, drum track of {fn} is not genereted into Mel spectrogram')
    
    for i, (fn, length) in enumerate(tqdm(pool.imap(
        partial(process_audios, 
                mel_dir=mel_dir,
                audio_dir=audio_dir,
                data_type='others',
                sample_rate=sr,
                extract_func=extract_func,
                ), fns)),1):
        if length == 0:
            print(f'Some error occurs, drumless track of {fn} is not genereted into Mel spectrogram')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, help='path of input wave audio', required=True)
    parser.add_argument('--mel_dir', type=str, help='path of output mel', required=True)
    parser.add_argument('--process_num', type=int, help='number of processor used to run for multitask pool', default=20)
    args = parser.parse_args()

    hps = MEL_HPARAMS
    extract_func = Audio2Mel(hps)
    sr = hps.sampling_rate

    # Get list of files
    audio_fns = [fn for fn in os.listdir(args.audio_dir) if fn.endswith(hps.extension)]
    audio_fns = sorted(list(audio_fns))

    # Initiate a pool
    pool = Pool(processes=args.process_num)
    dataset = []

    # Process
    for i, (fn, length) in enumerate(tqdm(pool.imap(
        partial(process_audios, 
                mel_dir=args.mel_dir,
                audio_dir=args.audio_dir,
                sample_rate=sr,
                extract_func=extract_func,
                ), audio_fns)),1):
        if length == 0:
            print(f'{fn} is not genereted into Mel spectrogram')


# import os
# # --- Reduce thread oversubscription BEFORE importing torch/numpy heavy stuff ---
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")

# import argparse
# import sys
# import traceback
# from functools import partial
# from multiprocessing import Pool

# import numpy as np
# import librosa
# from tqdm import tqdm

# import torch
# from torch import nn
# from torch.nn import functional as F
# from librosa.filters import mel as librosa_mel_fn

# # Make sure we can import hparams from CWD
# sys.path.append(os.getcwd())
# from hparams import MEL as MEL_HPARAMS


# """
# Audio2Mel module
# Modified from:
# https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py#L26
# """


# class Audio2Mel(nn.Module):
#     def __init__(self, hps):
#         super().__init__()
#         window = torch.hann_window(hps.win_length).float()

#         mel_basis = librosa_mel_fn(
#             sr=hps.sampling_rate,
#             n_fft=hps.n_fft,
#             n_mels=hps.n_mel_channels,
#             fmin=hps.mel_fmin,
#             fmax=hps.mel_fmax,
#         )

#         mel_basis = torch.from_numpy(mel_basis).float()
#         self.register_buffer("mel_basis", mel_basis)
#         self.register_buffer("window", window)

#         self.hop_length = hps.hop_length
#         self.n_fft = hps.n_fft
#         self.win_length = hps.win_length

#     @torch.no_grad()
#     def forward(self, audio: torch.Tensor):
#         """
#         audio: shape (B, 1, T) or (B, T) with float32
#         returns: log-mel spectrogram (B, n_mels, frames)
#         """
#         if audio.dim() == 3:
#             audio = audio.squeeze(1)  # (B, T)
#         elif audio.dim() != 2:
#             raise ValueError(f"Expected audio of shape (B, 1, T) or (B, T), got {audio.shape}")

#         p = (self.n_fft - self.hop_length) // 2
#         audio = F.pad(audio, (p, p), "reflect")

#         # stft -> (..., 2) because return_complex=False (for wide compatibility)
#         fft = torch.stft(
#             audio,
#             n_fft=self.n_fft,
#             hop_length=self.hop_length,
#             win_length=self.win_length,
#             window=self.window,
#             center=False,
#             return_complex=False,
#         )
#         real_part, imag_part = fft.unbind(-1)
#         magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)  # (B, n_fft//2+1, frames)

#         mel_output = torch.matmul(self.mel_basis, magnitude)      # (B, n_mels, frames)
#         log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
#         return log_mel_spec


# # ---- Global extractor created per-worker by initializer (not pickled per task) ----
# _EXTRACTOR = None
# _SAMPLE_RATE = None


# def _init_extractor(hps):
#     """Initializer runs once in each worker process."""
#     global _EXTRACTOR, _SAMPLE_RATE
#     torch.set_num_threads(1)  # keep each worker single-threaded
#     _EXTRACTOR = Audio2Mel(hps).eval()
#     _SAMPLE_RATE = hps.sampling_rate


# def _process_one(fn, *, mel_dir, audio_dir, data_type):
#     """
#     Worker function: load audio, compute mel, save .npy
#     Returns: (fn, mel_length) or (fn, 0) on failure
#     """
#     global _EXTRACTOR, _SAMPLE_RATE
#     assert _EXTRACTOR is not None, "Extractor was not initialized in worker"
#     assert _SAMPLE_RATE is not None, "Sample rate missing in worker"

#     src_path = os.path.join(audio_dir, data_type, fn)
#     out_dir = os.path.join(mel_dir, data_type)
#     os.makedirs(out_dir, exist_ok=True)

#     try:
#         # Load mono at target SR
#         y, _ = librosa.load(src_path, sr=_SAMPLE_RATE, mono=True)
#         if y is None or y.size == 0:
#             print(f"[WARN] Empty/invalid audio: {src_path}")
#             return fn, 0

#         peak = np.abs(y).max()
#         if peak > 1.0:
#             y = y / peak

#         # To torch
#         y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1,1,T)

#         mel = _EXTRACTOR(y_t).cpu().numpy()[0].astype(np.float32)  # (n_mels, frames)

#         # Basic sanity checks
#         if np.any(np.isnan(mel)):
#             print(f"[WARN] NaNs in mel for {src_path}")
#             return fn, 0

#         if np.sum(np.mean(mel, axis=0) < -8) > mel.shape[1] // 2:
#             # Too many super-low columns → probably silence/bad
#             return fn, 0

#         base, _ = os.path.splitext(fn)
#         out_path = os.path.join(out_dir, base + ".npy")
#         np.save(out_path, mel, allow_pickle=False)

#         return fn, int(mel.shape[-1])

#     except Exception as e:
#         print(f"[ERROR] Failed on {src_path}: {e}")
#         traceback.print_exc()
#         return fn, 0


# def _collect_filenames(root_dir, data_type, extension):
#     """
#     Collect files from {root_dir}/{data_type} that end with `extension`.
#     Returns a sorted list of filenames (not full paths).
#     """
#     subdir = os.path.join(root_dir, data_type)
#     if not os.path.isdir(subdir):
#         return []
#     fns = [fn for fn in os.listdir(subdir) if fn.endswith(extension)]
#     fns.sort()
#     return fns


# def run_for_datatype(data_type, audio_dir, mel_dir, hps, process_num):
#     """
#     Process one data_type subfolder (e.g., 'target' or 'others').
#     """
#     fns = _collect_filenames(audio_dir, data_type, hps.extension)
#     if not fns:
#         print(f"[INFO] No files found in {os.path.join(audio_dir, data_type)} with extension '{hps.extension}'. Skipping.")
#         return

#     print(f"[INFO] {data_type}: found {len(fns)} file(s). Extracting Mel spectrograms...")
#     worker = partial(_process_one, mel_dir=mel_dir, audio_dir=audio_dir, data_type=data_type)

#     # Use a pool with an initializer to avoid pickling the model per task
#     with Pool(processes=process_num, initializer=_init_extractor, initargs=(hps,)) as pool:
#         for fn, length in tqdm(pool.imap(worker, fns), total=len(fns), smoothing=0):
#             if length == 0:
#                 print(f"[WARN] {data_type}: '{fn}' was not generated into a Mel spectrogram.")

# def inference(fns, audio_dir, mel_dir, process_num=4, data_types=("target", "others"), hps=MEL_HPARAMS):
#     """
#     Run Mel extraction for a shared filename list across multiple data_type subfolders.
#     Assumes audio files live under {audio_dir}/{data_type}/{fn} for each data_type.

#     Returns a dict: {data_type: {"ok": count_ok, "fail": count_fail}}
#     """
#     # Ensure output subdirs exist
#     for dt in data_types:
#         os.makedirs(os.path.join(mel_dir, dt), exist_ok=True)

#     stats = {dt: {"ok": 0, "fail": 0} for dt in data_types}

#     worker = None  # filled after pool starts (partial is cheap; set per dt)

#     # One pool, two passes
#     with Pool(processes=process_num, initializer=_init_extractor, initargs=(hps,)) as pool:
#         for dt in data_types:
#             if not fns:
#                 print(f"[INFO] {dt}: no filenames provided. Skipping.")
#                 continue

#             print(f"[INFO] {dt}: extracting {len(fns)} file(s)...")
#             worker = partial(_process_one, mel_dir=mel_dir, audio_dir=audio_dir, data_type=dt)

#             for fn, length in tqdm(pool.imap(worker, fns), total=len(fns), smoothing=0):
#                 if length == 0:
#                     stats[dt]["fail"] += 1
#                     print(f"[WARN] {dt}: '{fn}' was not generated into a Mel spectrogram.")
#                 else:
#                     stats[dt]["ok"] += 1

#     # print("[INFO] Inference done:", stats)
#     return stats


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--audio_dir', type=str, required=True,
#                         help='Path with subfolders "target" and/or "others" containing input wavs')
#     parser.add_argument('--mel_dir', type=str, required=True,
#                         help='Output directory for .npy mels (subfolders will mirror input data types)')
#     parser.add_argument('--process_num', type=int, default=20,
#                         help='Number of worker processes')
#     args = parser.parse_args()

#     hps = MEL_HPARAMS

#     # Process each subfolder independently (don’t assume same filenames)
#     for data_type in ('target', 'others'):
#         run_for_datatype(data_type, args.audio_dir, args.mel_dir, hps, args.process_num)

#     print("[DONE] All requested subsets processed.")


# if __name__ == "__main__":
#     main()
