import os
import soundfile as sf
import librosa
from librosa.util import normalize
import numpy as np
import torch
from multiprocessing import Pool
import pickle

from meldataset import mel_spectrogram

def convert_file(path):
    y, sr = sf.read(path, always_2d=False)

    # ensure float32
    y = y.astype(np.float32)

    # stereo -> mono, mono untouched
    if y.ndim == 2:
        y = y.mean(axis=1)

    # normalize safely
    y = normalize(y) * 0.95

    y = torch.from_numpy(y)

    mel = mel_spectrogram(
        y.unsqueeze(0),
        n_fft=n_fft,
        num_mels=n_mel_channels,
        sampling_rate=sampling_rate,
        hop_size=hop_length,
        win_size=win_length,
        fmax=fmax,
        fmin=fmin,
    )

    mel = mel.squeeze(0).cpu().numpy().astype(np.float32)

    if np.isnan(mel).any() or np.isinf(mel).any():
        raise ValueError("NaN or Inf in mel")

    return mel


def process_audios(path):
    file_id = os.path.splitext(os.path.basename(path))[0]
    out_fp = os.path.join(base_out_dir, f"{file_id}.npy")

    if os.path.exists(out_fp):
        return file_id, True

    try:
        mel = convert_file(path)
        np.save(out_fp, mel)
        return file_id, True
    except Exception as e:
        print(f"FAILED: {path} | {e}")
        return file_id, False


if __name__ == "__main__":

    clip_dir = "/home/nikhil/jukedrummer/hifi_gan/LJSpeech-1.1/wavs"
    base_out_dir = "/home/nikhil/jukedrummer/hifi_gan/ft_dataset/mels"
    os.makedirs(base_out_dir, exist_ok=True)

    n_fft = 1024
    hop_length = 256
    win_length = 1024
    sampling_rate = 44100
    n_mel_channels = 80
    fmin = 0
    fmax = 8000

    audio_files = sorted(
        os.path.join(clip_dir, f)
        for f in os.listdir(clip_dir)
        if f.endswith(".wav")
    )

    dataset = []

    pool = Pool(processes=20)
    for file_id, ok in pool.imap_unordered(process_audios, audio_files):
        if ok:
            dataset.append(file_id)

    pool.close()
    pool.join()

    with open(os.path.join(base_out_dir, "others_dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)
