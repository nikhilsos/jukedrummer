"""
Generates data/token/dphase/{song}_{idx}.npy for every chunk in
data/token/others/vq1/, using chunk-level annotations from
data/annots/{song}_{idx}.beats.

Each output is shape (1024,) float32 — sawtooth phase in [0, 1]
ramping between consecutive downbeats.
Chunks without an annotation file get zeros.
"""

import os
import re
import numpy as np
import librosa
from tqdm import tqdm

SR         = 44100
HOP_LENGTH = 256
RATIO      = 4
EFF_HOP    = HOP_LENGTH * RATIO   # 1024 samples per output frame
N_FRAMES   = 1024                 # frames per chunk


def read_beat_times(annot_path):
    times = []
    with open(annot_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                times.append(float(parts[0]))
    return times


def times_to_phase(times, total_frames, next_downbeat_frame=None):
    """
    Sawtooth phase array of length total_frames, values in [0, 1].

    next_downbeat_frame: frame index of the first downbeat in the NEXT chunk
                         (relative to this chunk's start, so typically > total_frames).
                         Used to correctly bound the last cycle.
                         If None (end of song), the last cycle is estimated from
                         the average inter-downbeat interval, and frames beyond the
                         estimated cycle end are zeroed out.
    """
    if not times:
        return np.zeros(total_frames, dtype=np.float32)

    frame_idxs = librosa.time_to_frames(
        np.array(times), sr=SR, hop_length=EFF_HOP
    )
    frame_idxs = np.clip(frame_idxs, 0, total_frames - 1)

    # Average inter-downbeat interval (used for extrapolation)
    if len(frame_idxs) > 1:
        avg_cycle = int(np.mean(np.diff(frame_idxs)))
    else:
        avg_cycle = total_frames  # fallback: single downbeat

    # Determine where the last cycle ends
    if next_downbeat_frame is not None:
        # True next downbeat from adjacent chunk annotation
        last_cycle_end = min(next_downbeat_frame, total_frames)
        audio_ends_at  = total_frames  # audio continues into next chunk
    else:
        # End of song: estimate next downbeat from avg cycle
        estimated_next = frame_idxs[-1] + avg_cycle
        last_cycle_end = min(estimated_next, total_frames)
        audio_ends_at  = last_cycle_end  # zero out frames beyond estimated end

    phase = np.zeros(total_frames, dtype=np.float32)

    # Frames before the first downbeat: extrapolate backward
    if frame_idxs[0] > 0:
        pre_len     = frame_idxs[0]
        cycle       = (frame_idxs[1] - frame_idxs[0]) if len(frame_idxs) > 1 else avg_cycle
        start_phase = max(0.0, 1.0 - pre_len / max(cycle, 1))
        phase[:pre_len] = np.linspace(start_phase, 1.0, pre_len, endpoint=False)

    for i in range(len(frame_idxs)):
        start = frame_idxs[i]
        end   = frame_idxs[i + 1] if i + 1 < len(frame_idxs) else last_cycle_end
        end   = min(end, total_frames)
        if end > start:
            phase[start:end] = np.linspace(0.0, 1.0, end - start, endpoint=False)

    # Zero out frames after audio ends (silence / zero-padding)
    if audio_ends_at < total_frames:
        phase[audio_ends_at:] = 0.0

    return phase


def main():
    token_dir = 'data/token/others/vq1'
    annot_dir = 'data/annots'
    out_dir   = 'data/token/dphase'
    os.makedirs(out_dir, exist_ok=True)

    # Collect all chunks
    chunks = []
    for fn in os.listdir(token_dir):
        if not fn.endswith('.npy'):
            continue
        m = re.match(r'^(.*?)_(\d+)\.npy$', fn)
        if m:
            chunks.append((m.group(1), int(m.group(2))))

    print(f'{len(chunks)} chunks')

    # Build a set of all known chunks for fast next-chunk lookup
    chunk_set = {(s, i) for s, i in chunks}

    missing = 0
    for song, idx in tqdm(chunks):
        out_path = os.path.join(out_dir, f'{song}_{idx}.npy')
        if os.path.exists(out_path):
            continue

        # chunk-level annotation: times are relative to chunk start
        annot_path = os.path.join(annot_dir, f'{song}_{idx}.beats')
        if os.path.exists(annot_path):
            times = read_beat_times(annot_path)
        else:
            times = []
            missing += 1

        # Look up first downbeat of next chunk to correctly bound the last cycle
        next_downbeat_frame = None
        next_annot = os.path.join(annot_dir, f'{song}_{idx + 1}.beats')
        if (song, idx + 1) in chunk_set and os.path.exists(next_annot):
            next_times = read_beat_times(next_annot)
            if next_times:
                # Convert next chunk's first downbeat to a frame index relative
                # to this chunk's end (i.e., offset by N_FRAMES)
                next_frame = librosa.time_to_frames(
                    next_times[0], sr=SR, hop_length=EFF_HOP
                )
                next_downbeat_frame = N_FRAMES + int(next_frame)

        phase = times_to_phase(times, N_FRAMES, next_downbeat_frame=next_downbeat_frame)
        np.save(out_path, phase)

    print(f'Done. {missing} chunks had no annotation (written as zeros).')


if __name__ == '__main__':
    main()
