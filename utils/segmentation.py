import os
import numpy as np
import pickle
import ffmpeg
import librosa
from tqdm import tqdm
import soundfile as sf
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor
import argparse
from beattracker_rp import beatTracker



def pad_to(audio, length):
    assert len(audio.shape) == 1, audio.shape
    if len(audio) == length: 
        return audio
    return np.pad(audio, (0,length - len(audio)), 'constant', constant_values=0)

def get_downbeats(fn, beat_proc, track_proc, root):
    '''
    Extract downbeats from the audio file using the provided beat processor and track processor.
    Args:
        fn (str): The filename of the audio file.
        beat_proc (RNNDownBeatProcessor): The processor to extract downbeats.
        track_proc (DBNDownBeatTrackingProcessor): The processor to track downbeats.
        root (str): The root directory where the audio files are located.
    Returns:
        list: A list of downbeat times.
    '''
    drums, sr = librosa.load(os.path.join(root, 'target', fn), sr = 44100)
    others, sr = librosa.load(os.path.join(root, 'others', fn), sr = 44100)
    drums = pad_to(drums, max(len(drums), len(others)))
    others = pad_to(others, max(len(drums), len(others)))
    act = beat_proc(others+drums)
    downbeats = [ t[0] for t in track_proc(act) if t[1]==1] # beat probabilities 
    # debugging
    return downbeats

from beattracker_rp import beatTracker 
def get_downbeats_pansori(fn, checkpoint_file=None, downbeats=False, root = None):
    # downbeats = beat 
    fn = os.path.join(root, 'target', fn) if root else fn
    downbeats = [beatTracker(fn, checkpoint_file=checkpoint_file, downbeats=False)]
    # print(downbeats.shape, type(downbeats)) # debugging
    return downbeats


def segmentation(fn, downbeats, length, audio_dir):
    """
    Segment audio into fixed-length chunks.

    NOTE: Added padding for segments shorter than target length to ensure consistent
    mel-spectrogram dimensions. This fixes tensor size mismatch errors during training.
    To revert: Comment out the padding sections marked with === FIX === markers.
    """
    others, sr = librosa.load(os.path.join(audio_dir, 'others', fn), sr=44100)
    drums, sr = librosa.load(os.path.join(audio_dir, 'target', fn), sr=44100)
    if not len(others) == len(drums):
        to_pad = max(len(others), len(drums)) 
        others = pad_to(others, to_pad)
        drums = pad_to(drums, to_pad)
        # print(type(downbeats))
    if downbeats == None:
        count = 0
        while(count*length+length < len(others)):
            others_s = others[count*length:count*length+length]
            drums_s = drums[count*length:count*length+length]

            # === FIX: Ensure consistent segment lengths ===
            # Pad segments that are shorter than target length to prevent variable mel-spectrogram lengths
            if len(others_s) < length:
                others_s = pad_to(others_s, length)
                drums_s = pad_to(drums_s, length)
            # === END FIX ===

            sf.write(os.path.join('data/segment_audio', 'others', f'{fn.split(".")[0]}_{count}.wav'), others_s, 44100)
            sf.write(os.path.join('data/segment_audio', 'target',  f'{fn.split(".")[0]}_{count}.wav'), drums_s, 44100)
            count += 1
    else:
        count = 0 
        # handle empty list of downbeats
        if len(downbeats) == 0:
            print(f"No downbeats detected for file: {fn}")
            return []

        # print('segmentation by downbeats', downbeats)
        start = downbeats.pop(0)
        
        while len(downbeats) != 0:
            cur = downbeats.pop(0)
            if cur - start > 24:
                start = round(start * 44100 )
                drums_s = drums[start:start+length]
                others_s = others[start:start+length]

                # === FIX: Ensure consistent segment lengths ===
                # Pad segments that are shorter than target length to prevent variable mel-spectrogram lengths
                if len(drums_s) < length:
                    drums_s = pad_to(drums_s, length)
                    others_s = pad_to(others_s, length)
                # === END FIX ===

                # handle wav and wav

                if fn.endswith('.wav'):
                    sf.write(os.path.join('data/segment_audio', 'others', f'{fn.split(".")[0]}_{count}.wav'), others_s, 44100)
                    sf.write(os.path.join('data/segment_audio', 'target',  f'{fn.split(".")[0]}_{count}.wav'), drums_s, 44100)
                elif fn.endswith('.mp3'):
                    raise NotImplementedError("MP3 format is not supported yet.")


                else:
                    raise ValueError(f"Unsupported file format: {fn}")


                start = cur
                count += 1

def inference(fns, seg_by_downbeats, length, audio_dir):
    print('step 1: data segment')
    if seg_by_downbeats:
        
        beat_proc = RNNDownBeatProcessor()
        track_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    for fn in tqdm(fns):
        if seg_by_downbeats:
            print('segmentation by downbeats')
            downbeats = get_downbeats(fn, beat_proc, track_proc, audio_dir)
        else:
            print('segmentation by hop window')
            downbeats = None
        segmentation(fn, downbeats, length, audio_dir)

def inference_pansori(fns, seg_by_downbeats, length, audio_dir):
    print('step 1: data segment')
    for fn in tqdm(fns):
        if seg_by_downbeats:
            print('segmentation by downbeats_pansori')

            downbeats = get_downbeats_pansori(fn, checkpoint_file='/home/nikhil/projects/dbtracker/offline_tcn', downbeats=False, root=audio_dir)
        else:
            print('segmentation by hop window')
            downbeats = None
        segmentation(fn, downbeats, length, audio_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_dir', type=str, required=True)
    parser.add_argument('seg_by_downbeats', type=bool, default=False)
    args = parser.parse_args()

    audio_dir = args.audio_dir
    beat_proc = RNNDownBeatProcessor()
    track_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    length = 8192 * 8 * 4 * 4

    with open(os.path.join(args.audio_dir ,'dataset.pkl'), 'rb') as f:
        fns = pickle.read(f)

    for fn in tqdm(fns):
        if args.seg_by_downbeats:
            downbeats = get_downbeats(fn, beat_proc, track_proc, audio_dir)
        else:
            downbeats = None
        segmentation(fn, downbeats, length, audio_dir)