import os
import numpy as np
import pickle
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
    act = beat_proc(others+drums)
    downbeats = [ t[0] for t in track_proc(act) if t[1]==1] # beat probabilities 
    # debugging
    return downbeats

def get_downbeats_pansori(fn, checkpoint_file=None, downbeats=False, root = None):
    # downbeats = beat
    fn = os.path.join(root, 'target', fn) if root else fn
    beat_results = [beatTracker(fn, checkpoint_file=checkpoint_file, downbeats=False)]
    


    # print(beat_results.shape, type(beat_results)) # debugging
    return beat_results


def segmentation(fn, downbeats, length, audio_dir):
    os.makedirs('data/segment_audio/others', exist_ok=True)
    os.makedirs('data/segment_audio/target', exist_ok=True)

    others, sr = librosa.load(os.path.join(audio_dir, 'others', fn), sr=44100)
    drums, sr = librosa.load(os.path.join(audio_dir, 'target', fn), sr=44100)

    # equalize source lengths
    if len(others) != len(drums):
        to_pad = max(len(others), len(drums))
        others = pad_to(others, to_pad)
        drums = pad_to(drums, to_pad)

    def write_segment(others_s, drums_s, count):
        # HARD invariant
        others_s = pad_to(others_s, length)
        drums_s = pad_to(drums_s, length)

        assert len(others_s) == length
        assert len(drums_s) == length

        base = fn.split(".")[0]
        sf.write(f'data/segment_audio/others/{base}_{count}.wav', others_s, 44100)
        sf.write(f'data/segment_audio/target/{base}_{count}.wav', drums_s, 44100)

    # -------------------------
    # hop-window segmentation
    # -------------------------
    if downbeats is None:
        count = 0
        pos = 0
        while pos < len(others):
            others_s = others[pos:pos + length]
            drums_s = drums[pos:pos + length]
            write_segment(others_s, drums_s, count)
            pos += length
            count += 1
        return

    # -------------------------
    # downbeat-based segmentation
    # -------------------------
    if len(downbeats) == 0:
        return

    count = 0
    downbeats = list(downbeats)

    for start in downbeats:
        start_samples = int(round(start * 44100))
        if start_samples >= len(drums):
            continue
        others_s = others[start_samples:start_samples + length]
        drums_s = drums[start_samples:start_samples + length]
        write_segment(others_s, drums_s, count)
        count += 1

def inference(fns, seg_by_downbeats, length, audio_dir):
    print('step 1: data segment')
    if seg_by_downbeats:
        
        beat_proc = RNNDownBeatProcessor()
        track_proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)

    printed = False
    
    for fn in tqdm(fns):
        if seg_by_downbeats:
            if not printed:
                print('segmentation by downbeats')
                printed = True
            downbeats = get_downbeats(fn, beat_proc, track_proc, audio_dir)
        else:
            if not printed:
                print('segmentation by hop window')
                printed = True
            downbeats = None
            
        segmentation(fn, downbeats, length, audio_dir)

def inference_pansori(fns, seg_by_downbeats, length, audio_dir):
    print('step 1: data segment')
    for fn in tqdm(fns, desc="Segmenting files"):
        if seg_by_downbeats:
            print('segmentation by downbeats_pansori')

            downbeats = get_downbeats_pansori(fn, checkpoint_file='/home/nikhil/jukedrummer/offline_tcn', downbeats=False, root=audio_dir)
            # convert the list to list of float64s
            downbeats = downbeats[0]
            downbeats = [float(db) for db in downbeats]
           

        else:
            print('segmentation by hop window')
            downbeats = None

        # Debugging line to check downbeats
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