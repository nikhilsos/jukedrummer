# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:16:50 2020

@author: CITI
"""

import os
from tqdm import tqdm
import torch
import librosa
import numpy as np
import pandas as pd
import argparse
from scipy.signal import find_peaks
import sys
sys.path.append('.')
from DrumAware4Beat.models.DrumAwareBeatTracker2 import DrumAwareBeatTracker as DA2
import DrumAware4Beat.da_utils as utils

from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc

def getRNNembedding(rnn, audio_fea, device, head = 'mix'):
    ## convert nparray feature into tensor
    # print('audio feature', audio_fea.shape, type(audio_fea)) # debugging
    in_fea = torch.tensor(audio_fea[np.newaxis, :, :]).float().to(device)

    rnn.eval()
    rnn.to(device)
    ## four head types: ['mix' , 'drum', 'nodrum', 'fuser']
    if head == 'mix':
        blstm_out, cellstates = rnn.mixBeat.lstm(in_fea)
        out = rnn.mixBeat.fc1(blstm_out)
    elif head =='nodrum':
        blstm_out, cellstates = rnn.nodrumBeat.lstm(in_fea)
        out = rnn.nodrumBeat.feature_fc(blstm_out)
        out = rnn.nodrumBeat.fc1(out)
    elif head =='drum':
        blstm_out, cellstates = rnn.drumBeat.lstm(in_fea)
        out = rnn.drumBeat.fc1(blstm_out)
    out_feature = blstm_out.detach().cpu().numpy().squeeze()

    # print(head,out_feature.shape, out.shape) #[1,2378,3], [2378,50]
    return out, out_feature


def time2frame4beat(beat_est, ratio, hop_length=256, sr=44100):
    '''
    Converts beat times and values to a fixed-length frame array.

    Args:
        beat_est (np.ndarray): Array of shape (N, 2), where N is the number of beats. First column is beat times (in seconds), second column is beat values.
        ratio (int): Downsampling ratio for the output frame array.
        hop_length (int, optional): Hop length for frame calculation. Default is 256.
        sr (int, optional): Sample rate. Default is 44100.

    Returns:
        np.ndarray: Array of shape (4096 // ratio,) where each index corresponding to a beat time is set to the beat value, others are zero.

    Example:
        If beat_est = [[0.5, 1], [1.0, 2]], ratio=4, the result will be a zero array of length 1024 with nonzero values at indices corresponding to the times 0.5s and 1.0s.
    '''
    times = beat_est[:,0]
    result = np.zeros(4096 // ratio)
    idxs = librosa.time_to_frames(times, sr=sr, hop_length=hop_length*ratio,)
    for idx, beat in zip(idxs, beat_est[:,1]):
        result[idx] = beat
    return result

def time2frame4onset(beat_est, ratio, hop_length=256, sr=44100):
    result = np.zeros(4096 // ratio)
    idxs = librosa.time_to_frames(beat_est, sr=sr, hop_length=hop_length*ratio,)
    for idx in idxs:
        result[idx] = 1
    return result

import os
import numpy as np
import librosa

def load_downbeat_labels(audio_file_path, ratio=4, hop_length=256, sr=44100):
    """
    Load annotated downbeats and convert to fixed-length numeric array (shape=1024).
    Returns zeros if annotation is missing or empty.
    """
    fname = os.path.basename(audio_file_path)
    downbeat_annotation = os.path.join(
        "/home/nikhil/jukedrummer/data/annots/",
        fname.replace(".wav", ".beats")
    )

    # Return zeros if annotation missing
    if not os.path.exists(downbeat_annotation):
        return np.zeros(1024, dtype=np.float32)

    beats = []
    with open(downbeat_annotation, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            t, label = parts
            if int(label) == 1:
                beats.append(float(t))

    # Return zeros if no beats found
    if len(beats) == 0:
        return np.zeros(1024, dtype=np.float32)

    # Step 1: project to internal 4096-frame grid
    base_len = 4096
    activation = np.zeros(base_len, dtype=np.float32)
    frame_idxs = librosa.time_to_frames(beats, sr=sr, hop_length=hop_length)
    frame_idxs = frame_idxs[frame_idxs < base_len]
    activation[frame_idxs] = 1.0

    # Step 2: downsample to model sample length (1024)
    activation = activation.reshape(base_len // ratio, ratio).max(axis=1)
    activation = np.array(activation, dtype=np.float32)  # enforce numeric type

    return activation  # shape = (1024,)

# --------------------------
# Beat Info Extractor
# --------------------------
class BeatInfoExtractor:
    def __init__(self, binfo_type, device, input_csv_path='src/drumaware_hmmparams.csv'):
        self.binfo_type = binfo_type
        self.device = device

        # Only load RNN/HMM if needed
        if self.binfo_type != 'dbeats':
            self.hmm_proc, self.rnn = self.get_proc(input_csv_path, device)
        else:
            self.hmm_proc, self.rnn = None, None

    def __call__(self, audio_file_path):
        if self.binfo_type == 'dbeats':
            # Directly return numeric array
            beat_info = load_downbeat_labels(audio_file_path)
            return beat_info

        # ---- normal audio-based extraction ----
        feat = utils.get_feature(audio_file_path)
        out, out_fea = getRNNembedding(
            self.rnn,
            audio_fea=feat,
            device=self.device,
            head='nodrum'
        )
        out = utils.prediction_conversion(out)

        if self.binfo_type == 'high':
            beat_est = self.hmm_proc(out)
            beat_info = time2frame4beat(beat_est, ratio=4)

        elif self.binfo_type == 'mid':
            beats_spppk_tmp, _ = find_peaks(
                np.max(out, -1),
                height=0.1,
                distance=7,
                prominence=0.1
            )
            onset_est = beats_spppk_tmp / 100
            beat_info = time2frame4onset(onset_est, ratio=4)

        elif self.binfo_type == 'low':
            beat_info = out_fea

        else:
            beat_info = None

        return beat_info


def get_proc(input_csv_path, device):
    df = pd.read_csv(input_csv_path)
    modelinfo_list = utils.df2eval_dictlist(df, withMadmom =False)
    select_model = [i for i in modelinfo_list if i['model_type']=='DA2']
    model_info = select_model[0]
    hmm_proc = DownBproc(beats_per_bar = [3, 4], min_bpm = 60, 
                             max_bpm = 200, num_tempi = model_info['n_tempi'], 
                             transition_lambda = model_info['transition_lambda'], 
                             observation_lambda = model_info['observation_lambda'], 
                             threshold = model_info['threshold'], fps = 100)
    model_setting = model_info['model_setting']
    rnn = DA2(**eval(model_setting))
    model_path = os.path.join('ckpt/' , 'RNNBeatProc.pth')
    state = torch.load(model_path, map_location=device, weights_only=True)
    rnn.load_state_dict(state)
    return hmm_proc, rnn

def inference(fns, binfo_type, audio_dir, beat_dir, n_cuda):
    print('step 4: extract beat information')
    input_csv_path='src/drumaware_hmmparams.csv'
    device = torch.device(f'cuda:{n_cuda}' if torch.cuda.is_available() else 'cpu')
    extractor = BeatInfoExtractor(binfo_type, device, input_csv_path=input_csv_path)

    for fn in tqdm(fns, desc='Extracting Beat Information'):
        ### get feature of input audio file 
        save_path = os.path.join(beat_dir, binfo_type, fn.replace('.wav', '.npy'))
        audio_file_path = os.path.join(audio_dir, 'others', fn)
        # if os.path.isfile(save_path):
        #     continue
        try:
            print(f'Processing file: {fn}')  # debugging
            beat_info = extractor(audio_file_path)
            ### save
            try:
                beat_info = np.array(beat_info, dtype=np.float32)
                np.save(save_path, beat_info)
            except Exception as e:
                    print(f"Error saving {save_path}: {e}")
        except Exception as e:
            print(f'{fn} error occur during beat information extraction, {e}')

def main(args):     
    data_dir = args.audio_dir
    binfo_type = args.binfo_type
    output_path = args.output_path
    input_csv_path = args.hparam_csv

    fns = os.listdir(os.path.join(data_dir, 'others'))
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    extractor = BeatInfoExtractor(binfo_type, device, input_csv_path=input_csv_path)
    for fn in tqdm(fns):
        ### get feature of input audio file 
        save_path = os.path.join(output_path, binfo_type, fn.replace('.wav', '.npy'))
        audio_file_path = os.path.join(data_dir, 'others', fn)
        if os.path.isfile(save_path):
            continue
        try:
            beat_info = extractor(audio_file_path)
            ### save
            np.save(save_path, beat_info)
        except:
            print(f'{fn} error occur during beat information extraction')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('binfo_type', type=str, choices=['low', 'mid', 'high'])
    parser.add_argument('audio_dir', type=str)
    parser.add_argument('output_path', type=str, default='data/beats')
    parser.add_argument('--hparam_csv', type=str, default='src/drumaware_hmmparams.csv')
    parser.add_argument('--cuda', type=int, default=0)
    arg = parser.parse_args()
    main(arg)