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
from model.downbeat_model import BeatNetOffline
from madmom.features.downbeats import DBNDownBeatTrackingProcessor as DownBproc
# constants
HOP_LENGTH = 220
SR = 22500
FPS = 10


def load_checkpoint(model, checkpoint_file):
    """
    Restores a model to a given checkpoint, but loads directly to CPU, allowing
    model to be run on non-CUDA devices.
    """    
    model.load_state_dict(
        torch.load(checkpoint_file, map_location=torch.device('cpu')))
    
model = BeatNetOffline()
model.eval()
    
def getTCNembeddings(model, audio_fea, device, head = 'mix'):
    ## convert nparray feature into tensor
    in_fea = torch.tensor(audio_fea[np.newaxis, :, :]).float().to(device)
    model.eval()
    model.to(device)
    ## four head types: ['mix' , 'drum', 'nodrum', 'fuser']
    beat_output, rhythm_output, out_feature = model(in_fea)
    return beat_output, rhythm_output, out_feature


def time2frame4beat(beat_est, ratio, hop_length=HOP_LENGTH, sr=SR):
    '''Convert beat estimation from time to frame representation.'''
    times = beat_est[:,0]
    result = np.zeros(4096 // ratio)
    idxs = librosa.time_to_frames(times, sr=sr, hop_length=hop_length*ratio,)
    for idx, beat in zip(idxs, beat_est[:,1]):
        result[idx] = beat
    return result

def time2frame4onset(beat_est, ratio, hop_length=HOP_LENGTH, sr=SR):
    '''
    Convert onset estimation from time to frame representation.
    This function assumes that the beat_est is a 1D array of time values.'''
    result = np.zeros(4096 // ratio)
    idxs = librosa.time_to_frames(beat_est, sr=sr, hop_length=hop_length*ratio,)
    for idx in idxs:
        result[idx] = 1
    return result

class BeatInfoExtractor():
    '''
    Extract beat information from audio files using a trained model.
    '''

    def __init__(self, binfo_type, device, input_csv_path='src/drumaware_hmmparams.csv'):
        self.hmm_proc, self.rnn = get_proc(input_csv_path, device)
        self.binfo_type = binfo_type
        self.device = device

    def __call__(self, audio_file_path):
        feat = utils.get_feature(audio_file_path) 
        # extracts first order spec diffs from [1024, 2048, 4096] and num_bands = [3, 6, 12] -- need to be customised according to the tcn beat info extraction process
        # try:
        beat_output, rhythm_output, out_feature = getTCNembeddings(self.rnn, audio_feat=feat, device=self.device,
                            head = 'nodrum')
        out = utils.prediction_conversion(out)
        beat_info = None #TODO get a fully connected netork to convert and send or send directly and convert later in language model


        # if self.binfo_type == 'high':
        #     beat_est = self.hmm_proc(out)
        #     beat_info = time2frame4beat(beat_est, ratio=4)
        # elif self.binfo_type == 'mid':
        #     beats_spppk_tmp, _ = find_peaks(np.max(out, -1), height = 0.1, distance = 7, prominence = 0.1)
        #     onset_est = beats_spppk_tmp/ 100
        #     beat_info = time2frame4onset(onset_est, ratio=4)
        # elif self.binfo_type == 'low':
        #     beat_info = out_fea
        # else:
        #     beat_info = None
        return beat_info


def get_proc(input_csv_path, device):
    ''' Get the HMM processor and RNN model for beat information extraction.
    Args:
        input_csv_path (str): Path to the CSV file containing model parameters.
        device (torch.device): Device to run the model on (CPU or GPU).
    Returns:
        hmm_proc (DownBproc): HMM processor for beat tracking.
        rnn (DA2): RNN model for beat information extraction.'''
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

    for fn in tqdm(fns):
        ### get feature of input audio file 
        save_path = os.path.join(beat_dir, binfo_type, fn.replace('.wav', '.npy'))
        audio_file_path = os.path.join(audio_dir, 'others', fn)
        if os.path.isfile(save_path):
            continue
        try:
            beat_info = extractor(audio_file_path)
            ### save
            np.save(save_path, beat_info)
        except:
            print(f'{fn} error occur during beat information extraction')

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