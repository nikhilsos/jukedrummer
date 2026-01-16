import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import argparse 
import pickle
import random
import sys
sys.path.append('/home/nikhil/projects/dbtracker/')

from segmentation import inference as data_segmentation
from utils.melspec import inference as melspec_extraction
from subset_division import inference as subset_division
from pansori_beats import inference as beat_info_extraction
# import logging

# logging.basicConfig(filename='test.log', level=logging.DEBUG,
#                     format='%(asctime)s:%(levelname)s:%(message)s')

def main(args):
    # segment -> mel extract -> div subset -> beat information extract
    audio_dir = args.audio_dir
    mel_dir = args.mel_dir
    beat_dir= args.beat_dir
    segment_audio_dir = args.segment_audio_dir

    length = 8192 * 8 * 4 * 4 # This variation is recommended to be fixed
    fns = os.listdir(os.path.join(audio_dir, 'target'))
    # fns = fns[60]
    fns = [f for f in fns if f.endswith('.wav')]
    # fns = fns[:60]
    # # 1. Segmentation by either downbeats or hop window
    # data_segmentation(fns, False, length, audio_dir)

    fns = os.listdir(os.path.join(segment_audio_dir, 'target'))

    fns = [f for f in fns if f.endswith('.wav')]
    # # # # 2. Extract Mel spectrograms from segemented audio waves
    melspec_extraction(fns, segment_audio_dir, mel_dir)
    # # # # 3. Divide dataset into train & valid subset
    # subset_division(mel_dir, args.dataset_pkl_path)
    #TODO: Integrate timbral features extraction here

    # 4. Beat Information Extraction
    beat_info_extraction(fns, 'low', segment_audio_dir, beat_dir, args.cuda)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, help='directory path of unsegemented audio', default='data/audio')
    parser.add_argument('--segment_audio_dir', type=str, help='directory path of segmented audio', default='data/segment_audio')
    parser.add_argument('--mel_dir', type=str, help='directory path of segemented audio', default='data/mel/target')
    parser.add_argument('--beat_dir', type=str, help='directory path of beat information', default='data/token')
    parser.add_argument('--cuda', type=int, help='the id of cuda want to use')
    parser.add_argument('--dataset_pkl_path', type=str, help='the path of final dataset .pkl file', default='data/dataset.pkl')
    parser.add_argument('--segment_by_downbeats', type=bool, default=False, help='determine whether the segement would be made according to downbeats or not')
    args = parser.parse_args()
    main(args)

