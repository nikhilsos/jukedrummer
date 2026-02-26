import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import argparse 
import pickle
import random
import sys
sys.path.append('/home/nikhil/projects/dbtracker/')


from melspec import inference as melspec_extraction

segment_audio_dir = '/home/nikhil/projects/dbtracker/data/combined_data/train_val/audio_splitted_1min/'
mel_dir = '/home/nikhil/projects/dbtracker/data/combined_data/train_val/mel_splitted_1min/'
# create if does not exist
os.makedirs(mel_dir, exist_ok=True)
fns = os.listdir(segment_audio_dir)
melspec_extraction(fns, segment_audio_dir, mel_dir)