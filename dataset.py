import os 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def compute_mean_std(mel_dir, pkl=None):
    # Compute means and standards of Mel spectrograms for every Mel-filter bank before normalization
    # print('mel_dir:', mel_dir)
    # print(f"Files in directory: {os.listdir(mel_dir)}")
    
    in_fns = os.listdir(mel_dir)
    in_fns = [fn for fn in in_fns if '.npy' in fn]

    scaler = StandardScaler()
    # pbar = tqdm(in_fns, dynamic_ncols=True)
    non_nan = []
    valid_data_found = False  # Flag to check if valid data is found
    print('Computing mean and std ...')
    
    for fn in in_fns:
        # print('here now')
        if not fn.endswith('.npy'):
            fn += '.npy'
        in_fp = os.path.join(mel_dir, fn)
        # print(f"Trying to load: {in_fp}")
        if not os.path.exists(in_fp):
            print(f"File does not exist: {in_fp}")
            continue
        try:
            
            data = np.load(in_fp).T
            if np.isnan(data).any():
                print(f"Skipping {fn} — contains NaNs")
                continue
            if data.size == 0:
                print(f"Skipping {fn} — empty")
                continue
            non_nan.append(fn)
            scaler.partial_fit(data)
            valid_data_found = True
            print(f"Loaded {fn} successfully")
        except Exception as e:
            print(f"Error loading {fn}: {e}")


    if not valid_data_found:
        raise ValueError("No valid data found to compute mean and std.")

    mean = scaler.mean_
    std = scaler.scale_
    return torch.FloatTensor(mean).view(1, 80, 1), torch.FloatTensor(std).view(1, 80, 1), non_nan

class BeatInfoPairedDataset(Dataset):
    def __init__(self, fl, hps, return_fn=False):
        super().__init__()
        self.fl = fl
        self.root = hps.path
        self.binfo_type = hps.binfo_type
        self.vq_name = hps.vq_name
        self.return_fn = return_fn

    def __getitem__(self, idx):
        fname = self.fl[idx]
        tg_token = np.load(os.path.join(self.root, 'token', 'target', self.vq_name, fname))
        ot_token = np.load(os.path.join(self.root, 'token', 'others', self.vq_name, fname))
        if self.binfo_type is None:
            ot_binfo = np.load(os.path.join(self.root, 'beats', 'low', fname))
        else:
            ot_binfo = np.load(os.path.join(self.root, 'beats', self.binfo_type, fname))
        if self.return_fn:
            return tg_token.squeeze(), ot_token.squeeze(), ot_binfo, fname
        else:
            return tg_token.squeeze(), ot_token.squeeze(), ot_binfo

    def __len__(self):
        return len(self.fl)

class MelDataset(Dataset):
    def __init__(self, fl, hps, data_type):
        super().__init__()
        self.fl = fl
        self.root = hps.path
        self.data_type = data_type

    def __getitem__(self, idx):
        fname = self.fl[idx]
        if not fname.endswith('.npy'):
            fname = fname + '.npy'
        item = np.load(os.path.join(self.root, 'mel', self.data_type, fname))
        return item

    def __len__(self):
        return len(self.fl)

from utils.functions import mel2token, wav2mel

class End2EndWrapper(Dataset):
    
    def __init__(self, input_dir, vqvae, beat_extractor, mel_extractor, others_mean, others_std, device):
        super().__init__()
        self.mel_extractor = mel_extractor
        self.beat_extractor = beat_extractor
        self.others_mean, self.others_std = others_mean, others_std
        self.vqvae = vqvae
        self.device = device
        fns = os.listdir(input_dir)
        self.dpaths = [os.path.join(input_dir,f) for f in fns if f.endswith('.wav')]
    
    def __getitem__(self, index):
        beat_info = self.beat_extractor(self.dpaths[index])
        beat_info = torch.from_numpy(beat_info).unsqueeze(0).to(self.device) if not np.isnan(beat_info).any() else None
        mel = wav2mel(self.dpaths[index], self.mel_extractor)
        t = mel2token(mel, self.vqvae, self.others_mean, self.others_std, self.device)
        t = torch.from_numpy(t).long().unsqueeze(0).to(self.device)
        return t, beat_info, self.dpaths[index].split('/')[-1]

    def __len__(self,):
        return len(self.dpaths)

