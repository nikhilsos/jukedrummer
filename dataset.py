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
    # Input:
    #   mel_dir: an absolute path of Mel data directory 
    #   pkl: Or, filenames of .pkl file is also accepted

    if pkl is not None:
        in_fns = pkl[0]
    else:
        in_fns = os.listdir(mel_dir)
        in_fns = [fn for fn in in_fns if '.npy' in fn]

    scaler = StandardScaler()
    pbar = tqdm(in_fns, dynamic_ncols=True,)
    non_nan = []
    print('computing mean and std ...')
    for fn in pbar:
        in_fp = os.path.join(mel_dir, fn)
        data = np.load(in_fp).T 

        if np.isnan(data).any():
            print(fn)
            continue
        non_nan += [fn]
        scaler.partial_fit(data)
        if True in np.isnan(scaler.scale_):
            break

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
        raw = self.fl[idx]
        fname = os.path.basename(raw)              # ← strip any subdirs here

        tg_token = np.load(os.path.join(self.root, 'token', 'target', self.vq_name, fname))
        ot_token = np.load(os.path.join(self.root, 'token', 'others', self.vq_name, fname))

        if tg_token.shape[0] != ot_token.shape[0]:
            raise ValueError(f"Token length mismatch for {fname}: target {tg_token.shape[0]} vs others {ot_token.shape[0]}")

        if self.binfo_type is None:
            ot_binfo = np.load(os.path.join(self.root, 'token', 'low', fname))
        else:
            ot_binfo = np.load(os.path.join(self.root, 'token', self.binfo_type, fname))

        return (tg_token.squeeze(), ot_token.squeeze(), ot_binfo, fname) if self.return_fn \
               else (tg_token.squeeze(), ot_token.squeeze(), ot_binfo)

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

        # # === FIX: Ensure consistent tensor shapes ===
        # # Pad or truncate to prevent tensor size mismatch during training
        # target_frames = 5167  # Based on the error: 5167 vs 5164
        # if item.shape[1] < target_frames:
        #     # Pad with zeros if shorter
        #     pad_width = target_frames - item.shape[1]
        #     item = np.pad(item, ((0, 0), (0, pad_width)), mode='constant')
        # elif item.shape[1] > target_frames:
        #     # Truncate if longer
        #     item = item[:, :target_frames]
        # # === END FIX ===

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
