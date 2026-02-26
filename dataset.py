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

# class BeatInfoPairedDataset(Dataset):
#     def __init__(self, fl, hps, return_fn=False):
#         super().__init__()
#         self.fl = fl
#         self.root = hps.path
#         self.binfo_type = hps.binfo_type
#         self.vq_name = hps.vq_name
#         self.return_fn = return_fn
#         self.max_len = 4096 // int(np.prod(hps.upsample_ratios))

#     def __getitem__(self, idx):
#         raw = self.fl[idx]
#         fname = os.path.basename(raw)              # ← strip any subdirs here

#         tg_token = np.load(os.path.join(self.root, 'token', 'target', self.vq_name, fname))
#         ot_token = np.load(os.path.join(self.root, 'token', 'others', self.vq_name, fname))
#         # print(tg_token.shape, ot_token.shape, 'token shapes')  # debugging

#         assert tg_token.shape[0] == ot_token.shape[0] and tg_token.shape[0] <=self.max_len , \
#         print(f"Token length mismatch for {fname}: target {tg_token.shape[0]} vs others {ot_token.shape[0]}")

#         if self.binfo_type is None:
#             ot_binfo = np.load(os.path.join(self.root, 'token', 'low', fname))
#         else:
#             ot_binfo = np.load(os.path.join(self.root, 'token', self.binfo_type, fname))

#         return (tg_token.squeeze(), ot_token.squeeze(), ot_binfo, fname) if self.return_fn \
#                else (tg_token.squeeze(), ot_token.squeeze(), ot_binfo)

#     def __len__(self):
#         return len(self.fl)

class BeatInfoPairedDataset(Dataset):
    def __init__(self, fl, hps, return_fn=False, return_class=True):
        super().__init__()
        self.fl = fl
        self.root = hps.path
        self.binfo_type = hps.binfo_type
        self.vq_name = hps.vq_name
        self.return_fn = return_fn
        self.return_class = return_class          # ← new flag (default True)
        self.max_len = 4096 // int(np.prod(hps.upsample_ratios))

    def get_class_from_filename(self, fname):
        """
        Assign class ID based on lowercase prefix:
          0 = starts with 'jy'
          1 = starts with 'zaq'
          2 = starts with 'zz'
          3 = everything else
        """
        base = os.path.basename(fname).lower()
        if base.startswith('jy'):
            return 0
        elif base.startswith('za'):
            return 1
        elif base.startswith('zz'):
            return 2
        else:
            return 3

    def __getitem__(self, idx):
        raw = self.fl[idx]
        fname = os.path.basename(raw)   # already basename, but safe

        tg_token = np.load(os.path.join(self.root, 'token', 'target', self.vq_name, fname))
        ot_token = np.load(os.path.join(self.root, 'token', 'others', self.vq_name, fname))

        assert tg_token.shape[0] == ot_token.shape[0] and tg_token.shape[0] <= self.max_len, \
            f"Token length mismatch for {fname}: target {tg_token.shape[0]} vs others {ot_token.shape[0]}"

        if self.binfo_type is None:
            ot_binfo = np.load(os.path.join(self.root, 'token', 'low', fname))
        else:
            ot_binfo = np.load(os.path.join(self.root, 'token', self.binfo_type, fname))

        # Prepare return tuple
        out = [tg_token.squeeze(), ot_token.squeeze(), ot_binfo]

        if self.return_fn:
            out.append(fname)

        if self.return_class:
            class_id = self.get_class_from_filename(raw)   # or fname
            out.append(class_id)

        return tuple(out)

    def __len__(self):
        return len(self.fl)
    
class ConditionalInformationDataset(Dataset):
    def __init__(self, fl, hps):
        super().__init__()
        self.fl = fl
        self.root = hps.path
        self.binfo_type = hps.binfo_type
        self.vq_name = hps.vq_name

    def load_jangdan(self, fname):
        # logic to return the jangdan's embedding from file name
        pass

    def load_downbeat_labels(self, fname):
        # logic to return the downbeat labels from downbeat annotation folder search same name file -- vector (2378,1) downbeat activation vector
        downbeat_annotation = os.path.join(self.root, 'downbeat_annotations', fname.replace('.npy', '.txt'))
        if os.path.exists(downbeat_annotation):
            with open(downbeat_annotation, 'r') as f:
                # format = 'timestamp'/t'label' if label = 1 means downbeat
                downbeat_labels = []
                for line in f:
                    timestamp, label = line.strip().split('\t')
                    downbeat_labels.append((float(timestamp), int(label)))
                return downbeat_labels
        else:            
            print(f"Downbeat annotation not found for {fname}")
            return None
        pass

# class MelDataset(Dataset):
#     def __init__(self, fl, hps, data_type):
#         super().__init__()
#         self.fl = fl
#         self.root = hps.path
#         self.data_type = data_type
#         self.T = 4096

#     def __getitem__(self, idx):
#         fname = self.fl[idx]
#         if not fname.endswith('.npy'):
#             fname = fname + '.npy'
#         item = np.load(os.path.join(self.root, 'mel', self.data_type, fname))

#         # # === FIX: Ensure consistent tensor shapes ===
#         # # Pad or truncate to prevent tensor size mismatch during training
#         # target_frames = 5167  # Based on the error: 5167 vs 5164
#         # if item.shape[1] < target_frames:
#         #     # Pad with zeros if shorter
#         #     pad_width = target_frames - item.shape[1]
#         #     item = np.pad(item, ((0, 0), (0, pad_width)), mode='constant')
#         # elif item.shape[1] > target_frames:
#         #     # Truncate if longer
#         #     item = item[:, :target_frames]
#         # # === END FIX ===

#         return item

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
        fn = self.dpaths[index].split('/')[-1]
        base = fn.lower()
        
        if base.startswith('jy'):
            class_id = 0
        elif base.startswith('za'):
            class_id = 1
        elif base.startswith('zz'):
            class_id = 2
        else:
            class_id = 3
        
        class_id = torch.tensor([class_id], dtype=torch.long).to(self.device)

        beat_info = self.beat_extractor(self.dpaths[index])
        beat_info = np.asarray(beat_info, dtype=np.float32)
        if not np.isnan(beat_info).any():
            beat_info = torch.from_numpy(beat_info).unsqueeze(0).to(self.device)
        else:
            beat_info = None

        mel = wav2mel(self.dpaths[index], self.mel_extractor)
        t = mel2token(mel, self.vqvae, self.others_mean, self.others_std, self.device)
        t = torch.from_numpy(t).long().unsqueeze(0).to(self.device)
        
        return t, beat_info, fn, class_id  # ← added class_id

    def __len__(self,):
        return len(self.dpaths)
