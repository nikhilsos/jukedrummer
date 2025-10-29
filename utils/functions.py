import os
import numpy as np
import torch
from model.vqvae import Sampler, VQVAE
from hparams import Hyperparams
import librosa

def get_vqvae(vq_idx, data_type, ckpt_dir, device):
    ckpt_name = f'vq{vq_idx}_{data_type}.pkl'
    ckpt = torch.load(
        os.path.join(ckpt_dir, ckpt_name), 
        map_location=lambda storage, loc: storage, weights_only=False
    )

    # # load pth instead of pkl --- IGNORE ---
    # ckpt_name = f'vq{vq_idx}_{data_type}.pth' #--- IGNORE ---
    # ckpt = torch.load( #--- IGNORE ---
    #     os.path.join(ckpt_dir, ckpt_name),  #--- IGNORE ---
    #     map_location=lambda storage, loc: storage, weights_only=False #--- IGNORE ---
    # ) #--- IGNORE ---


    hps = Hyperparams(ckpt['hps'])
    # hps['mel_dir'] = '/home/lego/NAS189/home/codify/data/drums/feature/mel/hop'
    hps['output_dir'] = os.path.join(hps['path'], 'token', data_type, f'vq{vq_idx}')
    hps['seq_len'] = 4096 // np.prod(hps.upsample_ratios)
    hps['data_type'] = data_type
    encoder = Sampler(80, 64, hps.downsample_ratios)
    decoder = Sampler(64, 80, hps.upsample_ratios)
    vqvae = VQVAE(codebook_size=hps.codebook_size, encoder=encoder, decoder=decoder)
    vqvae.load_state_dict(ckpt['model'])
    vqvae.to(device)
    mean, std = torch.FloatTensor(ckpt['mean']).to(device), torch.FloatTensor(ckpt['std']).to(device)
    
    os.makedirs(hps.output_dir, exist_ok=True)
    print(hps.output_dir)
    print(ckpt_name)

    return vqvae, hps, mean, std

def mel2token(data, vqvae, mean, std, device):
    if isinstance(data, str):
        m = np.load(data)
    elif isinstance(data, np.ndarray):
        m = data
    if np.any(np.isnan(m)):
        return None

    with torch.no_grad():
        m = torch.FloatTensor(m[np.newaxis,:,:]).to(device)
        m = (m - mean) / std
        x_l = vqvae.encode(m)
    x_l = x_l.squeeze(0).long().detach().cpu().numpy()
    return x_l


def token2mel(data, vqvae, mean, std, device):
    if isinstance(data, str):
        t = np.load(data)
    elif isinstance(data, np.ndarray):
        t = data
    if np.any(np.isnan(t)):
        return None

    with torch.no_grad():
        t = torch.LongTensor(t).to(device)
        m = vqvae.decode(t)
    return m
        

def wav2mel(data, audio2mel):
    if isinstance(data, str):
        y, _ = librosa.load(data, sr = 44100)
    elif isinstance(data, np.ndarray):
        y = data
    peak = np.abs(y).max()
    if peak > 1.0:
        y /= peak
    y = torch.from_numpy(y)
    mel = audio2mel(y[None, None])
    mel = mel.numpy()[0].astype(np.float32)
    ## silence filter 
    if np.any(np.isnan(mel)) or np.sum(np.mean(mel, axis=0, keepdims=False) < -8) > mel.shape[1] // 2:
        return None
    return mel


# added later

# function to 