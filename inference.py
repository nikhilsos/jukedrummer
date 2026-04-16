import torch
import torch.nn
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from hparams import setup_lm_hparams, MODEL_LIST
import soundfile as sf
import argparse
import time

from model.LanguageModel import JukeTransformer
from test_vqvae_reconstruction import plot_comparison, plot_triple_comparison
from dataset import End2EndWrapper
# from model.vocoder import HiFiVocoder
from model.vocoder import HiFiVocoder
from hparams import MEL
from utils.functions import get_vqvae, wav2mel, mel2token
from utils.beats import BeatInfoExtractor  # note if its pansori or just beats 
from utils.melspec import Audio2Mel
from utils.pansori_beats import BeatNetEmbedder
# 
def get_raw_data(input_dir):
    fns =  os.listdir(input_dir)
    fns = [f for f in fns if f.endswith('.wav')]
import matplotlib.pyplot as plt

import librosa.display
def save_spectrogram_visuals(mel, out_path):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=44100, hop_length=256, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close() 
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_idx', type=int, help='Determine which experiment id of models is used to sample')
    parser.add_argument('--cuda', type=int, help='cuda id')
    parser.add_argument('--input_dir', type=str, default='input/')
    parser.add_argument('--output_dir', type=str, default='output/')
    parser.add_argument('--sample_iters', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.7, help='the temperature when sampling')
    parser.add_argument('--top_p', type=float, default=0.6, help='the threshold probability when sampling')
    args = parser.parse_args()

    # Parameters Initialization
    exp_idx = args.exp_idx
    output_dir = os.path.join(args.output_dir, f'exp{exp_idx}')
    hps = setup_lm_hparams(MODEL_LIST[exp_idx])
    hps.cuda = args.cuda
    hps.batch_size = 1
    # print(f'setting up exp{exp_idx} parameters')
    # print(f'cuda: {hps.cuda}')
    # print(f'bs  : {hps.batch_size}')
    # print(f'temp: {args.temp}')
    print(f'generate samples from {args.input_dir}')
    print(f'save generated samples to {args.output_dir}')
    print(f"vqvae name: {hps.vq_name},language model name: {MODEL_LIST[exp_idx]}")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f'cuda:{hps.cuda}')
    
    # Model Initialization 
    target_vqvae, _, target_mean, target_std = get_vqvae(
        vq_idx=hps.vq_name.strip('vq'),
        data_type='target', 
        ckpt_dir=hps.ckpt_dir, 
        device = device
    )

    others_vqvae, _, others_mean, others_std = get_vqvae(
        vq_idx=hps.vq_name.strip('vq'),
        data_type='others', 
        ckpt_dir=hps.ckpt_dir, 
        device = device
    )

    lm = JukeTransformer(hps).to(device)
    # lm_ckpt = torch.load(os.path.join(hps.ckpt_dir, f'exp{exp_idx}_og.pkl'), map_location=lambda storage, loc: storage)
    lm_ckpt = torch.load(('/home/nikhil/jukedrummer/ckpt/exp23_best_fad.pkl'), map_location=lambda storage, loc: storage )
    # print("lm_ckpt['model']", '\n'.join(lm_ckpt['model'].keys()))
    # print("lm", '\n'.join(lm.state_dict().keys()))
    lm.load_state_dict(lm_ckpt['model'], strict=False)

    
    # vocoder = HiFiVocoder(
    #     ckpt_path=os.path.join(hps.ckpt_dir, 'vocoder/generator'), 
    #     output_dir=output_dir, 
    #     device=device
    # )

    vocoder = HiFiVocoder(
        ckpt_path='/home/nikhil/jukedrummer/cp_hifigan/g_00000635', 
        output_dir=output_dir, 
        device=device
    )

    # vocoder = HiFiVocoder(
    #     ckpt_path='/home/nikhil/jukedrummer/hifi_gan/cp_hifigan/g_00073500', 
    #     output_dir=output_dir, 
    #     device=device
    # )

    

    # vocoder = HiFiVocoder(
    #     ckpt_path='/home/nikhil/jukedrummer/hifi_gan/cp_hifigan_pansori/pansori_generator', 
    #     output_dir=output_dir, 
    #     device=device
    # )
    
    beat_extractor = BeatInfoExtractor(binfo_type=hps.binfo_type, device=device)
    # beat_extractor = BeatNetEmbedder(binfo_type=hps.binfo_type, device=device)

    mel_extractor = Audio2Mel(MEL)
  
    dataset = End2EndWrapper(
        args.input_dir, 
        others_vqvae, 
        beat_extractor, 
        mel_extractor, 
        others_mean, 
        others_std, 
        device
    )
    
        # ── codebook acoustic inspection ─────────────────────────────────────────
    import matplotlib.pyplot as plt
    import librosa.display

    codebook_dir = os.path.join(output_dir, 'codebook_sounds')
    os.makedirs(codebook_dir, exist_ok=True)

    top_tokens = [11, 20, 16, 8, 2]  # top 5 most frequent from utilization check

    with torch.no_grad():
        for token_id in top_tokens:
            # decode a full sequence of the same token
            token = torch.tensor([[token_id] * 1024]).long().to(device)
            mel = target_vqvae.decode(token)  # [1, 80, T]
            
            # denormalize
            mel = mel * target_std + target_mean
            
            # save mel as image
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel[0].cpu().numpy(), 
                sr=44100, hop_length=256, 
                x_axis='time', y_axis='mel'
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Token {token_id} (frequency: {[58,11,9,3,3][top_tokens.index(token_id)]}%)')
            plt.tight_layout()
            plt.savefig(os.path.join(codebook_dir, f'token_{token_id}_mel.png'))
            plt.close()
            
            # convert to audio via vocoder
            wav_np = vocoder(mel)
            # wav_np = wav.detach().cpu().numpy()
            if wav_np.ndim == 1:
                wav_np = wav_np[:, None]
            sf.write(os.path.join(codebook_dir, f'token_{token_id}.wav'), wav_np, 44100)
            
            print(f"token {token_id}: mel mean={mel.mean().item():.3f}, std={mel.std().item():.3f}")

        print(f"Codebook sounds saved to {codebook_dir}")
# ─────────────────────────────────────────────────────────────────────────

    
    for j in range(args.sample_iters):
        for i in tqdm(range(len(dataset))):
            otz, binfo, fn, class_id = dataset[i]
            with torch.no_grad():

                lm.eval()
                print(f'otz: {otz.shape}', 'binfo shape during inference: ', binfo.shape)
                print(f"binfo_type: {lm.binfo_type}")
                print(f"binfo shape: {binfo.shape}, dtype: {binfo.dtype}")
                
                gen_mel = lm.primed_sample(n_samples=hps.batch_size, otz=otz, binfo=binfo, vqvae=target_vqvae, temp=args.temp, top_p=args.top_p, class_id=class_id)

                gen_mel = gen_mel * target_std + target_mean
                gen_wavs = vocoder(gen_mel)

                # Panels 1 & 2: original drums and VQ-VAE reconstruction
                target_audio_dir = args.input_dir.replace('others', 'target')
                target_wav_path  = os.path.join(target_audio_dir, fn)
                if os.path.exists(target_wav_path):
                    # Raw target mel (panel 1)
                    target_raw_mel = wav2mel(target_wav_path, mel_extractor)
                    raw_mel_np = target_raw_mel[0].cpu().numpy() if isinstance(target_raw_mel, torch.Tensor) else target_raw_mel

                    # VQ-VAE roundtrip: encode → decode (panel 2)
                    real_tgz = mel2token(target_raw_mel, target_vqvae, target_mean, target_std, device)
                    real_tgz = torch.from_numpy(real_tgz).long().unsqueeze(0).to(device)
                    recon_mel = target_vqvae.decode(real_tgz) * target_std + target_mean

                    plot_triple_comparison(
                        [raw_mel_np],
                        [recon_mel[0].cpu().numpy()],
                        [gen_mel[0].cpu().numpy()],
                        [fn],
                        os.path.join(output_dir, f'{j}_' + fn.replace('.wav', '_comparison.png')),
                    )
                else:
                    # No paired target — fall back to 2-panel (others vs generated)
                    others_mel = others_vqvae.decode(otz) * others_std + others_mean
                    plot_comparison(
                        [others_mel[0].cpu().numpy()],
                        [gen_mel[0].cpu().numpy()],
                        [fn],
                        os.path.join(output_dir, f'{j}_' + fn.replace('.wav', '_comparison.png')),
                        data_type='others vs generated'
                    )
                        
            gen_wavs_np = gen_wavs.detach().cpu().numpy() if isinstance(gen_wavs, torch.Tensor) else gen_wavs
            if gen_wavs_np.ndim == 1:
                gen_wavs_np = gen_wavs_np[:, None]
            name, ext = os.path.splitext(fn)

            out_path = os.path.join(
                output_dir,
                f"{name}_generated_{j}{ext}"
            )

            sf.write(out_path, gen_wavs_np, 44100)
            # sf.write(os.path.join(output_dir, (fn + f'{j}' + '_generated').replace('.npy', '.wav')), gen_wavs_np, 44100)

    # real_wavs = real_wav.detach().cpu().numpy()