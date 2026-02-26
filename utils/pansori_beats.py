# embedding_extractor.py
import os
import torch
import numpy as np
from tqdm import tqdm 
from model.downbeat_model import BeatNet_offlinetcn as BeatNet
from hparams import setup_lm_hparams, MODEL_LIST
from utils.beattracker_rp import beatTracker, create_spectrogram
from concurrent.futures import ProcessPoolExecutor
hps = setup_lm_hparams(MODEL_LIST[1])

def _load_checkpoint_(model: torch.nn.Module, checkpoint_file: str | None):
    if checkpoint_file is None:
        return model
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    sd = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(sd)
    return model
from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import (
            FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor,
            SpectrogramDifferenceProcessor)
from madmom.processors import ParallelProcessor, Processor, SequentialProcessor

def madmom_feature(wav):
    """ returns the madmom features mentioned in the paper"""
    sig = SignalProcessor(num_channels=1, sample_rate=44100 )
    multi = ParallelProcessor([])
    frame_sizes = [1024, 2048, 4096]
    num_bands = [3, 6, 12]
    for frame_size, num_bands in zip(frame_sizes, num_bands):
        frames = FramedSignalProcessor(frame_size=frame_size, fps=100)
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            num_bands=num_bands, fmin=30, fmax=17000, norm_filters=True)
        spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
        diff = SpectrogramDifferenceProcessor(
            diff_ratio=0.5, positive_diffs=True, stack_diffs=np.hstack)
        # process each frame size with spec and diff sequentially
        multi.append(SequentialProcessor((frames, stft, filt, spec, diff)))
    # stack the features and processes everything sequentially
    pre_processor = SequentialProcessor((sig, multi, np.hstack))
    feature = pre_processor.process( wav)
    return feature

class BeatNetEmbedder:
    """
    Runs BeatNetOffline and returns:
      - beat logits
      - rhythm logits
      - embeddings (latent feature map)  <-- what you asked for
    All as NumPy arrays on CPU.
    """
    def __init__(self, binfo_type, device, input_csv_path='src/drumaware_hmmparams.csv'):
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = BeatNet(downbeats=True)
        self.model.to(self.device)
        self.model.eval()
        self.binfo_type = binfo_type
        self.input_csv_path = input_csv_path
        
        checkpoint_file = '/home/nikhil/jukedrummer/offline_tcn_vocals_csv_1'
        _load_checkpoint_(self.model, checkpoint_file)
        
    # @torch.no_grad()
    # def __call__(self, audio_file_path):
    #     """
    #     spectrogram: expected shape [T, F] (time, freq) as np.ndarray or torch.Tensor
    #     Returns (, embedding) as np.ndarray (on CPU).
    #     """
    #     x = create_spectrogram(audio_file_path,   n_fft=FFT_SIZE,
    #                                     hop_length_in_seconds=HOP_LENGTH_IN_SECONDS,
    #                                     n_mels=N_MELS)
    #     print(x.shape) # debugging
    #      # shape: [T, F]
    #     spectrogram = torch.from_numpy(
    #                 np.expand_dims(x, axis=0)).float().to(self.device)
        
    #     spectrogram = spectrogram.transpose(-1,1).unsqueeze(0) # shape: [1, T, F]



    #     _, op = self.model(spectrogram)  # add channel dim: [1, 1, T, F]

    #     # op = op.cpu().numpy().squeeze(0).T
    #     op = op.cpu().numpy().squeeze(0)  # shape: [T, C]

    #     print(op.shape, 'op')  # shape: [2, 16, N]

    #     return op 
    @torch.no_grad()
    def __call__(self, audio_file_path):

        spec = madmom_feature(audio_file_path)
        extractor = beatTracker(checkpoint_file='/home/nikhil/jukedrummer/offline_tcn_vocals_csv_1', downbeats=True)

        acts, beat_info = extractor(audio_file_path)
        beat_info = np.array(beat_info)
        # spec: (T, F)

        # spec = torch.from_numpy(spec.T).float().to(self.device)
        # # (F, T)

        # spec = spec.unsqueeze(0).unsqueeze(0)
        # # (1, 1, F, T)

        # print("Spec shape:", spec.shape)

        # with torch.no_grad():
        #     _, op = self.model(spec)

        # op = op.squeeze(0).cpu().numpy()

        # print("Output shape:", op.shape)

        return beat_info
    

 
        

def BeatInfoExtractor_func(spectrogram, checkpoint_file=None, rp_mode=False, binfo_type = hps.binfo_type,device=None):
    """
    Kept for compatibility with your code. If rp_mode=True, returns (activations, rp).
    Otherwise returns activations only.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BeatNet()
    _load_checkpoint_(model, checkpoint_file)
    model.to(device).eval()

    with torch.no_grad():
        if not isinstance(spectrogram, torch.Tensor):
            spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float()
        else:
            spectrogram_tensor = spectrogram.unsqueeze(0).float()
        spectrogram_tensor = spectrogram_tensor.to(device)

        if rp_mode:
            rtrn, rp = model(spectrogram_tensor)  # adjust if the rp head exists
            return rtrn.detach().cpu().numpy(), rp.detach().cpu().numpy()
        else:
            rtrn = model(spectrogram_tensor)
            if isinstance(rtrn, (tuple, list)):
                rtrn = rtrn[0]  # first head as "activations"
            return rtrn.detach().cpu().numpy()
        
FFT_SIZE = 2048
HOP_LENGTH_IN_SECONDS = 0.1
SR = 22050
HOP_LENGTH_IN_SAMPLES = np.int64(SR * HOP_LENGTH_IN_SECONDS)
N_MELS = 81

class BeatInfoExtractor():

    def __init__(self, binfo_type, device, input_csv_path='src/drumaware_hmmparams.csv'):
        # self.hmm_proc, self.rnn = get_proc(input_csv_path, device)
        self.binfo_type = binfo_type
        self.device = device

    def __call__(self, audio_file_path):
        spectrogram = create_spectrogram(audio_file_path,   n_fft=FFT_SIZE,
                                        hop_length_in_seconds=HOP_LENGTH_IN_SECONDS,
                                        n_mels=N_MELS)
        beat_info = BeatInfoExtractor_func(spectrogram, checkpoint_file='/home/nikhil/jukedrummer/offline_tcn', rp_mode=False, binfo_type=self.binfo_type, device=self.device)

        return beat_info
        
def inference(fns, binfo_type, audio_dir, beat_dir, n_cuda):

    input_csv_path='src/drumaware_hmmparams.csv'
    device = torch.device(f'cuda:{n_cuda}' if torch.cuda.is_available() else 'cpu')
    # extractor = BeatInfoExtractor(binfo_type, device, input_csv_path=input_csv_path)
    extractor = beatTracker(checkpoint_file='/home/nikhil/jukedrummer/offline_tcn_vocals_csv_1', downbeats=True)

    for fn in tqdm(fns):
        ### get feature of input audio file 
        save_path = os.path.join(beat_dir, binfo_type, fn.replace('.wav', '.npy'))
        audio_file_path = os.path.join(audio_dir, 'others', fn)
        if os.path.isfile(save_path):
            continue
        try:
            acts, beat_info = extractor(audio_file_path)
            beat_info = np.array(beat_info)
            # print(f'beat_info shape: {beat_info.shape}')
            np.save(save_path, beat_info)   
        except Exception as e:
            raise Exception(f'{fn} error during beat information extraction: {e}')


if __name__ == "__main__":
    checkpoint_path = '/home/nikhil/jukedrummer/offline_tcn'  # Update with your checkpoint path
    embedder = beatTracker(checkpoint_file='/home/nikhil/jukedrummer/offline_tcn_vocals_csv_1', downbeats=True)
    # Example spectrogram input
    example_spectrogram = np.random.rand(3000, 314)  # Random
    _, embedding = embedder(example_spectrogram) # returns unfiltered activations [1,3000,1]
