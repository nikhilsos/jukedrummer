# embedding_extractor.py
import os
import torch
import numpy as np
import tqdm 
from model.downbeat_model import BeatNet
from hparams import setup_lm_hparams, MODEL_LIST

hps = setup_lm_hparams(MODEL_LIST[1])

def _load_checkpoint_(model: torch.nn.Module, checkpoint_file: str | None):
    if checkpoint_file is None:
        return model
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    sd = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(sd)
    return model

class BeatNetEmbedder:
    """
    Runs BeatNetOffline and returns:
      - beat logits
      - rhythm logits
      - embeddings (latent feature map)  <-- what you asked for
    All as NumPy arrays on CPU.
    """
    def __init__(self, checkpoint_file: str | None = None, device: str | torch.device | None = None):
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model = BeatNet()
        _load_checkpoint_(self.model, checkpoint_file)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, spectrogram: np.ndarray | torch.Tensor):
        """
        spectrogram: expected shape [T, F] (time, freq) as np.ndarray or torch.Tensor
        Returns (, embedding) as np.ndarray (on CPU).
        """
        if not isinstance(spectrogram, torch.Tensor):
            x = torch.from_numpy(spectrogram).float()
        else:
            x = spectrogram.float()

        # Model expected input: [B, C, T, F] or [B, T, F] depending on your impl.
        # Your original code used unsqueeze(0).unsqueeze(0), so mirror that:
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1,1,T,F]
        elif x.ndim == 3:
            x = x.unsqueeze(0)               # [1,T,F]? adapt if needed
        else:
            raise ValueError(f"Unexpected spectrogram shape: {x.shape}")

        x = x.to(self.device)
        embedding = self.model(x)  # assumes model returns 3 tensors
        return embedding.detach().cpu().numpy()
        

def BeatInfoExtractor(spectrogram, checkpoint_file=None, rp_mode=False, binfo_type = hps.binfo_type,device=None):
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
        
def inference(fns, binfo_type, audio_dir, beat_dir, n_cuda):

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

if __name__ == "__main__":
    checkpoint_path = '/home/nikhil/jukedrummer/offline_tcn'  # Update with your checkpoint path
    embedder = BeatNetEmbedder(checkpoint_file=checkpoint_path, device='cuda:0')
    # Example spectrogram input
    example_spectrogram = np.random.rand(3000, 81)  # Random
    embedding = embedder(example_spectrogram) # returns unfiltered activations [1,3000,1]