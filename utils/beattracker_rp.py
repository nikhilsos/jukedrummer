"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/beat_tracker.py

Descrption: The main entry point function for the beat tracker. This can be
imported as follows:

>>> from beat_tracking_tcn.beat_tracker import beatTracker

Then it can be invoked like so:

>>> beats, downbeats = beatTracker(path_to_audio_file)
"""
import os
import pickle
import numpy as np
from madmom.features import DBNBeatTrackingProcessor
import torch
import librosa
import sys
sys.path.append('/home/nikhil/projects/dbtracker/')

from model.downbeat_model import BeatNet 
# from beat_tracking_tcn.models.offlinetcn import BeatNet
# from beat_tracking_tcn.models.onlinetcn import BeatNet

import numpy as np

def create_spectrogram(
        file_path,
        n_fft,
        hop_length_in_seconds,
        n_mels):
    
    x, sr = librosa.load(file_path, sr = 22050)

    hop_length_in_samples = int(np.floor(hop_length_in_seconds * sr))

    spec = librosa.feature.melspectrogram(
        y=x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length_in_samples,
        n_mels=n_mels)

    mag_spec = np.abs(spec)
    return mag_spec

def load_checkpoint(model, checkpoint_file):
    """
    Restores a model to a given checkpoint, but loads directly to CPU, allowing
    model to be run on non-CUDA devices.
    """    
    model.load_state_dict(
        torch.load(checkpoint_file, map_location=torch.device('cpu')))


# Some important constants that don't need to be command line params
FFT_SIZE = 2048
HOP_LENGTH_IN_SECONDS = 0.1
SR = 22050
HOP_LENGTH_IN_SAMPLES = np.int64(SR * HOP_LENGTH_IN_SECONDS)
N_MELS = 81



# # Paths to checkpoints distributed with the beat tracker. It's possible to
# # call the below functions with custom checkpoints also.
# DEFAULT_CHECKPOINT_PATH = os.path.join(
#         os.path.dirname(__file__),
#         'checkpoints/default_checkpoint.torch')
# DEFAULT_DOWNBEAT_CHECKPOINT_PATH = os.path.join(
#         os.path.dirname(__file__),
#         'checkpoints/default_downbeat_checkpoint.torch')


# Prepare the models
model = BeatNet()
model.eval()
downbeat_model = BeatNet(downbeats=True)
downbeat_model.eval()

# # Prepare the post-processing dynamic Bayesian networks, courtesy of madmom.
# dbn = DBNBeatTrackingProcessor(
#     min_bpm=5,
#     max_bpm=25,
#     transition_lambda = 100,
#     fps= 10,
#     online=True)






from beat_tracking_tcn.utils.particle_filtering_cascade import particle_filter_cascade
dbn_pf = particle_filter_cascade(beats_per_bar=[], fps= (SR / HOP_LENGTH_IN_SAMPLES), plot=[], mode='offline', min_bpm=20, max_bpm=30, transition_lambda=100)

dbn = DBNBeatTrackingProcessor(
    min_bpm=5,
    max_bpm=10,
    transition_lambda=100,
    fps= (SR / HOP_LENGTH_IN_SAMPLES),
    online=True)

downbeat_dbn = DBNBeatTrackingProcessor(
    min_bpm=5,
    max_bpm=10,
    transition_lambda=100,
    fps=(SR / HOP_LENGTH_IN_SAMPLES),
    online=True)


import scipy
from scipy.signal import find_peaks


def beat_activations_from_spectrogram(
    spectrogram,
    checkpoint_file=None,
    downbeats=True):
    """
    Given a spectrogram, use the TCN model to compute a beat activation
    function.
    """

    # Load the appropriate checkpoint
    if checkpoint_file is not None:
        load_checkpoint(
            downbeat_model if downbeats else model,
            checkpoint_file)
        # If no checkpoint file is provided, the code  wil stop
    else:
        print("No checkpoint file provided, using default model weights.")
        
    



    
        

    # Speed up computation by skipping torch's autograd
    with torch.no_grad():
        # Convert to torch tensor if necessary
        if type(spectrogram) is not torch.Tensor:
            spectrogram_tensor = torch.from_numpy(spectrogram)\
                                    .unsqueeze(0)\
                                    .unsqueeze(0)\
                                    .float()
        else:
            # Otherwise use the spectrogram as-is
            spectrogram_tensor = spectrogram.unsqueeze(0)\
                                    .float()
            
        # print(spectrogram_tensor.shape)
        rtrn, embedding = model(spectrogram_tensor)
        # rp = [1,2,3]
        rtrn, embedding = rtrn.numpy(), embedding.numpy()

        # Forward the spectrogram through the model. Note there are no size
        # restrictions here, as the model is fully convolutional. 
        return downbeat_model(spectrogram_tensor).numpy() if downbeats\
               else rtrn, embedding
    
def predict_beats_from_spectrogram(
        spectrogram,
        checkpoint_file=None,
        downbeats=True,
        min_bpm=5,
        max_bpm=10
   ):
    """
    Given a spectrogram, predict a list of beat times using the TCN model and
    a DBN post-processor.
    """
    # ensure spectrogram is long enough, if not pad with zeros till req_length, if longer than req_length, truncate
    req_length = 3000  # in frames
    # if spectrogram.shape[1] < req_length:
    #     pad_width = req_length - spectrogram.shape[1]
    #     spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
    # elif spectrogram.shape[1] > req_length:
    #     spectrogram = spectrogram[:, :req_length]      

    raw_activations, embeddings = beat_activations_from_spectrogram(
        spectrogram,
        checkpoint_file,
        downbeats
    )

    # raw_activations, embeddings = beat_activations_from_spectrogram(
    #     spectrogram,
    #     checkpoint_file,
    #     downbeats
    # ).squeeze()

    # Perform independent post-processing for downbeats
    if downbeats:
        beat_activations = raw_activations[0]
        downbeat_activations = raw_activations[1]

        dbn.reset()
        dbn(min_bpm, max_bpm)
        predicted_beats = dbn.process_offline(beat_activations.squeeze())

        downbeat_dbn.reset()
        downbeat_dbn(min_bpm, max_bpm)
        predicted_downbeats = downbeat_dbn.process_offline(downbeat_activations.squeeze())

        return predicted_beats, predicted_downbeats
    else:
        beat_activations = raw_activations
        # select min max bpm of dbn
        embeddings = embeddings.squeeze()
        dbn.reset()
        
        predicted_beats = dbn.process(beat_activations.squeeze(), min_bpm=min_bpm, max_bpm=max_bpm)

        return predicted_beats, embeddings.T


class beatTracker:
    def __init__(self, checkpoint_file=None, downbeats=True):
        self.checkpoint_file = checkpoint_file
        self.downbeats = downbeats

    def __call__(self, input_file):
        """
        Load audio, compute spectrogram, and predict beats.
        """
        mag_spectrogram = create_spectrogram(
            input_file,
            FFT_SIZE,
            HOP_LENGTH_IN_SECONDS,
            N_MELS
        ).T

        return predict_beats_from_spectrogram(
            mag_spectrogram,
            self.checkpoint_file,
            self.downbeats
            )



