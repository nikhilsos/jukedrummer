import librosa 
import soundfile as sf
import numpy as np 

# load numpy
spectrogram = np.load('/home/nikhil/jukedrummer/data/audio/reconstructed_mel_recon_vq1/37_segment_1.npy')

audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr=44100)
# save the audio file
sf.write('/home/nikhil/jukedrummer/griffithlim_reconstructed.wav', audio, 44100)
