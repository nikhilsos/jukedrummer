#code to separate drum tracks and vocals from a mixed audio file


import librosa

from spleeter.separator import Separator

# Instantiate the separator with the desired model
separator = Separator('spleeter:5stems')

# List of audio files to process


for audio_file in audio_files:
    # Separate the audio file
    separator.separate_to_file(audio_file, 'output_directory')
