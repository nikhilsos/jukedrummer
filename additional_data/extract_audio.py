import os
import soundfile as sf
from tqdm import tqdm

input_folder_path = "/home/nikhil/jukedrummer/additional_data/unzipped"
output_folder_path = "/home/nikhil/jukedrummer/additional_data/extracted"
output_file_path_wav = "/home/nikhil/jukedrummer/additional_data/extracted_audio"
os.makedirs(output_folder_path, exist_ok=True)

audio_extensions = (".wav", ".mp3", ".ogg", ".flac")

# Collect all audio files first
audio_files = []
for root, _, files in os.walk(input_folder_path):
    for file in files:
        if file.lower().endswith(audio_extensions):
            audio_files.append(os.path.join(root, file))

# Process with tqdm over a known-length iterable
for audio_file_path in tqdm(audio_files, desc="Extracting audio"):
    file = os.path.basename(audio_file_path)
    output_file_path = os.path.join(output_folder_path, file)

    if os.path.exists(output_file_path):
        continue

    audio_data, sample_rate = sf.read(audio_file_path)
    sf.write(output_file_path, audio_data, sample_rate)

print("Audio extraction complete.")

# convert the mp3 files to wav and save in a different folder (output_file_path_wav)
os.makedirs(output_file_path_wav, exist_ok=True)
for audio_file_path in tqdm(audio_files, desc="Converting to WAV"):
    file = os.path.basename(audio_file_path)
    base_name, ext = os.path.splitext(file)
    if ext.lower() == '.wav':
        continue  # Skip if already WAV

    output_wav_path = os.path.join(output_file_path_wav, f"{base_name}.wav")

    if os.path.exists(output_wav_path):
        continue

    audio_data, sample_rate = sf.read(audio_file_path)
    sf.write(output_wav_path, audio_data, sample_rate)
    
print("Conversion to WAV complete.")