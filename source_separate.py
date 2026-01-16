import os
import subprocess
import shutil
from tqdm import tqdm

# ---- paths ----
input_directory = '/home/nikhil/jukedrummer/additional_data/extracted_audio'
output_base_dir = '/home/nikhil/jukedrummer/demucs_output'

vocals_directory = os.path.join(output_base_dir, 'vocals')
drums_directory = os.path.join(output_base_dir, 'no_vocals')

os.makedirs(vocals_directory, exist_ok=True)
os.makedirs(drums_directory, exist_ok=True)

# ---- audio files ----
# make a list of all audio files in the input directory
audio_files = [
    f for f in os.listdir(input_directory)
    if f.lower().endswith('.wav')
]

# ---- helpers ----
def already_processed(base_name):
    # returns True if both vocals and drums files exist for the given base_name
    return (
        os.path.isfile(os.path.join(vocals_directory, f'{base_name}.wav')) and
        os.path.isfile(os.path.join(drums_directory, f'{base_name}.wav'))
    )

def safe_move(src, dst):
    # moves src to dst only if src exists and dst does not exist
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    if not os.path.exists(dst):
        shutil.move(src, dst)

# ---- main loop ----
for filename in tqdm(audio_files):
    base_name = os.path.splitext(filename)[0]
    input_path = os.path.join(input_directory, filename)

    if already_processed(base_name):
        continue

    subprocess.run(
        [
            'demucs',
            '-n', 'htdemucs',
            '--two-stems', 'vocals',
            '-o', output_base_dir,
            input_path
        ],
        check=True
    )

    # demucs output layout:
    # output_base_dir/htdemucs/<track_name>/vocals.wav
    # output_base_dir/htdemucs/<track_name>/no_vocals.wav
    separated_path = os.path.join(output_base_dir, 'htdemucs', base_name)

    vocals_src = os.path.join(separated_path, 'vocals.wav')
    drums_src = os.path.join(separated_path, 'no_vocals.wav')

    vocals_dst = os.path.join(vocals_directory, f'{base_name}.wav')
    drums_dst = os.path.join(drums_directory, f'{base_name}.wav')

    safe_move(vocals_src, vocals_dst)
    safe_move(drums_src, drums_dst)

    shutil.rmtree(separated_path)

print('Demucs separation complete.')
