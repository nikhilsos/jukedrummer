# Audio stem separation using Demucs 4-stem model
# Separates audio files into vocals, bass, drums, and other stems
# Organizes output files into respective folders by stem type

import os
import subprocess
import argparse
from pathlib import Path

def separate_audio_files(input_folder, output_folder):
    """
    Separate all audio files in input_folder using Demucs 4-stem model
    and organize them into stem-specific folders.

    Args:
        input_folder (str): Path to folder containing audio files
        output_folder (str): Path to output folder for separated stems
    """

    # Supported audio formats
    audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}

    # Find all audio files in input folder
    audio_files = []
    for file_path in Path(input_folder).rglob('*'):
        if file_path.suffix.lower() in audio_extensions:
            audio_files.append(str(file_path))

    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return

    print(f"Found {len(audio_files)} audio files to process")

    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each audio file using Demucs
    for audio_file in audio_files:
        print(f"Processing: {audio_file}")

        audio_path = Path(audio_file)
        audio_name = audio_path.stem

        # Demucs command
        cmd = [
            'demucs',
            '-n', 'htdemucs',
            '-o', str(output_path),
            audio_file
        ]


        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Successfully processed: {audio_name}")

        except subprocess.CalledProcessError as e:
            print(f"Error processing {audio_file}: {e}")
            if e.stderr:
                print(f"Error output: {e.stderr}")

def organize_stems_by_folder(output_folder):
    """
    Organize separated audio files into stem-specific folders with song names as filenames.

    Args:
        output_folder (str): Path to output folder containing separated files
    """

    output_path = Path(output_folder) / 'htdemucs'

    # Create main stem folders
    stem_folders = {
        'bass': output_path / 'bass',
        'drums': output_path / 'drums',
        'vocals': output_path / 'vocals',
        'others': output_path / 'others'
    }

    for folder in stem_folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    # Find all separated audio files and reorganize them
    for audio_file in output_path.rglob('*.wav'):
        # Skip files already in stem folders
        if audio_file.parent in stem_folders.values():
            continue

        # Get the song name (parent directory name)
        song_name = audio_file.parent.name

        # Determine which stem this file belongs to
        filename = audio_file.name.lower()
        if 'bass' in filename:
            target_folder = stem_folders['bass']
            # Create new filename using song name
            new_filename = f"{song_name}.wav"
        elif 'drums' in filename:
            target_folder = stem_folders['drums']
            new_filename = f"{song_name}.wav"
        elif 'vocals' in filename:
            target_folder = stem_folders['vocals']
            new_filename = f"{song_name}.wav"
        else:
            target_folder = stem_folders['others']
            new_filename = f"{song_name}.wav"

        # Move file to appropriate folder with song name as filename
        new_path = target_folder / new_filename
        audio_file.rename(new_path)
        print(f"Moved {audio_file.name} to {target_folder.name}/{new_filename}")

    # Clean up empty subdirectories (original song folders)
    for subdir in output_path.rglob('*'):
        if subdir.is_dir() and not any(subdir.iterdir()) and subdir != output_path:
            try:
                subdir.rmdir()
            except OSError:
                pass  # Directory not empty or other error

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Separate audio files using Demucs 4-stem model')
    parser.add_argument('input_folder', help='Path to folder containing audio files')
    parser.add_argument('output_folder', help='Path to output folder for separated stems')

    args = parser.parse_args()

    # separate_audio_files(args.input_folder, args.output_folder)
    organize_stems_by_folder(args.output_folder)
    print("Audio separation completed successfully!")
