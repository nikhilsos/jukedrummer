# # Audio stem separation using Demucs 4-stem model
# # Separates audio files into vocals, bass, drums, and other stems
# # Organizes output files into respective folders by stem type

# import os
# import subprocess
# import argparse
# from pathlib import Path

# def separate_audio_files(input_folder, output_folder):
#     """
#     Separate all audio files in input_folder using Demucs 4-stem model
#     and organize them into stem-specific folders.

#     Args:
#         input_folder (str): Path to folder containing audio files
#         output_folder (str): Path to output folder for separated stems
#     """

#     # Supported audio formats
#     audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}

#     # Find all audio files in input folder
#     audio_files = []
#     for file_path in Path(input_folder).rglob('*'):
#         if file_path.suffix.lower() in audio_extensions:
#             audio_files.append(str(file_path))

#     if not audio_files:
#         print(f"No audio files found in {input_folder}")
#         return

#     print(f"Found {len(audio_files)} audio files to process")

#     # Create output directory
#     output_path = Path(output_folder)
#     output_path.mkdir(parents=True, exist_ok=True)

#     # Process each audio file using Demucs
#     for audio_file in audio_files:
#         print(f"Processing: {audio_file}")

#         audio_path = Path(audio_file)
#         audio_name = audio_path.stem

#         # Demucs command
#         cmd = [
#             'demucs',
#             '-n', 'htdemucs',
#             '-o', str(output_path),
#             audio_file
#         ]


#         try:
#             result = subprocess.run(cmd, check=True, capture_output=True, text=True)
#             print(f"Successfully processed: {audio_name}")

#         except subprocess.CalledProcessError as e:
#             print(f"Error processing {audio_file}: {e}")
#             if e.stderr:
#                 print(f"Error output: {e.stderr}")

# def organize_stems_by_folder(output_folder):
#     """
#     Organize separated audio files into stem-specific folders with song names as filenames.

#     Args:
#         output_folder (str): Path to output folder containing separated files
#     """

#     output_path = Path(output_folder) / 'htdemucs'

#     # Create main stem folders
#     stem_folders = {
#         'bass': output_path / 'bass',
#         'drums': output_path / 'drums',
#         'vocals': output_path / 'vocals',
#         'others': output_path / 'others'
#     }

#     for folder in stem_folders.values():
#         folder.mkdir(parents=True, exist_ok=True)

#     # Find all separated audio files and reorganize them
#     for audio_file in output_path.rglob('*.wav'):
#         # Skip files already in stem folders
#         if audio_file.parent in stem_folders.values():
#             continue

#         # Get the song name (parent directory name)
#         song_name = audio_file.parent.name

#         # Determine which stem this file belongs to
#         filename = audio_file.name.lower()
#         if 'bass' in filename:
#             target_folder = stem_folders['bass']
#             # Create new filename using song name
#             new_filename = f"{song_name}.wav"
#         elif 'drums' in filename:
#             target_folder = stem_folders['drums']
#             new_filename = f"{song_name}.wav"
#         elif 'vocals' in filename:
#             target_folder = stem_folders['vocals']
#             new_filename = f"{song_name}.wav"
#         else:
#             target_folder = stem_folders['others']
#             new_filename = f"{song_name}.wav"

#         # Move file to appropriate folder with song name as filename
#         new_path = target_folder / new_filename
#         audio_file.rename(new_path)
#         print(f"Moved {audio_file.name} to {target_folder.name}/{new_filename}")

#     # Clean up empty subdirectories (original song folders)
#     for subdir in output_path.rglob('*'):
#         if subdir.is_dir() and not any(subdir.iterdir()) and subdir != output_path:
#             try:
#                 subdir.rmdir()
#             except OSError:
#                 pass  # Directory not empty or other error

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Separate audio files using Demucs 4-stem model')
#     parser.add_argument('input_folder', help='Path to folder containing audio files')
#     parser.add_argument('output_folder', help='Path to output folder for separated stems')

#     args = parser.parse_args()

#     # separate_audio_files(args.input_folder, args.output_folder)
#     organize_stems_by_folder(args.output_folder)
#     print("Audio separation completed successfully!")


#!/usr/bin/env python3
# download_ckpts.py
# Robust Google Drive downloader with validation for your JukeDrummer checkpoints.
#!/usr/bin/env python3
# download_ckpts.py — robust Google Drive downloader with confirm-token + cookie handling

import os
import sys
import argparse
import re
import time

# Optional fallback (works if installed): pip install gdown
try:
    import gdown  # type: ignore
except Exception:
    gdown = None

import requests

GDRIVE_ITEMS = {
    "vq1_target": ("1sblulUyla-R61BU5Ky9Z4QDDISDwG2v5", "ckpt/vqvae/vq1_target.pth"),
    "vq1_others": ("1YRC0jFzPw1sgQoKD0tsjQ3KUknvZf8_t", "ckpt/vqvae/vq1_others.pth"),
    "exp11":      ("1sMRrOWqE9GvtxeO1jq8iEMsJjLl4lkij", "ckpt/exp11.pkl"),
    "exp1":       ("18uw2gvEXL6yQ2dQk3eHyvgPVhkGBB-Wa", "ckpt/exp1.pkl"),
    "tracker":    ("15GjIBsGbULRyDL3ze4wsR4opzUQN809d", "ckpt/RNNBeatProc.pth"),
    "generator":  ("1Un1pm_8NaG5lUIrVTS4l0s0oWbGkllrb", "ckpt/vocoder/generator.pth"),
}

MIN_BYTES = 1 << 20  # 1 MB
DRIVE_UC = "https://drive.google.com/uc?export=download&id={id}"
DRIVE_DL = "https://drive.usercontent.google.com/download?id={id}&export=download"

HTML_HINTS = (b"<!DOCTYPE html", b"<html", b"</html")
QUOTA_HINTS = (b"download quota", b"quota exceeded", b"Too many users")
LOGIN_HINTS = (b"ServiceLogin", b"accounts.google.com", b"signin")
VIRUS_HINTS = (b"Virus scan warning", b"Download anyway")

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def is_html(buf: bytes) -> bool:
    h = buf.strip().lower()
    return any(x.lower() in h for x in HTML_HINTS)

def looks_quota(buf: bytes) -> bool:
    low = buf.lower()
    return any(x.lower() in low for x in QUOTA_HINTS)

def looks_login(buf: bytes) -> bool:
    low = buf.lower()
    return any(x.lower() in low for x in LOGIN_HINTS)

def needs_confirm(buf: bytes) -> bool:
    low = buf.lower()
    return any(x.lower() in low for x in VIRUS_HINTS)

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def write_stream(resp: requests.Response, out_path: str):
    with open(out_path, "wb") as f:
        # peek to detect HTML
        head = resp.raw.read(4096)
        if head:
            f.write(head)
        for chunk in resp.iter_content(1 << 20):
            if chunk:
                f.write(chunk)

def valid_binary(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < MIN_BYTES:
        return False
    with open(path, "rb") as f:
        head = f.read(4096)
    return not is_html(head)

def get_confirm_from_cookies(cookies: requests.cookies.RequestsCookieJar) -> str | None:
    # Google sets cookies like download_warning_<random>=<token>
    for k, v in cookies.items():
        if k.startswith("download_warning"):
            return v
    return None

def get_confirm_from_html(html_text: str) -> str | None:
    # Multiple patterns; Drive changes often
    m = re.search(r'confirm=([0-9A-Za-z_]+)', html_text)
    if m:
        return m.group(1)
    m = re.search(r'name="confirm"\s+value="([0-9A-Za-z_]+)"', html_text)
    if m:
        return m.group(1)
    return None

def download_gdrive(file_id: str, out_path: str, max_retries: int = 3, timeout: int = 60):
    ensure_dir(out_path)
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA, "Accept": "*/*"})

    # Attempt 1: direct usercontent (often bypasses interstitial)
    for attempt in range(1, max_retries + 1):
        try:
            with sess.get(DRIVE_DL.format(id=file_id), stream=True, timeout=timeout) as r:
                r.raise_for_status()
                # quick sniff without writing to file system yet
                head = r.raw.read(4096)
                if is_html(head):
                    # store body preview for diagnostics
                    more = r.raw.read(8192)
                    blob = head + more
                    if looks_quota(blob):
                        raise RuntimeError("Google Drive quota exceeded")
                    # fall through to UC confirm flow
                else:
                    # write full content
                    with open(out_path, "wb") as f:
                        if head:
                            f.write(head)
                        for chunk in r.iter_content(1 << 20):
                            if chunk:
                                f.write(chunk)
                    if valid_binary(out_path):
                        return
                    # else try UC flow
        except Exception:
            if attempt == max_retries:
                break
            time.sleep(2 * attempt)

    # Attempt 2: UC page → confirm token via cookies and/or HTML
    for attempt in range(1, max_retries + 1):
        try:
            init = sess.get(DRIVE_UC.format(id=file_id), timeout=timeout)
            init.raise_for_status()
            token = get_confirm_from_cookies(init.cookies) or get_confirm_from_html(init.text)
            if not token and needs_confirm(init.content):
                token = get_confirm_from_html(init.text)

            params = {"export": "download", "id": file_id}
            if token:
                params["confirm"] = token

            with sess.get("https://drive.google.com/uc", params=params, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                # if still HTML, abort with diagnostics
                head = r.raw.read(4096)
                if is_html(head):
                    more = r.raw.read(8192)
                    blob = head + more
                    if looks_quota(blob):
                        raise RuntimeError("Google Drive quota exceeded")
                    if looks_login(blob):
                        raise RuntimeError("File is not public; enable 'Anyone with the link'")
                    # if token not present, we likely failed extraction
                    raise RuntimeError("Received HTML instead of binary; confirm token missing")
                # write
                with open(out_path, "wb") as f:
                    if head:
                        f.write(head)
                    for chunk in r.iter_content(1 << 20):
                        if chunk:
                            f.write(chunk)

            if not valid_binary(out_path):
                raise RuntimeError("Downloaded file is not valid binary (looks like HTML)")
            return
        except Exception:
            if attempt == max_retries:
                break
            time.sleep(2 * attempt)

    # Attempt 3: gdown fallback if available
    if gdown is not None:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.cached_download(url=url, path=out_path, quiet=False, postprocess=None)
        if valid_binary(out_path):
            return

    raise RuntimeError("Failed to download a valid binary from Google Drive")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="Subset: " + ", ".join(GDRIVE_ITEMS.keys()))
    ap.add_argument("--force", action="store_true", help="Redownload even if file exists")
    args = ap.parse_args()

    targets = GDRIVE_ITEMS if not args.only else {k: v for k, v in GDRIVE_ITEMS.items() if k in args.only}
    os.makedirs("ckpt/vqvae", exist_ok=True)
    os.makedirs("ckpt/vocoder", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)

    for name, (fid, path) in targets.items():
        try:
            if os.path.exists(path) and not args.force and valid_binary(path):
                print(f"[skip] {name}: {path}")
                continue
            if os.path.exists(path) and args.force:
                os.remove(path)
            print(f"[download] {name} -> {path}")
            download_gdrive(fid, path)
            if not valid_binary(path):
                raise RuntimeError("Invalid file after download (HTML or too small)")
            print(f"[ok] {name}: {path} ({os.path.getsize(path)} bytes)")
        except Exception as e:
            print(f"[fail] {name}: {e}", file=sys.stderr)
            sys.exit(2)

    print("[done]")

if __name__ == "__main__":
    main()
