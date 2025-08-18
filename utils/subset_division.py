from email import parser
import pickle
import numpy as np
import os
import argparse
import random


# def write_subset(segments, pkl_path, valid_precentage=0.2):
#     # divide dataset into training set and validation set by Mels of complete songs from segemented clips
#     nfile = len(os.listdir('data/audio/target/'))
#     segments = sorted(segments)
#     fns = {}
#     train_set = []
#     valid_set = []
#     for s in segments:
#         if s not in fns:
#             fns[s.rsplit('_', 1)[0]] = 'valid' if len(fns) < round(nfile * valid_precentage) else 'train'
#         if fns[s.rsplit('_', 1)[0]] == 'train':
#             train_set += [s]
#         elif fns[s.rsplit('_', 1)[0]] == 'valid':
#             valid_set += [s]
#     print(len(fns))
#     with open(os.path.join(pkl_path, 'dataset.pkl'), 'wb') as f:
#         pickle.dump([sorted(train_set), sorted(valid_set)], f)

# def comparing(data_dir):
#     # comparation between 2 dataset

#     segments = os.listdir(os.path.join(data_dir, 'target'))
#     fns = []
#     for s in segments:
#         if os.path.isfile(os.path.join(data_dir, 'others', s)) and \
#             np.load(os.path.join(data_dir, 'target', s)).shape[1]==4096 and \
#             np.load(os.path.join(data_dir, 'others', s)).shape[1]==4096:
#             fns.append(s)
#     return fns

# def inference(mel_dir, pkl_path):
#     print('step 3: divide into subsets')
#     fns = comparing(mel_dir)
#     write_subset(fns, pkl_path)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--audio_dir', type=str, default=None)
#     parser.add_argument('--mel_dir', type=str, default=None)
#     parser.add_argument('--pkl_path', type=str, default=None)

#     args = parser.parse_args()
#     fns = comparing(args.mel_dir)
#     write_subset(fns, args.pkl_path)
import os
import pickle
import numpy as np
import argparse

def write_subset(segments, pkl_path, mel_dir, valid_percentage=0.2):
    print("Writing dataset split...")
    target_dir = os.path.join(mel_dir, 'target')
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    nfile = len(os.listdir(target_dir))
    print(f"Found {nfile} files in target directory for splitting.")

    segments = sorted(segments)
    fns = {}
    train_set = []
    valid_set = []

    for s in segments:
        base_name = s  #.rsplit('_', 1)[0]
        #if base_name not in fns:
        fns[base_name] = 'valid' if len(fns) < round(nfile * valid_percentage) else 'train'
        if fns[base_name] == 'train':
            train_set.append(s)
        elif fns[base_name] == 'valid':
            valid_set.append(s)

    print(f"Unique base files found: {len(fns)}")
    print(f"Train set size: {len(train_set)}, Validation set size: {len(valid_set)}")

    os.makedirs(pkl_path, exist_ok=True)
    with open(os.path.join(pkl_path, 'dataset.pkl'), 'wb') as f:
        pickle.dump([sorted(train_set), sorted(valid_set)], f)
    print("Dataset split saved to:", os.path.join(pkl_path, 'dataset.pkl'))

def comparing(data_dir):
    print("Comparing files in target and others...", data_dir)
    segments_paths = os.path.join(data_dir)
    print("segments_paths:", segments_paths)
    segments = os.listdir(segments_paths)

    fns = []
    print('segments:', segments)
    for s in segments:
        print(f"Processing segment: {s}")
        try:
            target_path = os.path.join(data_dir, 'target', s)
            others_path = os.path.join(data_dir, 'others', s)
            

            if os.path.isfile(others_path):
                
                target = np.load(target_path)
                others = np.load(others_path)

                # if target.shape[1] == 4096 and others.shape[1] == 4096:
            fns.append(s)
        except Exception as e:
            print(f"Skipping {s} due to error: {e}")
    print(f"Valid segment pairs found: {len(fns)}")
    return fns

def inference(mel_dir, pkl_path):
    print("Step 3: divide into subsets")
    fns = comparing(mel_dir)
    print('mel_dir:', mel_dir)
    if fns:
        write_subset(fns, pkl_path, mel_dir)
    else:
        print("No valid segments found. No dataset created.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mel_dir', type=str, required=True, help="Path to mel spectrogram directory")
    parser.add_argument('--pkl_path', type=str, required=True, help="Path to save dataset.pkl")

    args = parser.parse_args()
    inference(args.mel_dir, args.pkl_path)
