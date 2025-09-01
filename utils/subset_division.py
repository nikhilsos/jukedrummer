import os
import pickle
import argparse
import random

def inference(mel_dir, dataset_pkl_path, valid_pct=0.2, seed=1337):
    target_dir = os.path.join(mel_dir, "target")
    others_dir = os.path.join(mel_dir, "others")

    if not os.path.isdir(target_dir) or not os.path.isdir(others_dir):
        raise FileNotFoundError(f"Expected {target_dir} and {others_dir} to exist.")

    # List only .npy files
    tgt_files = {f for f in os.listdir(target_dir) if f.endswith(".npy")}
    oth_files = {f for f in os.listdir(others_dir) if f.endswith(".npy")}

    # Take common files
    common = sorted(tgt_files & oth_files)
    if not common:
        raise RuntimeError(
            f"No common files found!\nTarget={len(tgt_files)} Others={len(oth_files)}"
        )

    # Shuffle for randomness
    random.Random(seed).shuffle(common)

    # Split train/valid
    cut = int(len(common) * valid_pct)
    valid = common[:cut]
    train = common[cut:]

    # Save dataset.pkl
    os.makedirs(os.path.dirname(dataset_pkl_path) or ".", exist_ok=True)
    with open(dataset_pkl_path, "wb") as f:
        pickle.dump([train, valid], f)

    print(f"[OK] train={len(train)}, valid={len(valid)}")
    print(f"[WROTE] {dataset_pkl_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mel_dir", required=True, help="Directory containing target/ and others/")
    ap.add_argument("--dataset_pkl_path", required=True, help="Output path for dataset.pkl")
    ap.add_argument("--valid_pct", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    inference(args.mel_dir, args.dataset_pkl_path, args.valid_pct, args.seed)
