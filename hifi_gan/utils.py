import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def create_splits(directory, split_ratio=0.1):
    '''
    split according to the given ratio and put filenames in training.txt and validation.txt
    '''
    fns =  os.listdir(f'{directory}/wavs')
    fns = [f for f in fns if f.endswith('.wav')]
    fns = sorted(fns)
    num_val = int(len(fns)*split_ratio)
    num_tr = len(fns) - num_val
    tr_fns = fns[:num_tr]
    val_fns = fns[num_tr:]
    with open(f'{directory}/training.txt', 'w') as f:
        for fn in tr_fns:
            f.write(f'{fn}\n')
    with open(f'{directory}/validation.txt', 'w') as f:
        for fn in val_fns:
            f.write(f'{fn}\n')
    print(f'create {len(tr_fns)} training data and {len(val_fns)} validation data')

    
