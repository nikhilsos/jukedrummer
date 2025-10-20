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
    print("Loading checkpoint from '{}'".format(filepath))
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
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        print("No checkpoint found")
        return None
    cp_list = sorted(cp_list)
    print(f"cp_g: {cp_list[-1]}")
    return cp_list[-1]


def create_splits(directory, split_ratio=0.9):
    '''
    split according to the given ratio and put filenames in training.txt and validation.txt
    '''
    fns = os.listdir(f'{directory}')
    fns = [filename for filename in fns if filename.endswith('.wav')]
    fns = sorted(fns)
    num_tr = int(len(fns) * split_ratio)
    num_val = len(fns) - num_tr
    tr_fns = fns[:num_tr]
    val_fns = fns[num_tr:]

    parent_folder_of_directory = os.path.dirname(directory)

    with open(f'{parent_folder_of_directory}/training.txt', 'w') as train_file:
        for fn in tr_fns:
            train_file.write(f'{fn}\n')
    with open(f'{parent_folder_of_directory}/validation.txt', 'w') as val_file:
        for fn in val_fns:
            val_file.write(f'{fn}\n')

    # ensure files are created, and count the lines to verify
    assert os.path.isfile(f'{parent_folder_of_directory}/training.txt')
    assert os.path.isfile(f'{parent_folder_of_directory}/validation.txt')
    with open(f'{parent_folder_of_directory}/training.txt', 'r') as train_file:
        train_lines = train_file.readlines()
    with open(f'{parent_folder_of_directory}/validation.txt', 'r') as val_file:
        val_lines = val_file.readlines()
    print(f'Created training.txt with {len(train_lines)} lines and validation.txt with {len(val_lines)} lines.')


