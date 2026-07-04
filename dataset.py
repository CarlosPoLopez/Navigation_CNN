import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import glob
import os
import random
import argparse

THRESHOLD = 0.727   # same as in the dataset generation/filtering


def _square(h, w, r0, r1, c0, c1):
    """float32 mask (h,w) with a filled rectangle [r0:r1, c0:c1] (=1, rest 0)."""
    m = np.zeros((h, w), dtype=np.float32)
    m[r0:r1, c0:c1] = 1.0
    return m


# -- Inventory and splits ---------------------------------------------------------

# Finds the sample files and sorts them alphabetically.
def list_samples(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, 'muestra_*.npz')))
    if not files:
        raise FileNotFoundError(f'No muestra_*.npz files found in {data_dir}')
    return files


# Freezes 10% of the dataset for test (stores the file names in the .txt).
def create_test_split(data_dir, test_frac=0.1, seed=42, output='test_ids.txt'):
    files = list_samples(data_dir)
    names = [os.path.basename(f) for f in files]
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = max(1, int(len(names) * test_frac))
    test = sorted(names[:n_test])
    with open(output, 'w') as fh:
        fh.write('\n'.join(test) + '\n')
    print(f'Test frozen: {len(test)}/{len(files)} samples -> {output}')
    return test


# Reads test_ids.txt (one sample name per line). Returns an empty set if it does not exist.
def load_test_ids(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


# Splits the non-test samples into k folds (k-1 for train, 1 for val).
# The fixed seed guarantees a fixed split across the k=5 runs, so it is comparable.
def kfold_partition(n, k, fold, seed=42):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    val = idx[fold::k]
    val_set = set(val)
    train = [i for i in idx if i not in val_set]
    return train, val


# -- Dataset ----------------------------------------------------------------------

class MazeDataset(Dataset):
    """Takes a LIST of files (not a directory), so that train.py controls exactly
    which samples go into each split. augment=True only on train."""

    def __init__(self, files, img_size=256, augment=False,
                 goal_channels=False, marker_size=50, marker_border=5):
        self.files = list(files)
        self.img_size = img_size
        self.augment = augment
        # goal_channels: adds 2 input channels with the entry and exit "rooms" of the
        # maze. They are marker_size x marker_size (50x50) squares at marker_border (5)
        # px from the border, SAME as in the generation (entry [5:55,5:55], exit
        # [N-55:N-5, N-55:N-5]).
        self.goal_channels = goal_channels
        self.marker_size = marker_size
        self.marker_border = marker_border
        if not self.files:
            raise ValueError('MazeDataset received an empty list of files')

    def __len__(self):
        return len(self.files)

    def _aug(self, x, y):
        # Same transform applied to input and mask (square symmetries, group D4).
        # A rotated/reflected maze is still valid, with its solution transformed alike.
        k = random.randint(0, 3)
        if k:
            x = torch.rot90(x, k, dims=(1, 2))
            y = torch.rot90(y, k, dims=(1, 2))
        if random.random() < 0.5:
            x = torch.flip(x, dims=(2,))   # horizontal flip
            y = torch.flip(y, dims=(2,))
        if random.random() < 0.5:
            x = torch.flip(x, dims=(1,))   # vertical flip
            y = torch.flip(y, dims=(1,))
        return x, y

    # Reads a sample and prepares it.
    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        # Input: maze as field F (-3=wall, 0=corridor) -> binary (1=wall, 0=corridor)
        maze = (data['laberinto'] < 0).astype(np.float32)
        # Label: final field u -> binary (1=path, 0=background)
        solution = (data['solucion'] >= THRESHOLD).astype(np.float32)

        if self.goal_channels:
            # 2 extra channels: a 50x50 square at the entry "room" [5:55,5:55] and at
            # the exit one [H-55:H-5, W-55:W-5], SAME as the generation. They are built
            # BEFORE augmentation so they rotate/flip together with the maze (always
            # paired with their real endpoints) and get resized the same way. The
            # network no longer has to guess the endpoints.
            h, w = maze.shape
            b, s = self.marker_border, self.marker_size
            start_ch = _square(h, w, b, b + s, b, b + s)
            end_ch   = _square(h, w, h - b - s, h - b, w - b - s, w - b)
            x = torch.tensor(np.stack([maze, start_ch, end_ch], 0))  # (3,H,W)
        else:
            x = torch.tensor(maze).unsqueeze(0)   # (1, H, W) --> adds a channel (color)
        y = torch.tensor(solution).unsqueeze(0)

        if self.augment:
            x, y = self._aug(x, y)

        # Resize to img_size x img_size to unify sizes.
        if self.img_size is not None:
            x = F.interpolate(x.unsqueeze(0), size=self.img_size, mode='nearest').squeeze(0)
            y = F.interpolate(y.unsqueeze(0), size=self.img_size, mode='nearest').squeeze(0)

        return x, y


# -- CLI: freeze the test set ONCE ------------------------------------------------
# Usage (only once, on the final dataset):
#   python dataset.py --data_dir /path/to/Dataset_Matrices \
#                     --output   /path/to/test_ids.txt

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Freeze the test set (test_ids.txt).')
    p.add_argument('--data_dir', required=True)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='test_ids.txt')
    a = p.parse_args()
    create_test_split(a.data_dir, a.test_frac, a.seed, a.output)
