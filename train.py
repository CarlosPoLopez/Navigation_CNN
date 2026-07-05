import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import os

from dataset import (MazeDataset, list_samples,
                     load_test_ids, kfold_partition)
from unet import UNet


# -- Loss ------------------------------------------------------------------------
# The total loss is a fixed sum of three complementary terms (thesis eq. loss-total):
#   L = L_BCE (pixel) + L_Tversky (region) + L_clDice (connectivity).

# Tversky loss (Salehi et al. 2017): generalizes Dice by weighting the two error types
# differently. Dice treats a false positive (painting extra path) the same as a false
# negative (leaving a gap), but gaps are what break connectivity and leave the maze
# unsolved. With beta > alpha the FN weigh more than the FP, so the network prefers to
# fill in rather than to trim. With alpha = beta = 0.5 it reduces to Dice; the thesis
# uses alpha = 0.3, beta = 0.7.
def tversky_loss(pred, target, alpha=0.3, beta=0.7, eps=1e-6):
    pred = torch.sigmoid(pred)
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - ti


# -- Soft (differentiable) clDice ------------------------------------------------
# BCE and Tversky optimize the pixel overlap, NOT connectivity: the predicted path comes
# out with the right shape but full of micro-cuts, so it never joins entry and exit
# (solved rate 0% without this term). clDice measures the overlap of the CENTERLINE
# (skeleton), so it penalizes those breaks. Here we use the "soft" version of Shit
# et al. (2021): the skeleton is approximated with morphology via min/max-pooling,
# which is differentiable (skimage's skeletonize used in net_test.py is not, and only
# serves as a metric, not as a loss).

def _soft_erode(img):
    # Erosion = min-pooling (= -maxpool(-x)), separable into 3x1 and 1x3.
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def _soft_open(img):
    # Opening = dilation(erosion); dilation is a 3x3 max-pooling.
    return F.max_pool2d(_soft_erode(img), (3, 3), (1, 1), (1, 1))


def soft_skel(img, iters):
    """Soft skeleton: peels the mask layer by layer keeping what the opening removes
    (the central ridges). 'iters' >= half-width of the path."""
    skel = F.relu(img - _soft_open(img))
    for _ in range(iters):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(logits, target, iters=10, eps=1e-6):
    pred = torch.sigmoid(logits)
    skel_pred = soft_skel(pred, iters)
    skel_true = soft_skel(target, iters)
    # Tprec: how much of the predicted skeleton falls on the real path (topological precision).
    tprec = ((skel_pred * target).sum(dim=(1, 2, 3)) + eps) / (skel_pred.sum(dim=(1, 2, 3)) + eps)
    # Tsens: how much of the real skeleton is covered by the prediction (topological sensitivity).
    tsens = ((skel_true * pred).sum(dim=(1, 2, 3)) + eps) / (skel_true.sum(dim=(1, 2, 3)) + eps)
    return 1 - 2 * tprec * tsens / (tprec + tsens + eps)


def loss_fn(logits, target):
    # Total loss = BCE (pixel) + Tversky (region) + clDice (connectivity). The three
    # terms always contribute (fixed combination, thesis eq. loss-total): BCE gives a
    # smooth pixel-wise gradient everywhere, Tversky handles the global shape and the
    # class imbalance, and clDice enforces the continuity of the path.
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    region = tversky_loss(logits, target).mean()
    connectivity = cldice_loss(logits, target).mean()
    return bce + region + connectivity


# -- Train / val loops -----------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total, total_dice = 0.0, 0.0
    with torch.set_grad_enabled(train):  # PyTorch graph for computing gradients
        for x, y in loader:  # each iteration is a batch
            x, y = x.to(device), y.to(device)
            logits = model(x)  # FORWARD: draws the path
            loss = loss_fn(logits, y)  # COMPARE drawn vs real path
            if train:
                optimizer.zero_grad()
                loss.backward()  # BACKPROPAGATION: gradient of each weight w.r.t. the loss
                optimizer.step()  # updates the weights
            total += loss.item()
            # "Hard" Dice coefficient (threshold 0.5), no gradient — a segmentation-quality
            # metric logged alongside the loss (val_dice column of curve.csv).
            with torch.no_grad():
                pred_bin = (torch.sigmoid(logits) > 0.5).float()
                inter = (pred_bin * y).sum()
                total_dice += ((2 * inter + 1e-6) / (pred_bin.sum() + y.sum() + 1e-6)).item()
    # Average loss and dice per batch (train or val depending on the mode).
    return total / len(loader), total_dice / len(loader)


# -- Main ------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1) Full inventory and exclusion of the frozen TEST (never enters the k-fold).
    files = list_samples(args.data_dir)
    test_ids = load_test_ids(args.test_ids)
    pool = [f for f in files if os.path.basename(f) not in test_ids]
    print(f'Total samples: {len(files)} | frozen test: {len(test_ids)} | '
          f'train/val pool: {len(pool)}')
    if not test_ids:
        print('  WARNING: no test_id was loaded. You are training WITHOUT a held-out test. '
              'Generate test_ids.txt with `python dataset.py --data_dir ... --output test_ids.txt`.')
    elif len(pool) == len(files):
        print('  WARNING: test_ids.txt does not match any sample (check --test_ids).')

    # 2) Partition of the pool. Two modes:
    #    fold >= 0 -> normal k-fold (holds out a validation fold).
    #    fold <  0 -> FINAL MODEL: trains on the WHOLE pool, without validation
    #                 (the k-fold already decided config and number of epochs; here we
    #                 only squeeze the model).
    no_val = args.fold < 0
    if no_val:
        train_files, val_files = pool, []
        print(f'FINAL MODEL: train={len(train_files)} (whole pool, no validation)')
    else:
        idx_tr, idx_val = kfold_partition(len(pool), args.k, args.fold, seed=args.seed)
        train_files = [pool[i] for i in idx_tr]
        val_files   = [pool[i] for i in idx_val]
        print(f'Fold {args.fold}/{args.k}: train={len(train_files)}  val={len(val_files)}')

    train_ds = MazeDataset(train_files, augment=not args.no_augment)
    if args.no_augment:
        print('WARNING: augmentation DISABLED (--no_augment) -> train without D4 symmetries.')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    if not no_val:
        val_ds     = MazeDataset(val_files, augment=False)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    # Creates the network with random weights. 5-level U-Net: the extra downsample gives
    # the bottleneck more global context (8x8 px @256) to plan the path, WITHOUT lowering
    # the output resolution (which would break the thin path). 3-channel input (fixed
    # goal-conditioning).
    features = (32, 64, 128, 256, 512)
    model = UNet(in_channels=3, features=features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)  # checks val_loss every 5 epochs; if it does not drop, halves the lr

    # Output subdirectory. In final mode it is saved in fold0/ so net_test.py can
    # evaluate it with --k 1 without any changes.
    sub = 'fold0' if no_val else f'fold{args.fold}'
    ckpt_dir = os.path.join(args.ckpt_dir, sub)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, 'curve.csv')
    with open(log_path, 'w') as fh:
        fh.write('epoch,train_loss,val_loss,val_dice\n')

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, optimizer, device, train=True)

        if no_val:
            # No validation: the scheduler follows train_loss and the model of every
            # epoch is saved (ending as the LAST one, which is the final one).
            scheduler.step(train_loss)
            print(f'Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}')
            with open(log_path, 'a') as fh:
                fh.write(f'{epoch},{train_loss:.6f},,\n')
            path = os.path.join(ckpt_dir, 'best.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': float('nan'), 'train_loss': train_loss,
                        'fold': -1}, path)
        else:
            val_loss, val_dice = run_epoch(model, val_loader, optimizer, device, train=False)
            scheduler.step(val_loss)
            print(f'Epoch {epoch:3d}/{args.epochs}  '
                  f'train={train_loss:.4f}  val={val_loss:.4f}  dice={val_dice:.4f}')
            with open(log_path, 'a') as fh:
                fh.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},{val_dice:.6f}\n')
            if val_loss < best_val:
                best_val = val_loss
                path = os.path.join(ckpt_dir, 'best.pt')
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'val_loss': best_val, 'fold': args.fold}, path)
                print(f'  -> checkpoint saved ({path})')

    if no_val:
        print(f'Final model finished: {args.epochs} epochs over the whole pool '
              f'-> {os.path.join(ckpt_dir, "best.pt")}')
    else:
        print(f'Fold {args.fold} finished. Best val loss: {best_val:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data/Dataset_Matrices',
                        help='folder with the muestra_*.npz dataset (from Part 1).')
    parser.add_argument('--test_ids',   default='data/test_ids.txt',
                        help='file listing the frozen test samples (create it once with dataset.py).')
    parser.add_argument('--ckpt_dir',   default='runs/baseline',
                        help='output folder; checkpoints go to <ckpt_dir>/fold<F>/best.pt and the '
                             'loss log to <ckpt_dir>/fold<F>/curve.csv.')
    parser.add_argument('--k',          type=int,   default=5)
    parser.add_argument('--fold',       type=int,   default=0,
                        help='k-fold fold (0..k-1). Use -1 for FINAL MODEL: '
                             'trains on the whole pool without validation (saves to fold0/).')
    parser.add_argument('--seed',       type=int,   default=42)
    parser.add_argument('--batch_size', type=int,   default=4)
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--lr',         type=float, default=1e-4)
    parser.add_argument('--no_augment', action='store_true',
                        help='DISABLES data augmentation (D4 symmetries: rot90 + flips) on '
                             'train. Enabled by default. For ablation: check whether '
                             'augmentation affects the results (e.g. generates more FP).')
    main(parser.parse_args())
