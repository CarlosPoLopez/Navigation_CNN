"""Evaluates one or several models on their VALIDATION set (not the test).

Used to CHOOSE between configurations (e.g. the losses of stage C) with a metric
aligned with the goal - % solved mazes and clDice - which val_dice (the only thing in
curve.csv) does not capture. The frozen TEST is NOT touched.

Correctness key: it rebuilds EXACTLY the same validation split that training used (same
--k, --fold, --seed and the same test exclusion), so it is evaluated on the same
held-out images that the model left out while training. It reuses the net_test metrics
(Dice / clDice / solved) -> identical to the test, only the image set changes.

Model: ONE fold (not ensemble), because the screening probes are single-fold.
Loads <ckpt_dir>/fold<fold>/best.pt.

Usage:
  python eval_val.py --glob "runs/stageC/lr1e-4_bs8_goal*" --goal_channels
  python eval_val.py --ckpt_dir runs/stageC/lr1e-4_bs8_goal_tv0.7_w0 --goal_channels

Outputs:
  - <ckpt_dir>/val_metrics.csv  (one row per sample: dice/cldice/solved)
  - <ckpt_dir>/val_overlays/    (if --n_overlays>0: N PNG maze/real/pred with TP/FN/FP map;
                                 --overlay_sel best/worst/first chooses which, best by default)
  - console table sorted by % solved (selection criterion).
"""
import os
import sys
import csv
import heapq
import glob as globmod
import argparse
import numpy as np
import torch

# The core modules (dataset, unet, net_test) live in the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dataset import (MazeDataset, list_samples, load_test_ids,
                     kfold_partition)
from unet import UNet
from net_test import (dice_coef, cldice, maze_solved, save_overlays, THRESHOLD)


def val_files_for(args):
    """Rebuilds the list of VALIDATION files of the fold, same as train.py:
    pool = all samples minus the frozen test; then kfold_partition."""
    files = list_samples(args.data_dir)
    test_ids = load_test_ids(args.test_ids)
    pool = [f for f in files if os.path.basename(f) not in test_ids]
    _, idx_val = kfold_partition(len(pool), args.k, args.fold, seed=args.seed)
    return [pool[i] for i in idx_val]


def evaluate_one(ckpt_dir, val_files, args, device):
    """Evaluates the best.pt of a fold over val_files. Returns (dice, cldice, %solved, n)."""
    ckpt = os.path.join(ckpt_dir, f'fold{args.fold}', 'best.pt')
    if not os.path.exists(ckpt):
        print(f'  WARNING: {ckpt} missing (not trained yet) -> skipped.')
        return None
    state = torch.load(ckpt, map_location=device)
    feats = (32, 64, 128, 256, 512)
    model = UNet(in_channels=3, features=feats).to(device)
    model.load_state_dict(state['model'])
    model.eval()

    ds = MazeDataset(val_files, augment=False)
    dices, cldices, svs, rows = [], [], [], []
    overlays = []  # heap bounded to n_overlays: (key, counter, dict) - best/worst/first
    _ov_cnt = 0    # breaks ties in the heap without comparing the dicts
    with torch.no_grad():
        for i in range(len(ds)):
            x, y = ds[i]
            name = os.path.basename(val_files[i])
            x = x.unsqueeze(0).to(device)
            gt = (y[0] > 0.5).float()
            gt_np = gt.numpy()
            prob = torch.sigmoid(model(x))[0, 0].cpu()
            pred = (prob > THRESHOLD).float()
            pred_np = pred.numpy()
            d = dice_coef(pred, gt)
            cl = cldice(pred_np, gt_np)
            sv = maze_solved(pred_np, gt_np, args.margin)
            dices.append(d); cldices.append(cl); svs.append(sv)
            rows.append((name, d, cl, sv))
            if args.n_overlays > 0:
                entry = {'maze': x[0, 0].cpu().numpy(), 'gt': gt_np,
                         'pred': pred_np, 'name': name,
                         'dice': d, 'cldice': cl, 'solved': bool(sv)}
                if args.overlay_sel == 'first':
                    if len(overlays) < args.n_overlays:
                        overlays.append((0, _ov_cnt, entry)); _ov_cnt += 1
                else:
                    # key (solved, dice): 'best' keeps the largest, 'worst' the smallest.
                    key = (sv, d) if args.overlay_sel == 'best' else (-sv, -d)
                    if len(overlays) < args.n_overlays:
                        heapq.heappush(overlays, (key, _ov_cnt, entry)); _ov_cnt += 1
                    elif key > overlays[0][0]:
                        heapq.heapreplace(overlays, (key, _ov_cnt, entry)); _ov_cnt += 1

    if overlays:
        # 'first' keeps the order of appearance; best/worst are sorted best->worst.
        if args.overlay_sel == 'first':
            samples_ov = [e for _, _, e in overlays]
        else:
            samples_ov = [e for _, _, e in sorted(overlays, key=lambda t: t[0], reverse=True)]
        save_overlays(samples_ov, os.path.join(ckpt_dir, 'val_overlays'))

    out_csv = os.path.join(ckpt_dir, 'val_metrics.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample', 'dice_val', 'cldice_val', 'solved_val'])
        for name, d, cl, sv in rows:
            w.writerow([name, f'{d:.6f}', f'{cl:.6f}', int(sv)])

    dices = np.array(dices); cldices = np.array(cldices); svs = np.array(svs, dtype=float)
    return dices.mean(), np.nanmean(cldices), 100 * svs.mean(), len(svs)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.glob:
        ckpt_dirs = sorted(d for d in globmod.glob(args.glob) if os.path.isdir(d))
        if not ckpt_dirs:
            raise SystemExit(f'The pattern {args.glob!r} found no folders.')
    else:
        ckpt_dirs = [args.ckpt_dir]

    val_files = val_files_for(args)
    print(f'\n  Evaluation on VALIDATION (fold {args.fold}, k={args.k}, seed={args.seed}) '
          f'- {len(val_files)} held-out samples\n')
    print(f'  {"config":42} {"Dice":>7} {"clDice":>8} {"solved":>10}')
    print(f'  {"-"*42} {"-"*7} {"-"*8} {"-"*10}')

    res = []
    for d in ckpt_dirs:
        r = evaluate_one(d, val_files, args, device)
        if r is None:
            continue
        dice, cl, pct, n = r
        res.append((os.path.basename(d.rstrip('/')), dice, cl, pct))

    # Sort by % solved (selection criterion), tie-break by clDice.
    res.sort(key=lambda t: (t[3], t[2]), reverse=True)
    for name, dice, cl, pct in res:
        print(f'  {name:42} {dice:7.3f} {cl:8.3f} {pct:9.1f}%')
    print(f'\n  Criterion: highest % solved (tie-break clDice). The test was NOT touched.')
    if res:
        print(f'  -> Winner on validation: {res[0][0]}  ({res[0][3]:.1f}% solved)\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='data/Dataset_Matrices',
                   help='folder with the muestra_*.npz dataset (from Part 1).')
    p.add_argument('--test_ids',   default='data/test_ids.txt',
                   help='file listing the frozen test samples (excluded from the k-fold).')
    p.add_argument('--ckpt_dir',   default=None, help='a single config (or use --glob).')
    p.add_argument('--glob',       default=None, help='pattern of several configs, e.g. "runs/stageC/*".')
    p.add_argument('--k',          type=int,   default=5)
    p.add_argument('--fold',       type=int,   default=0)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--margin',     type=int,   default=8,
                   help='tolerance (px) at entry/exit for the solving BFS.')
    p.add_argument('--n_overlays', type=int, default=0,
                   help='saves N overlays (maze/real/pred with TP/FN/FP map) in <ckpt_dir>/val_overlays/.')
    p.add_argument('--overlay_sel', choices=['first', 'best', 'worst'], default='best',
                   help="which samples to save as overlay: 'best' (N best, default), "
                        "'worst' (N worst) or 'first' (N first). Order: solved desc/asc, then Dice.")
    args = p.parse_args()
    if not args.glob and not args.ckpt_dir:
        p.error('provide --ckpt_dir or --glob.')
    main(args)
