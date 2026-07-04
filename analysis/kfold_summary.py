"""Summarizes the k-fold baseline for the report.

Reads runs/baseline/fold{0..k-1}/curve.csv (epoch,train_loss,val_loss) and produces:
  - Console table: best (min) val_loss of each fold + epoch, and MEAN +/- std across
    folds (the k-fold metric with which a configuration is compared).
  - kfold_summary.csv : mean curve across folds (epoch, train_mean/std, val_mean/std).
  - kfold_summary.png : mean val and train with a +/-sigma band, and the curves of each
    fold in the background.

Usage (after training the 5 folds; it is light, the login node is enough):
  python kfold_summary.py --runs_dir ../runs/baseline --k 5
"""

import os
import re
import csv
import argparse
import numpy as np


def label(name):
    """Formats 'lr<...>_bs<N>' as '$\\eta$=<...>, $B$=<N>' (report notation).

    If the name does not match that pattern (e.g. 'baseline'), it is returned as is.
    """
    m = re.search(r'lr([0-9.eE+-]+)_bs(\d+)', name)
    return f'$\\eta$={m.group(1)}, $B$={m.group(2)}' if m else name


def read_curve(path):
    """Returns (epochs, train, val) as numpy arrays from a curve.csv."""
    epochs, tr, val = [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row['epoch']))
            tr.append(float(row['train_loss']))
            val.append(float(row['val_loss']))
    return np.array(epochs), np.array(tr), np.array(val)


def main(args):
    # -- 1. Load the available folds -----------------------------------------------
    folds = {}
    for k in range(args.k):
        p = os.path.join(args.runs_dir, f'fold{k}', 'curve.csv')
        if not os.path.exists(p):
            print(f'  WARNING: {p} missing (fold {k} not trained or still running?)')
            continue
        folds[k] = read_curve(p)

    if not folds:
        raise SystemExit(f'No curve.csv found in {args.runs_dir}')

    order = sorted(folds)

    # -- 2. Best val per fold + mean across folds ----------------------------------
    print(f'\n  k-fold baseline - {args.runs_dir}')
    print(f'\n  {"fold":>4} {"best_val":>10} {"epoch":>6} {"epochs":>7}')
    print(f'  {"-"*4} {"-"*10} {"-"*6} {"-"*7}')
    bests = []
    for k in order:
        ep, tr, val = folds[k]
        i = int(np.argmin(val))
        bests.append(val[i])
        print(f'  {k:>4} {val[i]:>10.4f} {ep[i]:>6} {len(ep):>7}')
    bests = np.array(bests)
    print(f'\n  MEAN best val : {bests.mean():.4f} ± {bests.std():.4f}  (n={len(bests)})')
    print(f'  range         : {bests.min():.4f} ... {bests.max():.4f}')

    # -- 3. Mean curve across folds (aligned by epoch) -----------------------------
    # Truncate to the common minimum number of epochs in case a fold ran fewer.
    Lmin = min(len(folds[k][0]) for k in order)
    ep_common = folds[order[0]][0][:Lmin]
    tr_stack  = np.stack([folds[k][1][:Lmin] for k in order])
    val_stack = np.stack([folds[k][2][:Lmin] for k in order])
    tr_mean,  tr_std  = tr_stack.mean(0),  tr_stack.std(0)
    val_mean, val_std = val_stack.mean(0), val_stack.std(0)

    out_csv = os.path.join(args.runs_dir, 'kfold_summary.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['epoch', 'train_mean', 'train_std', 'val_mean', 'val_std'])
        for i in range(Lmin):
            w.writerow([int(ep_common[i]),
                        f'{tr_mean[i]:.6f}', f'{tr_std[i]:.6f}',
                        f'{val_mean[i]:.6f}', f'{val_std[i]:.6f}'])
    print(f'\n  Mean curve -> {out_csv}')

    # -- 4. Figure for the report --------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')   # headless (no display on the cluster)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        # Individual curves of each fold (val) in the background.
        for i, k in enumerate(order):
            ep, _, val = folds[k]
            ax.plot(ep, val, color='tab:blue', alpha=0.20, linewidth=1,
                    label='individual folds (validation)' if i == 0 else None)
        # Means with +/-sigma band.
        ax.plot(ep_common, val_mean, color='tab:blue', linewidth=2.5,
                label=r'$\mathcal{L}_v(t)$ - validation (mean)')
        ax.fill_between(ep_common, val_mean - val_std, val_mean + val_std,
                        color='tab:blue', alpha=0.15)
        ax.plot(ep_common, tr_mean, color='tab:orange', linewidth=2.5,
                label=r'$\mathcal{L}(t)$ - training (mean)')
        ax.fill_between(ep_common, tr_mean - tr_std, tr_mean + tr_std,
                        color='tab:orange', alpha=0.15)
        ax.set_xlabel('Epoch')
        cfg_name = label(os.path.basename(os.path.normpath(args.runs_dir)))
        ax.set_title(f'Training vs validation - {len(folds)} folds, {cfg_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_png = os.path.join(args.runs_dir, 'kfold_summary.png')
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f'  Figure     -> {out_png}')
    except Exception as e:
        print(f'  (no figure: {type(e).__name__}: {e})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Summarizes the k-fold: mean val + mean curve for the report.')
    p.add_argument('--runs_dir', default='runs/baseline',
                   help='config folder with fold*/curve.csv; the summary is written here '
                        '(kfold_summary.csv and kfold_summary.png).')
    p.add_argument('--k', type=int, default=5)
    main(p.parse_args())
