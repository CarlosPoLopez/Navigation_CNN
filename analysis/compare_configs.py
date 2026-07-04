"""Compares SEVERAL hyperparameter configurations against each other (picks the winner).

While kfold_summary.py summarizes ONE folder (intra-config), this script compares
SEVERAL runs/<config> folders and produces the material to justify in the report which
configuration is chosen and why:

  - For each config it reads fold{0..k-1}/curve.csv, computes the best (min) val_loss of
    each fold and summarizes the config by its MEAN +/- std across folds (the k-fold
    metric with which one config is compared against another).
  - comparison.csv  : one row per config, SORTED from best (lowest val) to worst.
  - comparison.png  : bars with error bar (+/-sigma). If two bars overlap, the
    difference is NOT significant (it is within the across-fold noise).
  - comparison_curves.png : mean val curve of each config overlaid (which one converges
    lower and earlier).

Usage (after training the k-fold of each config):
  python compare_configs.py ../runs/baseline ../runs/lr1e-3 ../runs/ep100
  # or by pattern:
  python compare_configs.py --glob "../runs/*" --k 5
The frozen test is NOT touched here: selection is done ONLY with validation.
"""

import os
import re
import csv
import glob as globmod
import argparse
import numpy as np


def label(name):
    """Formats 'lr<...>_bs<N>' as '$\\eta$=<...>, $B$=<N>' (report notation).

    If the name does not match that pattern (e.g. 'baseline'), it is returned as is.
    """
    m = re.search(r'lr([0-9.eE+-]+)_bs(\d+)', name)
    return f'$\\eta$={m.group(1)}, $B$={m.group(2)}' if m else name


def read_curve(path):
    """Returns (epochs, val) as arrays from a curve.csv.

    Ignores rows without val_loss (e.g. those of final-model mode, which do not validate).
    """
    epochs, val = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            v = row.get('val_loss', '').strip()
            if v == '':
                continue
            epochs.append(int(row['epoch']))
            val.append(float(v))
    return np.array(epochs), np.array(val)


def summarize_config(run_dir, k):
    """Summarizes a config folder. Returns a dict or None if there is no val data.

    bests: best val per fold | val_curve: mean val across folds (truncated to Lmin).
    """
    curves = []          # (epochs, val) of each fold with data
    bests, epochs_at = [], []
    for f in range(k):
        p = os.path.join(run_dir, f'fold{f}', 'curve.csv')
        if not os.path.exists(p):
            continue
        ep, val = read_curve(p)
        if len(val) == 0:
            continue
        curves.append((ep, val))
        i = int(np.argmin(val))
        bests.append(val[i]); epochs_at.append(int(ep[i]))

    if not bests:
        return None

    bests = np.array(bests)
    # Mean val curve across folds (truncated to the common minimum number of epochs).
    Lmin = min(len(v) for _, v in curves)
    ep_common = curves[0][0][:Lmin]
    val_stack = np.stack([v[:Lmin] for _, v in curves])
    return {
        'name':       os.path.basename(os.path.normpath(run_dir)),
        'n_folds':    len(bests),
        'val_mean':   float(bests.mean()),
        'val_std':    float(bests.std()),
        'val_min':    float(bests.min()),
        'val_max':    float(bests.max()),
        'epoch_mean': float(np.mean(epochs_at)),
        'ep_common':  ep_common,
        'val_curve':  val_stack.mean(0),
    }


def main(args):
    # 1) Resolve the config folders to compare.
    run_dirs = list(args.runs)
    if args.glob:
        run_dirs += sorted(d for d in globmod.glob(args.glob) if os.path.isdir(d))
    # Remove duplicates keeping the order.
    run_dirs = list(dict.fromkeys(run_dirs))
    if not run_dirs:
        raise SystemExit('No config folder was given (positional or --glob).')

    # Default out_dir: the common parent directory of the compared folders, so the
    # output lands NEXT to them (each experiment writes in its place and they do not
    # overlap between different grids). Can be forced with --out_dir.
    if args.out_dir is None:
        parents = {os.path.dirname(os.path.normpath(d)) for d in run_dirs}
        args.out_dir = parents.pop() if len(parents) == 1 else os.path.commonpath(run_dirs)

    summaries = []
    for d in run_dirs:
        r = summarize_config(d, args.k)
        if r is None:
            print(f'  WARNING: {d} without curve.csv with val -> skipped.')
            continue
        summaries.append(r)
    if not summaries:
        raise SystemExit('No folder had usable validation data.')

    # 2) Sort from best (lowest mean val) to worst.
    summaries.sort(key=lambda r: r['val_mean'])

    # 3) Console table.
    print(f'\n  Comparison of {len(summaries)} configurations (k={args.k}) - '
          f'sorted by mean val\n')
    print(f'  {"config":>16} {"folds":>5} {"mean val +/- sigma":>22} '
          f'{"range":>17} {"mean ep.":>8}')
    print(f'  {"-"*16} {"-"*5} {"-"*22} {"-"*17} {"-"*8}')
    for r in summaries:
        print(f'  {r["name"]:>16} {r["n_folds"]:>5} '
              f'{r["val_mean"]:>9.4f} +/- {r["val_std"]:<8.4f} '
              f'{r["val_min"]:>7.4f}..{r["val_max"]:<7.4f} {r["epoch_mean"]:>8.1f}')
    best = summaries[0]
    print(f'\n  Best by mean val: {best["name"]} '
          f'({best["val_mean"]:.4f} +/- {best["val_std"]:.4f})')
    # Significance note: if the 2nd overlaps with the 1st within +/-sigma.
    if len(summaries) > 1:
        second = summaries[1]
        overlap = (best['val_mean'] + best['val_std']) >= (second['val_mean'] - second['val_std'])
        if overlap:
            print(f'  NOTE: {second["name"]} overlaps within +/-sigma -> '
                  f'difference NOT conclusive (technical tie).')

    # 4) Comparison CSV.
    out_csv = os.path.join(args.out_dir, 'comparison.csv')
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['config', 'n_folds', 'val_mean', 'val_std',
                    'val_min', 'val_max', 'epoch_mean'])
        for r in summaries:
            w.writerow([r['name'], r['n_folds'],
                        f'{r["val_mean"]:.6f}', f'{r["val_std"]:.6f}',
                        f'{r["val_min"]:.6f}', f'{r["val_max"]:.6f}',
                        f'{r["epoch_mean"]:.2f}'])
    print(f'\n  Table   -> {out_csv}')

    # 5) Figures.
    try:
        import matplotlib
        matplotlib.use('Agg')   # headless (no display on the cluster)
        import matplotlib.pyplot as plt

        # 5a) Bars with error bar (+/-sigma).
        names   = [label(r['name']) for r in summaries]
        means   = [r['val_mean'] for r in summaries]
        sigmas  = [r['val_std'] for r in summaries]
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(summaries)), 5))
        x = np.arange(len(summaries))
        colors = ['tab:green'] + ['tab:blue'] * (len(summaries) - 1)  # highlights the best
        ax.bar(x, means, yerr=sigmas, capsize=5, color=colors, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_ylabel('Best val_loss (mean across folds)')
        ax.set_title(r'Comparison of configurations  (bar = $\pm\sigma$ across folds)')
        ax.grid(True, axis='y', alpha=0.3)
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, 'comparison.png')
        fig.savefig(out_png, dpi=150); plt.close(fig)
        print(f'  Bars    -> {out_png}')

        # 5b) Mean val curves overlaid.
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in summaries:
            ax.plot(r['ep_common'], r['val_curve'], linewidth=2, label=label(r['name']))
        ax.set_xlabel('Epoch'); ax.set_ylabel(r'$\mathcal{L}_v(t)$')
        ax.set_title('Mean validation curve per configuration')
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png2 = os.path.join(args.out_dir, 'comparison_curves.png')
        fig.savefig(out_png2, dpi=150); plt.close(fig)
        print(f'  Curves  -> {out_png2}')
    except Exception as e:
        print(f'  (no figures: {type(e).__name__}: {e})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Compares hyperparameter configurations by their k-fold (mean val +/- sigma).')
    p.add_argument('runs', nargs='*', help='runs/<config> folders to compare.')
    p.add_argument('--glob', default=None,
                   help='alternative pattern, e.g. "../runs/*" (added to the positional ones).')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--out_dir', default=None,
                   help='where to write comparison.csv/.png (default: the common parent of the compared folders).')
    main(p.parse_args())
