"""Analyzes the ROBUSTNESS of the hyperparameter selection criterion.

compare_configs.py picks the winner by the MINIMUM of the val curve (best epoch per
fold). That pointwise minimum is a BIASED estimator when the curve oscillates: taking
the minimum of a noisy series captures the bottom of the noise, not the real
performance. Since the val noise grows as the batch size drops, the minimum criterion
artificially favors small batches. This script quantifies that effect so it can be
justified in the report.

For each config runs/<lr_bs>/fold{0..k-1}/curve.csv it computes (mean +/- sigma across
folds):
  - min_val : pointwise minimum of val           (current criterion; biased by noise)
  - robmin  : minimum of the moving average (w)   (ROBUST minimum, no cherry-picking)
  - tail10  : mean of the last T epochs           (STABILIZED performance at the end)
  - slope   : OLS slope of the last S epochs       (x1000; <0 = still decreasing)
  - noise   : std of the differences val[t]-val[t-1] in the 2nd half (oscillation)

The bias of the minimum is (robmin - min_val): the larger, the more of a "gift" the
noise is.

Outputs (--out_dir, default .):
  robustness.csv               : table, one row per config.
  robustness_noise.png         : noise per config grouped by batch size.
  robustness_bias.png          : min_val vs robmin vs tail10 per config (the gap
                                 min->robmin is the noise bias).
  robustness_example_curves.png: at the SAME lr, one noisy val curve (smaller bs) vs a
                                 smooth one (larger bs), with their pointwise-extremum
                                 and last-epochs-mean marks. The lr is fixed with
                                 --lr_example (auto if not given) so the comparison
                                 isolates the batch size, not the lr.

Usage:
  python robustness_analysis.py --glob "runs/lr*_bs*" --k 5 --out_dir runs
The frozen test is NOT touched: this is validation analysis.
"""

import os
import re
import csv
import glob as globmod
import argparse
import numpy as np


def read_val(path, col='val_loss'):
    """Returns the array of column `col` of a curve.csv (ignores empty rows)."""
    val = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            v = (row.get(col) or '').strip()
            if v != '':
                val.append(float(v))
    return np.array(val)


def moving_avg(v, w):
    """Causal moving average of window w (returns len(v)-w+1 points)."""
    return np.convolve(v, np.ones(w) / w, mode='valid')


def fold_metrics(v, tail, slope_n, mvwin, higher_better=True):
    """The five metrics over the val curve of ONE fold.

    higher_better=False (val_loss): the relevant extremum is the MINIMUM -> min_val/robmin.
    higher_better=True  (val_dice): the relevant extremum is the MAXIMUM -> max_val/robmax.
    tail10, slope and noise do not depend on the direction."""
    ep = np.arange(len(v))
    mm = moving_avg(v, mvwin)
    if higher_better:
        best, rob = float(v.max()), float(mm.max())
        kb, kr = 'max_val', 'robmax'
    else:
        best, rob = float(v.min()), float(mm.min())
        kb, kr = 'min_val', 'robmin'
    return {
        kb: best,
        kr: rob,
        'tail10':  float(v[-tail:].mean()),
        'slope':   float(np.polyfit(ep[-slope_n:], v[-slope_n:], 1)[0] * 1000),
        'noise':   float(np.std(np.diff(v[len(v) // 2:]))),
    }


def bs_of(name):
    """Extracts the batch size from the name 'lr<...>_bs<N>' (None if it does not match)."""
    m = re.search(r'_bs(\d+)', name)
    return int(m.group(1)) if m else None


def lr_of(name):
    """Extracts the learning rate (float) from 'lr<...>_bs<N>' (None if it does not match)."""
    m = re.search(r'lr([0-9.eE+-]+)_bs', name)
    return float(m.group(1)) if m else None


def lr_str(name):
    """Extracts the learning rate as the original TEXT ('1e-4') for axis labels."""
    m = re.search(r'lr([0-9.eE+-]+)_bs', name)
    return m.group(1) if m else name


def grouped_pos(keys, gap=0.8):
    """x positions for grouped bars: leaves a `gap` when switching group.

    `keys` is the list (already sorted by group) of each bar's group. Returns
    (xs, centers) where centers={key: mean x of the group} to label each group.
    """
    xs, x, prev = [], 0.0, None
    for k in keys:
        if prev is not None and k != prev:
            x += gap
        xs.append(x); x += 1.0; prev = k
    xs = np.array(xs)
    centers = {k: xs[[i for i, c in enumerate(keys) if c == k]].mean()
               for k in dict.fromkeys(keys)}
    return xs, centers


def summarize(run_dir, args):
    """Averages the metrics across folds. Returns a dict or None if there is no data."""
    per_fold = []
    for f in range(args.k):
        p = os.path.join(run_dir, f'fold{f}', 'curve.csv')
        if not os.path.exists(p):
            continue
        v = read_val(p, args.col)
        if len(v) >= max(args.tail, args.slope_n, args.mvwin):
            per_fold.append(fold_metrics(v, args.tail, args.slope_n, args.mvwin,
                                         higher_better=args.higher_better))
    if not per_fold:
        return None
    keys = per_fold[0].keys()
    out = {'name': os.path.basename(os.path.normpath(run_dir)),
           'n_folds': len(per_fold),
           'bs': bs_of(os.path.basename(os.path.normpath(run_dir)))}
    for c in keys:
        vals = np.array([d[c] for d in per_fold])
        out[f'{c}_mean'] = float(vals.mean())
        out[f'{c}_std']  = float(vals.std())
    # bias = "noise gift" of the pointwise extremum vs the robust one (always >= 0):
    #   val_loss: robmin - min_val (the pointwise min falls lower than sustainable)
    #   val_dice: max_val - robmax (the pointwise max rises higher than sustainable)
    if args.higher_better:
        out['bias'] = out['max_val_mean'] - out['robmax_mean']
    else:
        out['bias'] = out['robmin_mean'] - out['min_val_mean']
    return out


def main(args):
    run_dirs = sorted(d for d in globmod.glob(args.glob) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f'The pattern {args.glob!r} found no folders.')

    # Default out_dir: the common parent of the analyzed folders (the output lands next
    # to them, not overlapping between different grids). Forced with --out_dir.
    if args.out_dir is None:
        parents = {os.path.dirname(os.path.normpath(d)) for d in run_dirs}
        args.out_dir = parents.pop() if len(parents) == 1 else os.path.commonpath(run_dirs)

    res = [r for r in (summarize(d, args) for d in run_dirs) if r is not None]
    if not res:
        raise SystemExit('No folder had a usable val curve.csv.')
    # Extremum keys depending on the metric direction.
    kb = 'max_val' if args.higher_better else 'min_val'
    kr = 'robmax' if args.higher_better else 'robmin'
    # order = pointwise criterion (best first): loss ascending, dice descending.
    res.sort(key=lambda r: r[f'{kb}_mean'], reverse=args.higher_better)

    # -- Console table -------------------------------------------------------------
    print(f'\n  Robustness of {len(res)} configurations (k={args.k}, col={args.col}) - '
          f'tail={args.tail} slope={args.slope_n} mvwin={args.mvwin}\n')
    head = (f'  {"config":>24} {kb:>8} {kr:>8} {"tail10":>8} '
            f'{"slope":>8} {"noise":>8} {"bias":>8}')
    print(head); print('  ' + '-' * (len(head) - 2))
    for r in res:
        print(f'  {r["name"]:>24} {r[f"{kb}_mean"]:>8.4f} {r[f"{kr}_mean"]:>8.4f} '
              f'{r["tail10_mean"]:>8.4f} {r["slope_mean"]:>8.3f} {r["noise_mean"]:>8.4f} '
              f'{r["bias"]:>8.4f}')
    if args.higher_better:
        print(f'\n  bias = max_val - robmax (how much the pointwise maximum overestimates due to noise)')
    else:
        print(f'\n  bias = robmin - min_val (how much the pointwise minimum overestimates due to noise)')

    # -- CSV -----------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f'robustness_{args.col}.csv'
                           if args.col != 'val_loss' else 'robustness.csv')
    fields = ['name', 'bs', 'n_folds',
              f'{kb}_mean', f'{kb}_std', f'{kr}_mean', f'{kr}_std',
              'tail10_mean', 'tail10_std', 'slope_mean', 'slope_std',
              'noise_mean', 'noise_std', 'bias']
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in res:
            w.writerow({k: (f'{r[k]:.6f}' if isinstance(r[k], float) else r[k])
                        for k in fields})
    print(f'\n  Table   -> {out_csv}')

    # -- Figures -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Palette by batch size: fixed for the usual ones, and the rest (e.g. bs16) take
        # a color from a cycle so they do not all fall into gray. The legend is built
        # from the bs actually present (sorted), not a fixed list.
        col_bs = {2: 'tab:red', 4: 'tab:orange', 8: 'tab:green', 16: 'tab:blue'}
        extra = iter(['tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan'])
        for b in sorted({r['bs'] for r in res if r['bs'] is not None}):
            if b not in col_bs:
                col_bs[b] = next(extra, 'gray')

        # For the bars the configs are re-sorted GROUPED by batch size (bs DESCENDING,
        # lr descending within each group) instead of by val. This way the X axis goes
        # bs16->...->bs2 and the trend "noise and bias GO UP as bs drops" reads as an
        # ascending slope towards the right (message of section 4.3). The val order stays
        # intact in the table and the CSV.
        res_g = sorted(res, key=lambda r: (-(r['bs'] or 0), -(lr_of(r['name']) or 0)))
        xs, centers = grouped_pos([r['bs'] for r in res_g])

        # NON-CONVERGED configs: atypically bad performance (val extremum more than 2
        # sigma worse than the grid mean) AND a curve still not stabilized (final slope
        # still improving). Their metrics and their noise are an artifact of
        # undertraining, not comparable with the rest: they are MARKED with a hatch and
        # left to be clipped by the axis so as not to squash the others. Objective
        # criterion, not manual. (For val_dice the direction flips: bad = low, improving
        # = slope >0.)
        _kbv = np.array([r[f'{kb}_mean'] for r in res])
        _mu, _sd = float(_kbv.mean()), float(_kbv.std())
        def _no_conv(r):
            if args.higher_better:
                return r[f'{kb}_mean'] < _mu - 2 * _sd and r['slope_mean'] > 0.1
            return r[f'{kb}_mean'] > _mu + 2 * _sd and r['slope_mean'] < -0.1
        flags = [_no_conv(r) for r in res_g]
        if any(flags):
            print('  not converged (hatched): '
                  + ', '.join(r['name'] for r, f in zip(res_g, flags) if f))

        def _group_axes(ax, ymax):
            """Separators between bs groups and a 'bs=N' label above each group."""
            for b, cx in centers.items():
                ax.text(cx, ymax, f'$\\mathbf{{B={b}}}$', ha='center', va='bottom',
                        fontsize=11, color=col_bs.get(b, 'gray'))
            borders = sorted(centers.values())
            for a, b in zip(borders[:-1], borders[1:]):
                ax.axvline((a + b) / 2, color='lightgray', lw=1, zorder=0)

        # 1) Noise per config, color = batch size, grouped by bs. The non-converged ones
        #    go with a hatch and black edge; the axis is capped at the largest whisker of
        #    the CONVERGED ones, so the whisker (not the bar) of the non-converged one is
        #    clipped and does not squash the rest. The bar keeps its REAL height.
        fig, ax = plt.subplots(figsize=(max(6, 1.0 * len(res) + 2), 5.2))
        for x, r, fl in zip(xs, res_g, flags):
            c = col_bs.get(r['bs'], 'gray')
            if fl:
                ax.bar(x, r['noise_mean'], yerr=r['noise_std'], capsize=4,
                       color=c, alpha=0.6, edgecolor='black', lw=1.6, hatch='///')
            else:
                ax.bar(x, r['noise_mean'], yerr=r['noise_std'], capsize=4, color=c, alpha=0.85)
        ax.set_xticks(xs); ax.set_xticklabels([f'$\\eta$={lr_str(r["name"])}' for r in res_g],
                                              rotation=45, ha='right')
        ax.set_ylabel(r'Noise $\rho = \sigma(\mathcal{L}_v(t)-\mathcal{L}_v(t-1))$')
        ax.set_title(r'Validation noise $\rho$ (grouped by batch size)')
        _conv = [r for r, fl in zip(res_g, flags) if not fl] or res_g
        ymax = max(r['noise_mean'] + r['noise_std'] for r in _conv)
        ax.set_ylim(top=ymax * 1.06); _group_axes(ax, ymax)
        ax.grid(True, axis='y', alpha=0.3); fig.tight_layout()
        p1 = os.path.join(args.out_dir, 'robustness_noise.png')
        fig.savefig(p1, dpi=150); plt.close(fig); print(f'  Noise   -> {p1}')

        # 2) pointwise vs robust vs tail10 extremum (the gap is the noise bias), grouped
        #    by bs to see the gap grow as the batch drops.
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(res) + 2), 6.0))
        w = 0.27
        label_kb = r'$\mathcal{L}_{\max}$' if args.higher_better else r'$\mathcal{L}_{\min}$'
        label_kr = r'$\mathcal{L}_{\mathrm{robmax}}$' if args.higher_better else r'$\mathcal{L}_{\mathrm{robmin}}$'
        label_tail = r'$\mathcal{L}_{\mathrm{tail}}$'
        # Bars at their REAL height (the non-converged ones will be clipped by the axis);
        # the non-converged ones with hatch + edge and no whisker (their extremum is out
        # of scale).
        def _bars(off, key, color, label):
            for i, (x, r, fl) in enumerate(zip(xs, res_g, flags)):
                lab = label if i == 0 else None
                if fl:
                    ax.bar(x + off, r[f'{key}_mean'], w, color=color, alpha=0.5,
                           edgecolor='black', lw=1.2, hatch='///', label=lab)
                else:
                    ax.bar(x + off, r[f'{key}_mean'], w, yerr=r[f'{key}_std'],
                           capsize=3, label=lab, color=color)
        _bars(-w,  kb,       'tab:blue',   label_kb)
        _bars(0.0, kr,       'tab:purple', label_kr)
        _bars(+w,  'tail10', 'tab:gray',   label_tail)
        ax.set_xticks(xs); ax.set_xticklabels([f'$\\eta$={lr_str(r["name"])}' for r in res_g],
                                              rotation=45, ha='right')
        # axis ymin/ymax from the CONVERGED ones (the non-converged ones fall out of
        # scale, clipped, with a ▲ mark above to signal it).
        _conv = [r for r, fl in zip(res_g, flags) if not fl] or res_g
        ymin = min(r[f'{kb}_mean'] for r in _conv) - 0.02
        ymax = max(max(r[f'{kb}_mean'] + r[f'{kb}_std'], r[f'{kr}_mean'] + r[f'{kr}_std'],
                       r['tail10_mean'] + r['tail10_std']) for r in _conv)
        rng = ymax - ymin
        ax.set_ylim(bottom=ymin, top=ymax + rng * 0.20)
        if not args.higher_better:   # val_loss: the bad extremum sticks out UPWARDS -> ▲
            for x, fl in zip(xs, flags):
                if fl:
                    ax.plot([x - w, x, x + w], [ymax + rng * 0.12] * 3, marker='^',
                            color='black', ms=8, ls='none', zorder=5)
        ax.set_title('Evaluation of different hyperparameter configurations')
        _group_axes(ax, ymax + rng * 0.02)
        ax.legend(loc='lower right'); ax.grid(True, axis='y', alpha=0.3); fig.tight_layout()
        p2 = os.path.join(args.out_dir, 'robustness_bias.png')
        fig.savefig(p2, dpi=150); plt.close(fig); print(f'  Bias    -> {p2}')

        # 3) Visual example: SAME lr, small bs (noisy) vs large bs (smooth) of fold 0.
        #    Before, the best bs2 and the best bs8 were taken, which could have a
        #    DIFFERENT lr -> the comparison mixed batch size and lr. Fixing the lr
        #    isolates the batch size effect (which is what the figure illustrates).
        def _cfg(bs, lr):
            for r in res:
                if r['bs'] == bs and lr_of(r['name']) == lr:
                    return r
            return None
        # lr with BOTH bs2 and bs8 available (the only ones comparable at fixed lr).
        lrs_pair = sorted(lr for lr in {lr_of(r['name']) for r in res
                                        if lr_of(r['name']) is not None}
                          if _cfg(2, lr) and _cfg(8, lr))
        if args.lr_example is not None:
            lr_target = float(args.lr_example)
        elif lrs_pair:
            # auto: among the lr with a bs2/bs8 pair, the one whose bs8 has the best extremum.
            lr_target = (max if args.higher_better else min)(
                lrs_pair, key=lambda lr: _cfg(8, lr)[f'{kb}_mean'])
        else:
            lr_target = None
        cfg_bs2 = _cfg(2, lr_target) if lr_target is not None else None
        cfg_bs8 = _cfg(8, lr_target) if lr_target is not None else None
        if cfg_bs2 and cfg_bs8:
            # the two curves (v, color, bs) are kept to reuse them in the inset.
            curves_data = []
            for cfg, color in ((cfg_bs2, 'tab:red'), (cfg_bs8, 'tab:green')):
                v = read_val(os.path.join(args.glob.rsplit('/', 1)[0], cfg['name'], 'fold0', 'curve.csv'), args.col)
                curves_data.append((v, color, cfg['bs']))
            fig, ax = plt.subplots(figsize=(8, 5))
            for v, color, bs in curves_data:
                ep = np.arange(len(v))
                ax.plot(ep, v, color=color, lw=1.5, alpha=0.9, label=f"$\\eta$={lr_target:g}, $B$={bs} (fold0)")
                ax.axhline(v.max() if args.higher_better else v.min(), color=color, ls=(0, (6, 3)), lw=2, alpha=0.9)
                ax.axhline(v[-args.tail:].mean(), color=color, ls=(0, (1, 1.5)), lw=2.2, alpha=0.9)
            ylabel = r'$\mathcal{L}_v(t)$' if args.col == 'val_loss' else r'$\mathrm{Dice}_v(t)$'
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
            ax.set_title(f'Noisy ($B$ small) vs smooth ($B$ large) curve - same $\\eta$={lr_target:g}\n'
                         f'--- = pointwise {"max" if args.higher_better else "min"}   ···  = mean of last epochs')
            ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
            # Optional inset: at a small loss scale the noise of the small bs is squashed
            # against the floor of the full curve and cannot be told apart from the large
            # bs. --zoom_from adds a box with the last epochs (from that epoch), with
            # connectors to the two BOTTOM corners of the box.
            if args.zoom_from is not None:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
                z = int(args.zoom_from)
                axins = inset_axes(ax, width='48%', height='42%', loc='center right', borderpad=1.2)
                for v, color, _bs in curves_data:
                    ep = np.arange(len(v)); m = ep >= z
                    axins.plot(ep[m], v[m], color=color, lw=1.3, alpha=0.9)
                    axins.axhline(v.max() if args.higher_better else v.min(), color=color, ls=(0, (6, 3)), lw=1.6, alpha=0.9)
                    axins.axhline(v[-args.tail:].mean(), color=color, ls=(0, (1, 1.5)), lw=1.8, alpha=0.9)
                sub = np.concatenate([v[z:] for v, _c, _b in curves_data])
                refs = [(v.max() if args.higher_better else v.min()) for v, _c, _b in curves_data]
                lo = min(sub.min(), min(refs)) - 0.001
                hi = max(sub.max(), max(refs)) + 0.002
                nmax = max(len(v) for v, _c, _b in curves_data) - 1
                axins.set_xlim(z, nmax); axins.set_ylim(lo, hi)
                axins.tick_params(labelsize=7); axins.grid(True, alpha=0.3)
                mark_inset(ax, axins, loc1=3, loc2=4, fc='none', ec='0.5', lw=0.8)
            fig.tight_layout()
            p3 = os.path.join(args.out_dir, 'robustness_example_curves.png')
            fig.savefig(p3, dpi=150); plt.close(fig); print(f'  Example -> {p3}')
        else:
            target = f'lr={args.lr_example}' if args.lr_example is not None else 'no lr'
            print(f'  (no example figure: there is no bs2/bs8 pair at the same lr [{target}])')
    except Exception as e:
        print(f'  (no figures: {type(e).__name__}: {e})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Quantifies the bias of the val minimum against the noise (selection robustness).')
    p.add_argument('--glob', default='runs/lr*_bs*', help='pattern of config folders.')
    p.add_argument('--col', default='val_loss', choices=['val_loss', 'val_dice'],
                   help='curve.csv column to analyze. val_loss: best=minimum (original '
                        'behavior). val_dice: best=maximum (max_val/robmax; to compare configs '
                        'with different losses, e.g. stage B).')
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--tail', type=int, default=10, help='final epochs for the stabilized val.')
    p.add_argument('--slope_n', type=int, default=20, help='final epochs for the slope.')
    p.add_argument('--mvwin', type=int, default=5, help='moving-average window for robmin.')
    p.add_argument('--out_dir', default=None,
                   help='where to write robustness.csv/.png (default: the common parent of the analyzed folders).')
    p.add_argument('--lr_example', default=None,
                   help='lr at which to compare bs2 vs bs8 in the example figure (e.g. 1e-4). '
                        'Default: auto = the lr with a bs2/bs8 pair and the best bs8. Isolates the '
                        'batch size by fixing the lr (before, the figure could mix different lr).')
    p.add_argument('--zoom_from', type=int, default=None,
                   help='if given, adds a box (inset) with the zoom of epochs >= ZOOM_FROM to the '
                        'example figure. Useful when at a small loss scale the noise is squashed and '
                        'cannot be told apart in the full curve (e.g. --zoom_from 90).')
    args = p.parse_args()
    args.higher_better = (args.col == 'val_dice')
    main(args)

