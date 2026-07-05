"""Evaluates the k-fold baseline on the frozen TEST (never seen during training).

Loads runs/baseline/fold{0..k-1}/best.pt and, over the test_ids.txt samples:
  - Computes three metrics per sample and fold -> mean +/- std across folds:
      * Dice   : general pixel overlap (global quality).
      * clDice : centerline preservation (sensitive to breaks/connectivity).
      * solved : does the prediction connect entry and exit? (4-connected BFS). Rate %.
  - Computes the ENSEMBLE (mean of the sigmoids of the k folds) and its metrics.
  - Writes test_metrics.csv (one row/sample: dice/cldice/solved of the ensemble).
  - Saves N overlays: maze + real mask + prediction with an error map
    (green=hit TP, blue=real path not detected FN, red=false positive FP).

Usage (after training the k folds; it is light, the login node is enough):
  python net_test.py --ckpt_dir ../runs/baseline --k 5 --n_overlays 12
"""

import os
import csv
import heapq
import argparse
from collections import deque
import numpy as np
import torch

try:
    from skimage.morphology import skeletonize, closing, disk
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def apply_closing(pred_np, r):
    """Post-processing: morphological closing (dilate+erode, disk r) on the binary
    prediction, BEFORE measuring. Reconnects 1-2px gaps of the path without removing
    pixels (extensive operation: it can only reconnect, never break). r<=0 disables it."""
    if r <= 0:
        return pred_np
    return closing(pred_np > 0.5, disk(r)).astype(np.float32)

from dataset import MazeDataset, list_samples, load_test_ids
from unet import UNet

# Fixed model/inference settings (part of the model, not tunable from the CLI):
THRESHOLD = 0.5   # binarization threshold applied to the sigmoid output
CLOSING = 1       # radius of the morphological closing post-processing


# -- Metrics ----------------------------------------------------------------------

def dice_coef(pred, target, eps=1e-6):
    """pred and target are binary masks (0/1) of the same size (tensors)."""
    inter = (pred * target).sum().item()
    p, t = pred.sum().item(), target.sum().item()
    # If both masks are empty the overlap is perfect by convention.
    if p + t == 0:
        return 1.0
    return (2 * inter + eps) / (p + t + eps)


# -- Connectivity / topology ------------------------------------------------------

def cldice(pred, gt, eps=1e-6):
    """clDice (centerline Dice): harmonic mean of
        Tprec = |skeleton(pred) ∩ gt|  / |skeleton(pred)|   (topological precision)
        Tsens = |skeleton(gt)   ∩ pred| / |skeleton(gt)|    (topological sensitivity)
    Measures whether the CENTERLINE of the path is preserved -> it drops as soon as the
    path breaks, even if the pixel Dice stays high. pred and gt: 2D binary numpy arrays."""
    if not _HAS_SKIMAGE:
        return float('nan')
    p, g = (pred > 0.5), (gt > 0.5)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    skp, skg = skeletonize(p), skeletonize(g)
    tprec = (skp & g).sum() / (skp.sum() + eps)
    tsens = (skg & p).sum() / (skg.sum() + eps)
    if tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens + eps))


def maze_solved(pred, gt, margin=8):
    """Does the prediction connect ENTRY and EXIT with a traversable path?

    Entry/exit = endpoints of the REAL path (opposite corners of the maze: the gt pixel
    nearest to (0,0) and the one nearest to (H,W)). A 4-connected BFS is run over the
    PREDICTED path (in a maze you move in a cross, not diagonally: a diagonal-only
    contact is not traversable). 'margin' tolerates a few pixels of endpoint misalignment.
    Returns 1 if the exit is reachable from the entry, 0 otherwise. pred and gt: 2D
    binary numpy arrays."""
    p, g = (pred > 0.5), (gt > 0.5)
    ys, xs = np.where(g)
    if len(ys) == 0:                                  # no real path
        return int(p.sum() == 0)
    s = ys.astype(int) + xs.astype(int)
    iy, ix = int(ys[s.argmin()]), int(xs[s.argmin()])  # entry (top-left corner)
    fy, fx = int(ys[s.argmax()]), int(xs[s.argmax()])  # exit  (bottom-right corner)
    H, W = p.shape

    def box(cy, cx):
        return (max(0, cy - margin), min(H, cy + margin + 1),
                max(0, cx - margin), min(W, cx + margin + 1))

    visit = np.zeros_like(p, dtype=bool)
    dq = deque()
    y0, y1, x0, x1 = box(iy, ix)                       # seeds next to the entry
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            if p[yy, xx] and not visit[yy, xx]:
                visit[yy, xx] = True; dq.append((yy, xx))
    if not dq:
        return 0                                       # not even a path at the entry
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and p[ny, nx] and not visit[ny, nx]:
                visit[ny, nx] = True; dq.append((ny, nx))
    fy0, fy1, fx0, fx1 = box(fy, fx)
    return int(visit[fy0:fy1, fx0:fx1].any())


# -- Overlay for the report -------------------------------------------------------

def overlay_error(maze, gt, pred):
    """Returns an RGB image (H,W,3) in [0,1]:
    light-gray wall in the background, and over the path:
      green = TP (hit), blue = FN (real path not detected), red = FP."""
    h, w = gt.shape
    img = np.ones((h, w, 3), dtype=np.float32)        # white background
    img[maze > 0.5] = (0.80, 0.80, 0.80)              # light-gray walls
    tp = (pred > 0.5) & (gt > 0.5)
    fn = (pred <= 0.5) & (gt > 0.5)
    fp = (pred > 0.5) & (gt <= 0.5)
    img[fn] = (0.20, 0.40, 1.00)                       # blue: what was missed
    img[fp] = (1.00, 0.25, 0.25)                       # red: what is extra
    img[tp] = (0.10, 0.70, 0.20)                       # green: hit
    return img


def save_overlays(samples, out_dir):
    """samples: list of dicts with maze/gt/pred (np 2D) + name + dice/cldice."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'  (no overlays: {type(e).__name__}: {e})')
        return

    from matplotlib.patches import Patch
    # Legend of the error map (same colors as overlay_error).
    legend = [
        Patch(facecolor=(0.10, 0.70, 0.20), label='TP: hit (path correctly detected)'),
        Patch(facecolor=(0.20, 0.40, 1.00), label='FN: real path not detected'),
        Patch(facecolor=(1.00, 0.25, 0.25), label='FP: false positive (invented path)'),
        Patch(facecolor=(0.80, 0.80, 0.80), label='Wall'),
    ]

    os.makedirs(out_dir, exist_ok=True)
    for m in samples:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
        # origin='lower' SAME as the generation: entry bottom-left, exit top-right, as
        # in the original figures (consistency with the report).
        axs[0].imshow(m['maze'], cmap='gray_r', origin='lower'); axs[0].set_title('Maze')
        axs[1].imshow(m['gt'],   cmap='gray_r', origin='lower'); axs[1].set_title('Real solution')
        axs[2].imshow(overlay_error(m['maze'], m['gt'], m['pred']), origin='lower')
        status = 'SOLVED' if m.get('solved') else 'NOT solved'
        axs[2].set_title(f"Prediction  Dice={m['dice']:.3f}  "
                         f"clDice={m.get('cldice', float('nan')):.3f}  [{status}]")
        for ax in axs:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(m['name'], fontsize=10)
        # Shared legend below the three panels.
        fig.legend(handles=legend, loc='lower center', ncol=4,
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        out = os.path.join(out_dir, f"overlay_{m['name'].replace('.npz','')}.png")
        fig.savefig(out, dpi=130); plt.close(fig)
    print(f'  Overlays    -> {out_dir}  ({len(samples)} images)')


# -- Main -------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1) Frozen test: the SAME samples held out from the k-fold.
    files = list_samples(args.data_dir)
    test_ids = load_test_ids(args.test_ids)
    if not test_ids:
        raise SystemExit('test_ids.txt empty or missing: there is no test to evaluate.')
    test_files = [f for f in files if os.path.basename(f) in test_ids]
    print(f'Test: {len(test_files)} samples (out of {len(files)} total).')

    test_ds = MazeDataset(test_files, augment=False)

    # 2) Load the available checkpoints of each fold.
    models = {}
    for k in range(args.k):
        ckpt = os.path.join(args.ckpt_dir, f'fold{k}', 'best.pt')
        if not os.path.exists(ckpt):
            print(f'  WARNING: {ckpt} missing (fold {k} not trained).')
            continue
        state = torch.load(ckpt, map_location=device)
        feats = (32, 64, 128, 256, 512)
        model = UNet(in_channels=3, features=feats).to(device)
        model.load_state_dict(state['model'])
        model.eval()
        models[k] = model
        print(f'  fold{k}: best.pt epoch {state.get("epoch","?")} '
              f'val_loss {state.get("val_loss", float("nan")):.4f}')
    if not models:
        raise SystemExit(f'No best.pt was loaded in {args.ckpt_dir}')
    order = sorted(models)

    # 3) Inference. Accumulates Dice/clDice/solved per fold and the mean prediction.
    dice_fold = {k: [] for k in order}
    cl_fold   = {k: [] for k in order}
    sv_fold   = {k: [] for k in order}
    dice_ens, cl_ens, sv_ens, rows = [], [], [], []
    overlays = []            # heap bounded to n_overlays: (key, counter, entry)
    _ov_cnt = 0              # breaks ties in the heap without comparing the dicts

    with torch.no_grad():  # no gradients, faster, we are not training
        for i in range(len(test_ds)):
            x, y = test_ds[i]
            name = os.path.basename(test_files[i])
            x = x.unsqueeze(0).to(device)               # (1,1,H,W)
            gt = (y[0] > 0.5).float()                   # (H,W) on CPU
            gt_np = gt.numpy()

            prob_sum = torch.zeros_like(y[0])           # accumulates sigmoids (CPU)
            for k in order:
                prob = torch.sigmoid(models[k](x))[0, 0].cpu()
                prob_sum += prob
                pk_np = apply_closing((prob > THRESHOLD).numpy().astype(np.float32), CLOSING)
                pred_k = torch.from_numpy(pk_np)
                d = dice_coef(pred_k, gt)
                dice_fold[k].append(d)
                cl_fold[k].append(cldice(pk_np, gt_np))
                sv_fold[k].append(maze_solved(pk_np, gt_np, args.margin))

            prob_ens = prob_sum / len(order)
            pe_np = apply_closing((prob_ens > THRESHOLD).numpy().astype(np.float32), CLOSING)
            pred_ens = torch.from_numpy(pe_np)
            d = dice_coef(pred_ens, gt)
            cl = cldice(pe_np, gt_np)
            sv = maze_solved(pe_np, gt_np, args.margin)
            dice_ens.append(d); cl_ens.append(cl); sv_ens.append(sv)
            rows.append((name, d, cl, sv))

            if args.n_overlays > 0:
                entry = {'name': name,
                         'maze': x[0, 0].cpu().numpy(),
                         'gt': gt_np,
                         'pred': pe_np,
                         'dice': d, 'cldice': cl, 'solved': sv}
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

    # 4) Console summary.
    dice_ens = np.array(dice_ens)
    cl_ens = np.array(cl_ens); sv_ens = np.array(sv_ens, dtype=float)
    print(f'\n  Test - {args.ckpt_dir}  ({len(test_files)} samples)\n')
    print(f'  {"fold":>6} {"Dice":>14} {"clDice":>14} {"solved":>9}')
    print(f'  {"-"*6} {"-"*14} {"-"*14} {"-"*9}')
    md, mc, ms = [], [], []
    for k in order:
        df = np.array(dice_fold[k])
        cf, sf = np.array(cl_fold[k]), np.array(sv_fold[k], dtype=float)
        md.append(df.mean())
        mc.append(np.nanmean(cf)); ms.append(sf.mean())
        print(f'  {k:>6} {df.mean():>6.3f}±{df.std():<5.3f} '
              f'{np.nanmean(cf):>6.3f}±{np.nanstd(cf):<5.3f} '
              f'{100*sf.mean():>7.1f}%')
    md, mc, ms = map(np.array, (md, mc, ms))
    print(f'  {"-"*6} {"-"*14} {"-"*14} {"-"*9}')
    print(f'  {"means":>6} {md.mean():>6.3f}±{md.std():<5.3f} '
          f'{mc.mean():>6.3f}±{mc.std():<5.3f} '
          f'{100*ms.mean():>7.1f}%   (across folds)')
    print(f'\n  ENSEMBLE (mean of the sigmoids of the {len(order)} folds):')
    print(f'     Dice   = {dice_ens.mean():.4f} ± {dice_ens.std():.4f}')
    print(f'     clDice = {np.nanmean(cl_ens):.4f} ± {np.nanstd(cl_ens):.4f}')
    print(f'     Solved = {int(sv_ens.sum())}/{len(sv_ens)} ({100*sv_ens.mean():.1f}%)')

    # 5) Per-sample CSV (ensemble).
    out_csv = os.path.join(args.ckpt_dir, 'test_metrics.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample', 'dice_ensemble',
                    'cldice_ensemble', 'solved_ensemble'])
        for name, d, cl, sv in rows:
            w.writerow([name, f'{d:.6f}', f'{cl:.6f}', int(sv)])
    print(f'\n  Metrics     -> {out_csv}')

    # 6) Overlays. Unwinds the heap into a list of dicts; best/worst sorted (best first).
    if args.overlay_sel == 'first':
        samples_ov = [e for _, _, e in overlays]
    else:
        # the key is built so that the largest is the "most desired" (best in best, worst in worst).
        samples_ov = [e for _, _, e in sorted(overlays, key=lambda t: t[0], reverse=True)]
    save_overlays(samples_ov, os.path.join(args.ckpt_dir, 'test_overlays'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluates the k-fold on the frozen test.')
    p.add_argument('--data_dir',   default='data/Dataset_Matrices',
                   help='folder with the sample_*.npz dataset (from Part 1).')
    p.add_argument('--test_ids',   default='data/test_ids.txt',
                   help='file listing the frozen test samples.')
    p.add_argument('--ckpt_dir',   default='runs/baseline',
                   help='folder with fold*/best.pt; results go here (test_metrics.csv, test_overlays/).')
    p.add_argument('--k',          type=int,   default=5,
                   help='number of folds to load: 5 = ensemble, 1 = single model (fold0).')
    p.add_argument('--n_overlays', type=int,   default=12)
    p.add_argument('--overlay_sel', choices=['first', 'best', 'worst'], default='best',
                   help="which samples to save as overlay: 'best' (N best, default), "
                        "'worst' (N worst) or 'first' (N first). Order: solved desc/asc, then Dice.")
    p.add_argument('--margin',     type=int,   default=8,
                   help='tolerance (px) at entry/exit for the solving BFS.')
    main(p.parse_args())
