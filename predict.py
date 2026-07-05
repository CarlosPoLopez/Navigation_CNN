"""predict.py - solves ONE maze with the network (single model or ensemble).

Meant to try ARBITRARY mazes (e.g. generated from maps in another project), with or
without a reference solution:
  - WITH solution (.npz with 'solucion'): TP/FN/FP overlay + Dice/clDice/solved.
  - WITHOUT solution: only predicts; draws the path over the maze and checks with a
    4-connected BFS whether it connects the ENTRY room (top-left corner) with the EXIT
    one (bottom-right corner), using the same 50x50 goal-conditioning markers.

The preprocessing is IDENTICAL to dataset.MazeDataset (wall = maze<0, 3 channels with
the entry/exit squares, nearest resize to 256) so the model sees the same as in training.
By default only the overlay PNG is saved; with --save_npy the mask, rescaled back to the
ORIGINAL resolution of the maze, is also saved as .npy (reusable in your pipeline).

Input (--input):
  *.npz  -> dataset format: key 'laberinto' (field F, <0=wall) and, optionally,
            'solucion' (field u, >=THRESHOLD=path).
  *.npy  -> plain 2D maze matrix (no solution). Wall rule via --wall:
            auto (default: if there are negatives -> '<0', else '>0.5'), neg (<0),
            pos (>0), nonzero.

A light morphological closing (radius 1) is applied to the binary prediction, reconnecting
1-2px gaps of the path without altering its thickness or overlap (raises the solved rate
at ~zero cost). The input resolution (256), the binarization threshold (0.5) and this
closing are fixed (part of the model); edit the constants at the top to change them.

Checkpoints: --ckpt_dir with fold{0..k-1}/best.pt. --k 1 = single model (fold0);
--k 5 = ensemble (mean of sigmoids). Without --k, auto-detects how many folds there are.

Usage:
  python predict.py --input my_maze.npz --ckpt_dir runs/final_model --k 1
  python predict.py --input map42.npy  --ckpt_dir runs/kfold/lr1e-4_bs8 --k 5
"""

import os
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from unet import UNet
# Reuse the metrics and error overlay from the test (same definition, no divergence).
from net_test import dice_coef, cldice, maze_solved, overlay_error

THRESHOLD = 0.727      # reads the .npz 'solucion' field into a binary path mask (data)
IMG_SIZE = 256         # fixed working resolution (same as training)
PRED_THRESHOLD = 0.5   # fixed binarization threshold for the network output
CLOSING = 1            # fixed morphological closing radius (post-processing)


def _square(h, w, r0, r1, c0, c1):
    m = np.zeros((h, w), dtype=np.float32)
    m[r0:r1, c0:c1] = 1.0
    return m


def load_maze(path, wall):
    """Returns (maze_bin HxW {0,1}, solution_bin or None) at ORIGINAL resolution.

    .npz: uses 'laberinto' (<0=wall) and, if present, 'solucion' (>=THRESHOLD=path).
    .npy / 2D array: plain maze matrix; wall according to `wall`."""
    if path.endswith('.npz'):
        data = np.load(path)
        if 'laberinto' not in data:
            raise SystemExit(f"{path}: .npz without 'laberinto' key (keys: {list(data.keys())})")
        maze = (data['laberinto'] < 0).astype(np.float32)
        sol = (data['solucion'] >= THRESHOLD).astype(np.float32) if 'solucion' in data else None
        return maze, sol
    arr = np.asarray(np.load(path)).astype(np.float32)
    if arr.ndim != 2:
        raise SystemExit(f'{path}: a 2D matrix was expected, shape={arr.shape}')
    if wall == 'neg':
        maze = arr < 0
    elif wall == 'pos':
        maze = arr > 0
    elif wall == 'nonzero':
        maze = arr != 0
    else:  # auto: if there are negatives it is field F (<0=wall); else binary (>0.5=wall)
        maze = (arr < 0) if (arr < 0).any() else (arr > 0.5)
    return maze.astype(np.float32), None


def build_input(maze, mb, ms):
    """Same assembly as MazeDataset: 3 channels (maze + entry + exit) and nearest
    resize to IMG_SIZE. The squares are built at the ORIGINAL resolution."""
    h, w = maze.shape
    start = _square(h, w, mb, mb + ms, mb, mb + ms)
    end   = _square(h, w, h - mb - ms, h - mb, w - mb - ms, w - mb)
    x = torch.tensor(np.stack([maze, start, end], 0))          # (3,H,W)
    x = F.interpolate(x.unsqueeze(0), size=IMG_SIZE, mode='nearest').squeeze(0)
    return x


def load_models(ckpt_dir, k, device):
    """Loads fold0..k-1/best.pt. Returns a list of models in eval()."""
    models = []
    feats = (32, 64, 128, 256, 512)
    for i in range(k):
        ckpt = os.path.join(ckpt_dir, f'fold{i}', 'best.pt')
        if not os.path.exists(ckpt):
            print(f'  WARNING: {ckpt} missing (fold {i} not trained).')
            continue
        state = torch.load(ckpt, map_location=device)
        m = UNet(in_channels=3, features=feats).to(device)
        m.load_state_dict(state['model'])
        m.eval()
        models.append(m)
        print(f'  fold{i}: best.pt epoch {state.get("epoch", "?")}')
    if not models:
        raise SystemExit(f'No best.pt was loaded in {ckpt_dir}')
    return models


def detect_k(ckpt_dir):
    """Counts consecutive folds with best.pt (fold0, fold1, ...)."""
    k = 0
    while os.path.exists(os.path.join(ckpt_dir, f'fold{k}', 'best.pt')):
        k += 1
    return k


def connects_regions(pred, entry, exit_):
    """4-connected BFS over the predicted mask: is there a traversable path from the
    ENTRY region to the EXIT region? (in a maze you move in a cross, not diagonally).
    entry/exit_: boolean masks of the same shape as pred."""
    p = pred > 0.5
    H, W = p.shape
    visit = np.zeros_like(p, dtype=bool)
    dq = deque()
    for y, x in np.argwhere(p & entry):
        visit[y, x] = True; dq.append((int(y), int(x)))
    if not dq:
        return 0
    while dq:
        y, x = dq.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and p[ny, nx] and not visit[ny, nx]:
                visit[ny, nx] = True; dq.append((ny, nx))
    return int((visit & exit_).any())


def overlay_pred(maze, pred):
    """RGB (H,W,3): gray walls, PREDICTED path in red (case without solution)."""
    h, w = maze.shape
    img = np.ones((h, w, 3), dtype=np.float32)
    img[maze > 0.5] = (0.80, 0.80, 0.80)
    img[pred > 0.5] = (0.90, 0.12, 0.12)
    return img


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1) Maze (original resolution) and solution if it comes with one.
    maze, sol = load_maze(args.input, args.wall)
    h, w = maze.shape
    print(f'Maze: {os.path.basename(args.input)}  {h}x{w}  '
          f'{"with" if sol is not None else "WITHOUT"} reference solution.')

    # 2) Models.
    k = args.k if args.k is not None else detect_k(args.ckpt_dir)
    if k <= 0:
        raise SystemExit(f'No fold*/best.pt in {args.ckpt_dir}')
    models = load_models(args.ckpt_dir, k, device)

    # 3) Inference at IMG_SIZE. It is MEASURED and drawn at IMG_SIZE (as in training and
    #    net_test): the path is 1-2 px wide and rescaling it back would fragment it,
    #    breaking connectivity and lowering Dice. The mask that is SAVED is rescaled to
    #    the original resolution (nearest, no fragmentation) for your pipeline.
    xb = build_input(maze, args.marker_border, args.marker_size).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_sum = None
        for m in models:
            prob = torch.sigmoid(m(xb))[0, 0].cpu()
            prob_sum = prob if prob_sum is None else prob_sum + prob
    prob = prob_sum / len(models)
    pred = (prob > PRED_THRESHOLD).numpy().astype(np.float32)        # at IMG_SIZE
    # Post-processing: morphological closing (dilate+erode, disk r=1). Bridges 1-2px gaps
    # and fills small holes WITHOUT thickening isolated noise (it is local). Raises the
    # solved rate without touching the clDice on a good model.
    from skimage.morphology import closing, disk
    pred = closing(pred > 0.5, disk(CLOSING)).astype(np.float32)
    print(f'  Post-processing: morphological closing disk r={CLOSING}.')
    maze512 = xb[0, 0].cpu().numpy()                                 # rescaled maze
    kind = 'ensemble' if len(models) > 1 else 'single model'
    mode = 'ensemble' if len(models) > 1 else 'single'
    print(f'Prediction: {kind} of {len(models)} network(s), threshold {PRED_THRESHOLD}.')

    # 4) Entry->exit connectivity (always; does not need a solution). The entry/exit
    #    regions come from the goal channels.
    entry = xb[0, 1].cpu().numpy() > 0.5
    exit_ = xb[0, 2].cpu().numpy() > 0.5
    connects = connects_regions(pred, entry, exit_)
    print(f'  Connects entry-exit (BFS): {"YES" if connects else "NO"}')

    # 5) Metrics vs the reference solution, if any (at IMG_SIZE, like net_test).
    if sol is not None:
        sol512 = F.interpolate(torch.tensor(sol)[None, None], size=IMG_SIZE,
                               mode='nearest')[0, 0]
        sol512_np = sol512.numpy()
        d = dice_coef(torch.from_numpy(pred), sol512)
        cl = cldice(pred, sol512_np)
        solved = maze_solved(pred, sol512_np, args.margin)
        print(f'  Dice={d:.3f}  clDice={cl:.3f}  solved(vs gt)={"YES" if solved else "NO"}')
    else:
        sol512_np = None
        d = cl = float('nan'); solved = connects

    # 6) Outputs: overlay PNG (always). The .npy mask (rescaled to the ORIGINAL
    #    resolution, nearest) is only saved with --save_npy, in case you want to reuse it
    #    in another pipeline; by default it is not written.
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_stem = f'pred_{base}_{mode}'
    if args.save_npy:
        pred_orig = F.interpolate(torch.tensor(pred)[None, None], size=(h, w),
                                  mode='nearest')[0, 0].numpy().astype(np.uint8)
        np.save(os.path.join(out_dir, f'{out_stem}.npy'), pred_orig)
        print(f'  Mask        -> {os.path.join(out_dir, f"{out_stem}.npy")}  ({h}x{w})')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        # origin='lower' SAME as the generation: the entry [5:55,5:55] ends up
        # bottom-left and the exit top-right, as in the original maze figures
        # (consistency with the report).
        if sol512_np is not None:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4.8))
            axs[0].imshow(maze512,   cmap='gray_r', origin='lower'); axs[0].set_title('Maze')
            axs[1].imshow(sol512_np, cmap='gray_r', origin='lower'); axs[1].set_title('Real solution')
            axs[2].imshow(overlay_error(maze512, sol512_np, pred), origin='lower')
            axs[2].set_title(f'Prediction  Dice={d:.3f}  clDice={cl:.3f}  '
                             f'[{"SOLVED" if solved else "NOT solved"}]')
            legend = [Patch(facecolor=(0.10, 0.70, 0.20), label='TP hit'),
                      Patch(facecolor=(0.20, 0.40, 1.00), label='FN not detected'),
                      Patch(facecolor=(1.00, 0.25, 0.25), label='FP invented'),
                      Patch(facecolor=(0.80, 0.80, 0.80), label='Wall')]
            axes = axs
        else:
            fig, axs = plt.subplots(1, 2, figsize=(8.6, 4.8))
            axs[0].imshow(maze512, cmap='gray_r', origin='lower'); axs[0].set_title('Maze')
            axs[1].imshow(overlay_pred(maze512, pred), origin='lower')
            axs[1].set_title(f'Prediction  [{"connects" if connects else "does NOT connect"}]')
            legend = [Patch(facecolor=(0.90, 0.12, 0.12), label='Predicted path'),
                      Patch(facecolor=(0.80, 0.80, 0.80), label='Wall')]
            axes = axs
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(base, fontsize=10)
        fig.legend(handles=legend, loc='lower center',
                   ncol=len(legend), fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        out_png = os.path.join(out_dir, f'{out_stem}.png')
        fig.savefig(out_png, dpi=130); plt.close(fig)
        print(f'  Overlay     -> {out_png}')
    except Exception as e:
        print(f'  (no overlay: {type(e).__name__}: {e})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Solves a single maze with the network (single or ensemble).')
    p.add_argument('--input', required=True, help='maze .npz (dataset) or .npy (2D matrix).')
    p.add_argument('--ckpt_dir', required=True, help='folder with fold*/best.pt.')
    p.add_argument('--k', type=int, default=None,
                   help='number of folds to use: 1=single model, 5=ensemble. Default: auto-detect.')
    p.add_argument('--marker_size', type=int, default=50,
                   help='side of the entry/exit square (same as in training).')
    p.add_argument('--marker_border', type=int, default=5,
                   help='margin of the square to the border (same as in training).')
    p.add_argument('--margin', type=int, default=8, help='endpoint tolerance in the connectivity BFS.')
    p.add_argument('--wall', choices=['auto', 'neg', 'pos', 'nonzero'], default='auto',
                   help='wall rule for a raw .npy (does not apply to .npz, which uses <0).')
    p.add_argument('--out_dir', default=None, help='where to write pred_*.png (default: next to the input).')
    p.add_argument('--save_npy', action='store_true',
                   help='also saves the .npy mask (original resolution) besides the PNG.')
    main(p.parse_args())
