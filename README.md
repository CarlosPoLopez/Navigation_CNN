# Navigation_CNN — Maze solving with a U-Net

A convolutional neural network (U-Net) that learns to solve mazes, reproducing in
milliseconds the paths that the FitzHugh-Nagumo reaction-diffusion algorithm computes by
physical simulation.

**Part 2** of the BSc thesis *"Navigation algorithms based on biological waves and their
optimization with AI tools"* (BSc in Physics, University of Santiago de Compostela,
2025–2026).

> **Part 1 — physical simulation:** the algorithm that generates the training data lives
> in a separate repository:
> **[Navigation_algorithm](https://github.com/CarlosPoLopez/Navigation_algorithm)**
> (FitzHugh-Nagumo bistable reaction-diffusion system).

---

## Idea

Part 1 solves each maze by propagating a reaction-diffusion wave: it is robust but
**slow** (~1 hour of computation per maze). This repository trains a U-Net on those
*(maze → path)* pairs to obtain equivalent solutions almost instantly, with a speedup of
several orders of magnitude once trained.

```
Maze (+ entry/exit)  ──►  U-Net  ──►  path mask
```

The input has **3 channels** (maze, entry marker, exit marker — *goal-conditioning*), so
the same network solves different entry–exit pairs on the same maze. The working
resolution is **256×256**.

## Architecture

- **U-Net** encoder–decoder with skip connections (`unet.py`).
- `DoubleConv` blocks (Conv 3×3 → BatchNorm → ReLU, ×2).
- 5-level U-Net (`features=(32,64,128,256,512)`): a deep encoder whose large receptive
  field lets the bottleneck see enough global context to plan the whole path.
- Loss: a fixed sum of three complementary terms (as in the thesis) — **BCE** (pixel) +
  **Tversky** (region, α=0.3, β=0.7) + **clDice** (connectivity). Tversky generalizes Dice
  (α=β=0.5) and penalizes gaps/false negatives more; clDice enforces path continuity.

## Evaluation

Besides the usual segmentation metrics (Dice, IoU), the **connectivity of the path** is
evaluated with a BFS check: a maze counts as *solved* only if a continuous path exists
between entry and exit in the prediction. Model selection is done by **k-fold
cross-validation** over the learning-rate × batch-size grid, comparing the single model
against the fold ensemble.

## Pretrained model (no training needed)

The project has **two trained variants**, both used in the thesis:

- **Single final model** — one U-Net trained on the whole dataset. This is the one
  **published here** (a ~59 MB float16 checkpoint on the
  [Releases](https://github.com/CarlosPoLopez/Navigation_CNN/releases) page), so you can
  solve mazes right away. Use it with `--k 1`.
- **5-fold ensemble** — the five cross-validation models averaged together; it gives
  better path connectivity and is the main result reported in the thesis. It is **not
  distributed** (five checkpoints, ~0.6 GB); reproduce it by training the folds (see
  [Workflow](#workflow)) and evaluating with `--k 5`.

Fully self-contained demo — no dataset needed:

```bash
# 1. download the weights into runs/final_model/fold0/best.pt
python download_model.py

# 2. generate a random maze (1=wall, 0=corridor), saved as maze.npy (+ a preview PNG)
python generate_maze.py --out maze.npy --png

# 3. solve it — writes pred_maze_single.png (maze + predicted path) next to the input
python predict.py --input maze.npy --ckpt_dir runs/final_model --k 1
```

`predict.py` also accepts a dataset `.npz` (from Part 1, with a reference `solucion`); in
that case it additionally reports Dice / clDice / solved against the ground truth.

## Layout

```
unet.py                  Core: U-Net definition
dataset.py               Core: data loading and augmentation
train.py                 Core: training (one fold)
net_test.py              Core: evaluation on the frozen test set (single / ensemble)
predict.py               Core: inference on a single maze (.npz or 2D .npy)
generate_maze.py         Generates a random maze (.npy) to feed predict.py
download_model.py        Fetches the pretrained final model into runs/final_model/
analysis/                Model selection and analysis (not needed to use the network)
  eval_val.py            Validation metrics for choosing between configurations
  dispatch_grid.py       Dispatch of the lr × batch grid across several GPUs
  compare_configs.py     Comparison of configurations (validation only)
  kfold_summary.py       Aggregated summary of k-fold results
  robustness_analysis.py Robustness analysis of the model-selection criterion
slurm/                   SBATCH scripts for the cluster (CESGA)
requirements.txt
```

## Data and expected layout

The dataset is produced by **Part 1** and is not versioned here. Each sample is an
`.npz` file named `muestra_<id>.npz` with two arrays:

| key         | meaning                                                      |
|-------------|-------------------------------------------------------------|
| `laberinto` | maze as a field F (`< 0` = wall, `0` = corridor)            |
| `solucion`  | solution field u (`>= 0.727` = path) — the training target |

All paths default to a repository-relative layout, so the simplest setup is to run every
command **from the repository root** with this structure (both are git-ignored):

```
data/
  Dataset_Matrices/     muestra_*.npz  (the dataset)
  test_ids.txt          list of held-out test samples (created in step 1)
runs/                   checkpoints and results (created by training)
```

Any of these can be overridden with `--data_dir`, `--test_ids`, `--ckpt_dir`, etc. The
dataset is only needed to **train or evaluate**; to just try the pretrained model you do
not need it — generate inputs with `generate_maze.py` (see *Pretrained model* above).

## Workflow

```bash
pip install -r requirements.txt          # torch, numpy, scikit-image, matplotlib
```

**1. Freeze the test set** (once): reserves 10% of the samples so they are never seen in
training.

```bash
python dataset.py --data_dir data/Dataset_Matrices --output data/test_ids.txt
```

**2. Train** the k folds of a configuration (repeat `--fold 0..4`, or use the SLURM array):

```bash
python train.py --ckpt_dir runs/my_model \
    --k 5 --fold 0 --img_size 256 \
    --lr 1e-4 --batch_size 8 --epochs 140 \
    --goal_channels
```
For the final single model, train on the whole pool with `--fold -1` (saved to `fold0/`).

**3. Evaluate on the frozen test set** (ensemble of the k folds, or `--k 1` for a single model):

```bash
python net_test.py --ckpt_dir runs/my_model \
    --k 5 --img_size 256 --threshold 0.5 --closing 1 \
    --goal_channels
```

**4. Solve a single maze** (any `.npz` from the dataset, or a raw 2D `.npy`):

```bash
python predict.py --input maze.npy --ckpt_dir runs/my_model --img_size 256
```

> The `--goal_channels` flag must be the **same** at train, test and predict time (it
> changes the number of input channels). See `--help` on each script for all options.

**Model selection** (optional, `analysis/`): sweep the learning-rate × batch-size grid
(`analysis/dispatch_grid.py`), then rank configs on validation with
`analysis/eval_val.py` / `analysis/compare_configs.py`, summarize a k-fold with
`analysis/kfold_summary.py`, and study the selection robustness with
`analysis/robustness_analysis.py`.

## Where results are saved

| Command                        | Writes                                                                 |
|--------------------------------|------------------------------------------------------------------------|
| `train.py`                     | `<ckpt_dir>/fold<F>/best.pt` and `<ckpt_dir>/fold<F>/curve.csv`         |
| `net_test.py`                  | `<ckpt_dir>/test_metrics.csv` and `<ckpt_dir>/test_overlays/*.png`      |
| `predict.py`                   | `pred_<name>_<mode>.png` next to the input (or `--out_dir`); `.npy` with `--save_npy` |
| `analysis/eval_val.py`         | `<ckpt_dir>/val_metrics.csv` (+ `val_overlays/` if `--n_overlays>0`)    |
| `analysis/compare_configs.py`  | `comparison.csv`, `comparison.png`, `comparison_curves.png` in `--out_dir` |
| `analysis/kfold_summary.py`    | `<runs_dir>/kfold_summary.{csv,png}`                                    |
| `analysis/robustness_analysis.py` | `robustness.csv` and `robustness_*.png` in `--out_dir`              |
| `analysis/dispatch_grid.py`    | `<RUNS_ROOT>/lr<LR>_bs<BS>/fold<F>/best.pt` (+ logs in `<RUNS_ROOT>/_logs`) |

## Running on your PC vs. on an HPC cluster

- **Locally**: everything runs with the commands above (CUDA GPU used automatically if
  available, otherwise CPU). Training the full pipeline on CPU is slow; a single `predict.py`
  is fast on CPU.
- **On a cluster (SLURM)**, the `slurm/` folder has submit scripts (one per stage: train,
  test, eval, full grid, final model). They are the author's **CESGA** scripts kept as a
  template. Submit them **from the repository root**, e.g. `sbatch slurm/sbatch_train.sh`.
  Each script has an *"Adapt to your environment"* block at the top: set `PYTHON` (your
  interpreter/conda env), `CODE` (defaults to the submit dir), `DATA`, `TEST_IDS`, `RUNS`,
  or `export` them beforehand. The `module load cesga/...` lines and the `#SBATCH`
  partitions/GPU types are cluster-specific — replace them with your site's equivalents.

---

**Author:** Carlos Polo López
**Advisor:** Alberto Pérez Muñuzuri · **Co-advisor:** David García Selfa
University of Santiago de Compostela
