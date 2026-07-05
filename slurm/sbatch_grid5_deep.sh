#!/bin/sh
#SBATCH -J grid5_deep
#SBATCH -o grid5deep_%j.o
#SBATCH -e grid5deep_%j.e
#SBATCH -p medium
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:5
#SBATCH -c 60
#SBATCH --mem=480G
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your@email          # uncomment and set your address for job emails

# FULL GRID (3 lr x 4 bs {16,8,4,2} x 5 folds = 60), driven by analysis/dispatch_grid.py:
# it keeps one train.py per GPU running until the queue is
# empty. Resumable (skips (config,fold) that already have best.pt). Requires a multi-GPU
# node. Submit from the repository root:  sbatch slurm/sbatch_grid5_deep.sh
# Results -> $RUNS_ROOT/lr<LR>_bs<BS>/fold<F>/best.pt ; per-job logs in $RUNS_ROOT/_logs.

# ---- Adapt to your environment (or export these before submitting) ------------
CODE="${CODE:-${SLURM_SUBMIT_DIR:-.}}"              # folder with this repo
export PYTHON="${PYTHON:-python}"                   # python that launches train.py
export NET_DIR="${NET_DIR:-$CODE}"                  # folder where train.py lives
export DATA_DIR="${DATA_DIR:-$CODE/data/Dataset_Matrices}"
export TEST_IDS="${TEST_IDS:-$CODE/data/test_ids.txt}"
export RUNS_ROOT="${RUNS_ROOT:-$CODE/runs/cross_validation_final}"
export NGPU="${NGPU:-5}"                            # number of GPUs on the node
command -v module >/dev/null 2>&1 && { module purge; module load cesga/system miniconda3; }
# -------------------------------------------------------------------------------

mkdir -p "$RUNS_ROOT"

"${PYTHON}" "${CODE}/analysis/dispatch_grid.py"
