#!/bin/sh
#SBATCH -J net_test
#SBATCH -o test_%j.o
#SBATCH -e test_%j.e
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH -p viz
#SBATCH --gres=gpu:t4:1

# FROZEN TEST: evaluates a k-fold (or single model) on the held-out test.
# net_test.py prints per-fold metrics (mean+-sigma) AND the ENSEMBLE ones, and writes
# test_metrics.csv and test_overlays/ INSIDE <CKPT_DIR>. Does NOT touch training.
# Submit from the repository root:  sbatch slurm/sbatch_test.sh [CKPT_DIR] [K] [N_OVERLAYS]
#   CKPT_DIR   -> folder with fold*/best.pt (required; e.g. runs/my_model)
#   K          -> 5 = ensemble of the 5 folds ; 1 = single model (fold0)   (default 5)
#   N_OVERLAYS -> number of overlays to save (default 12)

# ---- Adapt to your environment (or export these before submitting) ------------
PYTHON="${PYTHON:-python}"
CODE="${CODE:-${SLURM_SUBMIT_DIR:-.}}"              # folder with this repo (net_test.py at its root)
DATA="${DATA:-$CODE/data/Dataset_Matrices}"
TESTIDS="${TESTIDS:-$CODE/data/test_ids.txt}"
command -v module >/dev/null 2>&1 && { module purge; module load cesga/system miniconda3; }
# -------------------------------------------------------------------------------

CKPT="${1:-$CODE/runs/baseline}"
K="${2:-5}"
N_OV="${3:-12}"

echo "TEST: ckpt=${CKPT} | k=${K} | overlays=${N_OV}"

"${PYTHON}" "${CODE}/net_test.py" \
    --data_dir "${DATA}" --test_ids "${TESTIDS}" \
    --ckpt_dir "${CKPT}" --k "${K}" --n_overlays "${N_OV}"
