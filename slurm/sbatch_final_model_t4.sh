#!/bin/sh
#SBATCH -J final_model
#SBATCH -o finalmodel_%j.o
#SBATCH -e finalmodel_%j.e
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 7:00:00
#SBATCH -p viz
#SBATCH --gres=gpu:t4:1

# FINAL SINGLE MODEL: U-Net retrained from scratch over the WHOLE pool
# (--fold -1, no validation), with the chosen config (lr1e-4, bs8, 256, 140ep,
# goal-conditioning). Saved to $RUNS/final_model/fold0/. Then a single-model
# test (k=1). One GPU is enough.
# Submit from the repository root:  sbatch slurm/sbatch_final_model_t4.sh

# ---- Adapt to your environment (or export these before submitting) ------------
PYTHON="${PYTHON:-python}"
CODE="${CODE:-${SLURM_SUBMIT_DIR:-.}}"
DATA="${DATA:-$CODE/data/Dataset_Matrices}"
TESTIDS="${TESTIDS:-$CODE/data/test_ids.txt}"
RUNS="${RUNS:-$CODE/runs}"
command -v module >/dev/null 2>&1 && { module purge; module load cesga/system miniconda3; }
# -------------------------------------------------------------------------------

CKPT="$RUNS/final_model"

echo "########## FINAL SINGLE MODEL TRAINING (whole pool) ##########"
"${PYTHON}" -u "${CODE}/train.py" \
    --data_dir "${DATA}" --test_ids "${TESTIDS}" --ckpt_dir "${CKPT}" \
    --fold -1 --batch_size 8 --epochs 140 --lr 1e-4

echo
echo "########## SINGLE MODEL TEST (k=1, threshold 0.5) ##########"
"${PYTHON}" -u "${CODE}/net_test.py" \
    --data_dir "${DATA}" --test_ids "${TESTIDS}" \
    --ckpt_dir "${CKPT}" --k 1 --n_overlays 12
