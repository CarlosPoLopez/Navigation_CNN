#!/bin/sh
#SBATCH -J eval_val
#SBATCH -o eval_%j.o
#SBATCH -e eval_%j.e
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 0:40:00
#SBATCH -p viz
#SBATCH --gres=gpu:t4:1

# Evaluates one or several configs on VALIDATION (Dice/clDice/%solved) to choose between
# losses/configs. Does NOT touch the test. Writes val_metrics.csv (+ val_overlays/) INSIDE
# each config folder, and prints a table sorted by % solved.
# Submit from the repository root:  sbatch slurm/sbatch_eval.sh "runs/probe/*" [N_OV]
#   $1 = glob of config folders (IN QUOTES, else the shell expands it and breaks the args)
#   $2 = N overlays per config (default 0 = none)

# ---- Adapt to your environment (or export these before submitting) ------------
PYTHON="${PYTHON:-python}"
CODE="${CODE:-${SLURM_SUBMIT_DIR:-.}}"              # folder with this repo (eval_val.py in analysis/)
DATA="${DATA:-$CODE/data/Dataset_Matrices}"
TESTIDS="${TESTIDS:-$CODE/data/test_ids.txt}"
command -v module >/dev/null 2>&1 && { module purge; module load cesga/system miniconda3; }
# -------------------------------------------------------------------------------

GLOB="${1:-$CODE/runs/probe/*}"
N_OV="${2:-0}"

"${PYTHON}" "${CODE}/analysis/eval_val.py" \
    --data_dir "${DATA}" --test_ids "${TESTIDS}" \
    --glob "${GLOB}" --n_overlays "${N_OV}"
