#!/bin/sh
#SBATCH -J deep5_final
#SBATCH -o deep5final_%j.o
#SBATCH -e deep5final_%j.e
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --gres=gpu:a100:5
#SBATCH -c 60
#SBATCH --mem=480G
#SBATCH -t 6:00:00
#SBATCH -p short

# Trains the k=5 folds of the U-Net with the chosen config
# (lr1e-4 bs8, 256, 140ep, goal + Tversky0.7 + clDice1), 5 folds in PARALLEL on one node
# (one per GPU). Then runs the ENSEMBLE test. Requires a 5-GPU node.
# Submit from the repository root:  sbatch slurm/sbatch_deep5_final.sh
# Results -> $RUNS/deep5_final/lr1e-4_bs8/fold*/best.pt ; test metrics + overlays in that folder.

# ---- Adapt to your environment (or export these before submitting) ------------
PYTHON="${PYTHON:-python}"
CODE="${CODE:-${SLURM_SUBMIT_DIR:-.}}"
DATA="${DATA:-$CODE/data/Dataset_Matrices}"
TESTIDS="${TESTIDS:-$CODE/data/test_ids.txt}"
RUNS="${RUNS:-$CODE/runs}"
command -v module >/dev/null 2>&1 && { module purge; module load cesga/system miniconda3; }
# -------------------------------------------------------------------------------

CKPT="$RUNS/deep5_final/lr1e-4_bs8"
LOGDIR="$RUNS/_logs"
mkdir -p "$LOGDIR"

for f in 0 1 2 3 4; do
  echo "[GPU $f] START deep fold=$f"
  CUDA_VISIBLE_DEVICES=$f "${PYTHON}" "${CODE}/train.py" \
      --data_dir "${DATA}" --test_ids "${TESTIDS}" --ckpt_dir "${CKPT}" \
      --k 5 --fold $f --img_size 256 --batch_size 8 --epochs 140 \
      --lr 1e-4 --cldice_w 1 --tversky_b 0.7 --goal_channels \
      > "$LOGDIR/deep5_final_fold${f}.log" 2>&1 &
done
wait
echo "=== k=5 TRAININGS FINISHED ==="

echo "################## ENSEMBLE TEST (threshold 0.5) ##################"
"${PYTHON}" -u "${CODE}/net_test.py" \
    --data_dir "${DATA}" --test_ids "${TESTIDS}" \
    --ckpt_dir "${CKPT}" --k 5 --goal_channels --n_overlays 12
