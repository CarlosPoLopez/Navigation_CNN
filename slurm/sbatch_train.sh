#!/bin/sh
#SBATCH -J unet_train
#SBATCH -o train_%A_%a.o
#SBATCH -e train_%A_%a.e
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 4:29:00          # ~3:30-4:15 for 100ep at bs=2; 4:29 backfills under the round 4:30
#SBATCH -p short
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-4          # one fold per task (k-fold of 5)

# Trains the k folds of one configuration (one fold per array task) on the cluster.
# Submit FROM THE REPOSITORY ROOT:  sbatch slurm/sbatch_train.sh [CONFIG_NAME] [LR] ...
# Results -> $RUNS/<CONFIG_NAME>/fold<N>/best.pt (+ curve.csv). Job logs -> submit dir.
#
# The model is fixed (3-channel input, 256px, 5-level U-Net, BCE+Tversky+clDice loss);
# only these positional args are configurable (all optional):
#   CONFIG_NAME  -> output folder <RUNS>/<NAME>            (default: baseline)
#   LR           -> learning rate                          (default: 1e-4)
#   EPOCHS       -> number of epochs                       (default: 50)
#   BATCH        -> batch size                             (default: 4)
#   AUGMENT      -> 1 = D4 augmentation ON (default); 0 = OFF (--no_augment, ablation)
# Examples:
#   sbatch slurm/sbatch_train.sh baseline 1e-4
#   sbatch slurm/sbatch_train.sh bs8 1e-4 100 8
#   sbatch --array=0 slurm/sbatch_train.sh probe/lr1e-4_bs8 1e-4 140 8

# ---- Adapt to your environment (or export these before submitting) ------------
PYTHON="${PYTHON:-python}"                          # python interpreter
CODE="${CODE:-${SLURM_SUBMIT_DIR:-.}}"              # folder with this repo (train.py at its root)
DATA="${DATA:-$CODE/data/Dataset_Matrices}"         # dataset (muestra_*.npz)
TESTIDS="${TESTIDS:-$CODE/data/test_ids.txt}"       # frozen test list
RUNS="${RUNS:-$CODE/runs}"                          # where checkpoints are written
# CESGA modules; drop or replace on your cluster:
command -v module >/dev/null 2>&1 && { module purge; module load cesga/system miniconda3; }
# -------------------------------------------------------------------------------

NAME="${1:-baseline}"
LR="${2:-1e-4}"
EPOCHS="${3:-50}"
BATCH="${4:-4}"
AUGMENT="${5:-1}"
AUG_FLAG=""
[ "${AUGMENT}" = "0" ] && AUG_FLAG="--no_augment"

echo "Config: ${NAME} | lr=${LR} | epochs=${EPOCHS} | batch=${BATCH} | augment=${AUGMENT} | fold ${SLURM_ARRAY_TASK_ID} -> ${RUNS}/${NAME}"

"${PYTHON}" "${CODE}/train.py" \
    --data_dir "${DATA}" \
    --test_ids "${TESTIDS}" \
    --ckpt_dir "${RUNS}/${NAME}" \
    --k 5 \
    --fold "${SLURM_ARRAY_TASK_ID}" \
    --batch_size "${BATCH}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    ${AUG_FLAG}
