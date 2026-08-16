#!/bin/bash
#SBATCH --job-name=bleedsense
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --time=01:00:00
#SBATCH --output=/work/cvcs2026/bleedsense/logs/%x_%j.out
#SBATCH --error=/work/cvcs2026/bleedsense/logs/%x_%j.err
#SBATCH --account=cvcs2026
# NB: QOS non specificata volutamente: per utenti studente viene assegnata
# automaticamente all_qos_sprod da parte del sistema di submission.
#
# Uso: sbatch --job-name=<nome> train.sh <dataset> <architecture> <fold> <augmentation>
# Esempio: sbatch --job-name=hemoset_unet_f0 train.sh hemoset unet 0 light
#          sbatch --job-name=hemoset_unet_f0_aug train.sh hemoset unet 0 aggressive
#
# Per un test rapido (debug, priorita' alta, max 1h/1GPU) sovrascrivi
# partition/qos/time da riga di comando, es.:
#   sbatch --qos=all_qos_dbg --time=00:10:00 --job-name=debug train.sh hemoset unet 0 light

set -e

DATASET=$1
ARCHITECTURE=$2
FOLD=${3:-0}
AUGMENTATION=${4:-light}

mkdir -p /work/cvcs2026/bleedsense/logs

module unload cuda >/dev/null 2>&1 || true
module load cuda/12.6.3
source /homes/rmazzi/cvcs2026/venv/bin/activate

cd /work/cvcs2026/bleedsense

python train.py \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --fold "$FOLD" \
    --augmentation "$AUGMENTATION"
