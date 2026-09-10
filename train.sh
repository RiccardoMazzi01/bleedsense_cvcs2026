#!/bin/bash
#SBATCH --job-name=bleedsense
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G|gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G"
#SBATCH --time=01:00:00
#SBATCH --output=/work/cvcs2026/bleedsense/logs/%x_%j.out
#SBATCH --error=/work/cvcs2026/bleedsense/logs/%x_%j.err
#SBATCH --account=cvcs2026
# NB: QOS deliberately not specified: for student users it is assigned
# automatically as all_qos_sprod by the submission system.
#
# Usage: sbatch --job-name=<name> train.sh <dataset> <architecture> <fold> <augmentation> <adaptation> <joint_sampling> <blood_index> <mixstyle>
# Example: sbatch --job-name=hemoset_unet_f0 train.sh hemoset unet 0 light none
#          sbatch --job-name=hemoset_unet_f0_aug train.sh hemoset unet 0 aggressive none
#          sbatch --job-name=hemoset_unet_f0_reinhard train.sh hemoset unet 0 light reinhard
#          sbatch --job-name=joint_unet_f0_bal train.sh joint unet 0 light none balanced
#          sbatch --job-name=hemoset_unet_f0_bloodidx train.sh hemoset unet 0 aggressive none natural true
#          sbatch --job-name=hemoset_unet_f0_mixstyle train.sh hemoset unet 0 light none natural false true
#
# For a quick test (debug, high priority, max 1h/1GPU) override
# partition/qos/time from the command line, e.g.:
#   sbatch --qos=all_qos_dbg --time=00:10:00 --job-name=debug train.sh hemoset unet 0 light none

set -e

DATASET=$1
ARCHITECTURE=$2
FOLD=${3:-0}
AUGMENTATION=${4:-light}
ADAPTATION=${5:-none}
JOINT_SAMPLING=${6:-natural}
BLOOD_INDEX=${7:-false}
MIXSTYLE=${8:-false}

BLOOD_INDEX_FLAG=""
if [ "$BLOOD_INDEX" = "true" ] || [ "$BLOOD_INDEX" = "1" ]; then
    BLOOD_INDEX_FLAG="--use-blood-index"
fi

MIXSTYLE_FLAG=""
if [ "$MIXSTYLE" = "true" ] || [ "$MIXSTYLE" = "1" ]; then
    MIXSTYLE_FLAG="--mixstyle"
fi

mkdir -p /work/cvcs2026/bleedsense/logs

module unload cuda >/dev/null 2>&1 || true
module load cuda/12.6.3
source /homes/rmazzi/cvcs2026/venv/bin/activate

cd /work/cvcs2026/bleedsense

python train.py \
    --dataset "$DATASET" \
    --architecture "$ARCHITECTURE" \
    --fold "$FOLD" \
    --augmentation "$AUGMENTATION" \
    --adaptation "$ADAPTATION" \
    --joint-sampling "$JOINT_SAMPLING" \
    $BLOOD_INDEX_FLAG \
    $MIXSTYLE_FLAG
