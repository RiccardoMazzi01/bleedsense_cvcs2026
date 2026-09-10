#!/bin/bash
# Extra extension (post Phase 5): additional "blood-index" input channel (R/(R+G+B)),
# meant to be less sensitive to camera/illumination differences between HemoSet and
# Rabbani. Tested on top of the best condition so far (aggressive augmentation), to
# see whether it improves cross-dataset robustness further compared to Phase 2.
# 3 architectures x (5 HemoSet folds + 1 Rabbani run) = 18 jobs.
# Launch from /work/cvcs2026/bleedsense on the cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}_bloodidx" train.sh hemoset "$ARCH" "$FOLD" aggressive none natural true
    done
    sbatch --job-name="rabbani_${ARCH}_bloodidx" train.sh rabbani "$ARCH" 0 aggressive none natural true
done
