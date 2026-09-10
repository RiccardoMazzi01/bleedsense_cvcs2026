#!/bin/bash
# Submits all Phase 1 runs (baseline, 'light' augmentation):
# 3 architectures x (5 HemoSet folds + 1 Rabbani run) = 18 jobs.
# Launch from /work/cvcs2026/bleedsense on the cluster, after verifying train.sh with a debug run.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}" train.sh hemoset "$ARCH" "$FOLD" light
    done
    sbatch --job-name="rabbani_${ARCH}" train.sh rabbani "$ARCH" 0 light
done
