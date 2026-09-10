#!/bin/bash
# Submits the Phase 2 runs ('aggressive' augmentation):
# 3 architectures x (5 HemoSet folds + 1 Rabbani run) = 18 jobs.
# The 'light' condition is already covered by the Phase 1 results (submit_phase1_baseline.sh),
# so only the aggressive condition is launched here for comparison.
# Launch from /work/cvcs2026/bleedsense on the cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}_aug" train.sh hemoset "$ARCH" "$FOLD" aggressive
    done
    sbatch --job-name="rabbani_${ARCH}_aug" train.sh rabbani "$ARCH" 0 aggressive
done
