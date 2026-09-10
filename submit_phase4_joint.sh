#!/bin/bash
# Submits the Phase 4 runs (joint training): training on HemoSet+Rabbani together,
# evaluated separately on the two held-out test sets (never mixed).
# 'light' augmentation and 'none' adaptation to isolate the effect of joint training
# alone against the single-dataset baselines from Phase 1.
# 3 architectures x 5 HemoSet folds (each paired with the same Rabbani train split) = 15 jobs.
# Launch from /work/cvcs2026/bleedsense on the cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="joint_${ARCH}_f${FOLD}" train.sh joint "$ARCH" "$FOLD" light none
    done
done
