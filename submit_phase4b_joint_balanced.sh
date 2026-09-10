#!/bin/bash
# Phase 4b (extra experiment): joint training with domain-balanced sampling
# (WeightedRandomSampler, ~50% HemoSet / ~50% Rabbani per batch regardless of the
# two datasets' sizes), to correct the imbalance observed in the "natural" Phase 4
# (where HemoSet weighed ~59% simply because it was more numerous).
# 3 architectures x 5 folds = 15 jobs.
# Launch from /work/cvcs2026/bleedsense on the cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="joint_${ARCH}_f${FOLD}_bal" train.sh joint "$ARCH" "$FOLD" light none balanced
    done
done
