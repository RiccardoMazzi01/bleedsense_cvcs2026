#!/bin/bash
# Extra extension (after the negative blood-index attempt): MixStyle (Zhou et al.,
# ICLR 2021), feature-level style mixing inside the encoder instead of on pixels.
# Tested on 'light' augmentation / 'none' adaptation (same condition as Phase 1 and
# the 'none' baseline in Phase 3), to isolate the effect of MixStyle alone and compare
# it directly against both the baseline and Reinhard/FDA on the same basis.
# 3 architectures x (5 HemoSet folds + 1 Rabbani run) = 18 jobs.
# Launch from /work/cvcs2026/bleedsense on the cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}_mixstyle" train.sh hemoset "$ARCH" "$FOLD" light none natural false true
    done
    sbatch --job-name="rabbani_${ARCH}_mixstyle" train.sh rabbani "$ARCH" 0 light none natural false true
done
