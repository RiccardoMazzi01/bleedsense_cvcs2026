#!/bin/bash
# Submits the Phase 3 runs (appearance/domain adaptation): Reinhard color transfer
# and Fourier Domain Adaptation (FDA), 'light' augmentation as in Phase 1 to isolate
# the effect of adaptation. 3 architectures x (5 HemoSet folds + 1 Rabbani run) x 2 methods
# = 36 jobs.
# Launch from /work/cvcs2026/bleedsense on the cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)
ADAPTATIONS=(reinhard fda)

for ARCH in "${ARCHITECTURES[@]}"; do
    for ADAPT in "${ADAPTATIONS[@]}"; do
        for FOLD in 0 1 2 3 4; do
            sbatch --job-name="hemoset_${ARCH}_f${FOLD}_${ADAPT}" train.sh hemoset "$ARCH" "$FOLD" light "$ADAPT"
        done
        sbatch --job-name="rabbani_${ARCH}_${ADAPT}" train.sh rabbani "$ARCH" 0 light "$ADAPT"
    done
done
