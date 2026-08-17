#!/bin/bash
# Fase 4b (esperimento extra): joint training con campionamento bilanciato tra domini
# (WeightedRandomSampler, ~50% HemoSet / ~50% Rabbani per batch indipendentemente dalla
# dimensione dei due dataset), per correggere lo sbilanciamento osservato nella Fase 4
# "naturale" (dove HemoSet pesava ~59% solo perche' piu' numeroso).
# 3 architetture x 5 fold = 15 job.
# Da lanciare da /work/cvcs2026/bleedsense sul cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="joint_${ARCH}_f${FOLD}_bal" train.sh joint "$ARCH" "$FOLD" light none balanced
    done
done
