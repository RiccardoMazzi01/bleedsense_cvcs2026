#!/bin/bash
# Sottomette i run della Fase 4 (joint training): training su HemoSet+Rabbani insieme,
# valutazione separata sui due test set held-out (mai mescolati).
# Augmentation 'light' e adaptation 'none' per isolare l'effetto del solo joint training
# rispetto alle baseline single-dataset della Fase 1.
# 3 architetture x 5 fold HemoSet (accoppiate allo stesso train split di Rabbani) = 15 job.
# Da lanciare da /work/cvcs2026/bleedsense sul cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="joint_${ARCH}_f${FOLD}" train.sh joint "$ARCH" "$FOLD" light none
    done
done
