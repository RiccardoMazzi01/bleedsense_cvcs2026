#!/bin/bash
# Sottomette tutti i run della Fase 1 (baseline): 3 architetture x (5 fold HemoSet + 1 run Rabbani) = 18 job.
# Da lanciare da /work/cvcs2026/bleedsense sul cluster, dopo aver verificato train.sh con un run di debug.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}" train.sh hemoset "$ARCH" "$FOLD"
    done
    sbatch --job-name="rabbani_${ARCH}" train.sh rabbani "$ARCH"
done
