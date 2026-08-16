#!/bin/bash
# Sottomette i run della Fase 2 (augmentation 'aggressive'):
# 3 architetture x (5 fold HemoSet + 1 run Rabbani) = 18 job.
# La condizione 'light' e' gia' coperta dai risultati della Fase 1 (submit_phase1_baseline.sh),
# quindi qui si lancia solo la condizione aggressive per il confronto.
# Da lanciare da /work/cvcs2026/bleedsense sul cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}_aug" train.sh hemoset "$ARCH" "$FOLD" aggressive
    done
    sbatch --job-name="rabbani_${ARCH}_aug" train.sh rabbani "$ARCH" 0 aggressive
done
