#!/bin/bash
# Sottomette i run della Fase 3 (appearance/domain adaptation): Reinhard color transfer
# e Fourier Domain Adaptation (FDA), augmentation 'light' come nella Fase 1 per isolare
# l'effetto dell'adaptation. 3 architetture x (5 fold HemoSet + 1 run Rabbani) x 2 metodi
# = 36 job.
# Da lanciare da /work/cvcs2026/bleedsense sul cluster.
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
