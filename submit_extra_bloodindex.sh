#!/bin/bash
# Estensione extra (post Fase 5): canale di input aggiuntivo "blood-index" (R/(R+G+B)),
# pensato per essere meno sensibile a differenze di camera/illuminazione tra HemoSet e
# Rabbani. Testato sopra la condizione migliore finora (augmentation aggressive), per
# vedere se migliora ulteriormente la robustezza cross-dataset rispetto alla Fase 2.
# 3 architetture x (5 fold HemoSet + 1 run Rabbani) = 18 job.
# Da lanciare da /work/cvcs2026/bleedsense sul cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}_bloodidx" train.sh hemoset "$ARCH" "$FOLD" aggressive none natural true
    done
    sbatch --job-name="rabbani_${ARCH}_bloodidx" train.sh rabbani "$ARCH" 0 aggressive none natural true
done
