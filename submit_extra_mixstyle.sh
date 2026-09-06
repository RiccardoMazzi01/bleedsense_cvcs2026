#!/bin/bash
# Estensione extra (dopo il tentativo negativo col blood-index): MixStyle (Zhou et al.,
# ICLR 2021), style mixing a livello di feature dentro l'encoder invece che sui pixel.
# Testato su augmentation 'light' / adaptation 'none' (stessa condizione della Fase 1 e
# della baseline 'none' in Fase 3), per isolare l'effetto del solo MixStyle e confrontarlo
# direttamente sia con la baseline sia con Reinhard/FDA sulla stessa base.
# 3 architetture x (5 fold HemoSet + 1 run Rabbani) = 18 job.
# Da lanciare da /work/cvcs2026/bleedsense sul cluster.
set -e

ARCHITECTURES=(unet unetplusplus deeplabv3plus)

for ARCH in "${ARCHITECTURES[@]}"; do
    for FOLD in 0 1 2 3 4; do
        sbatch --job-name="hemoset_${ARCH}_f${FOLD}_mixstyle" train.sh hemoset "$ARCH" "$FOLD" light none natural false true
    done
    sbatch --job-name="rabbani_${ARCH}_mixstyle" train.sh rabbani "$ARCH" 0 light none natural false true
done
