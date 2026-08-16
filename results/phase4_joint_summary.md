# Fase 4 - Joint training vs single-source (per dominio di test)

Media +/- deviazione standard sui 5 fold HemoSet; Rabbani single-source e' un run singolo.

| Test set | Architettura | Training | DICE | IOU | PRECISION | RECALL | F1 | HD95 |
|---|---|---|---|---|---|---|---|---|
| HemoSet test | unet | single-source (Fase 1) | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet test | unet | joint (Fase 4) | 0.747±0.041 | 0.620±0.044 | 0.756±0.045 | 0.769±0.051 | 0.747±0.041 | 25.614±8.944 |
| Rabbani test | unet | single-source (Fase 1) | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani test | unet | joint (Fase 4) | 0.638±0.053 | 0.511±0.055 | 0.688±0.081 | 0.717±0.052 | 0.638±0.053 | 59.293±9.693 |
| HemoSet test | unetplusplus | single-source (Fase 1) | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet test | unetplusplus | joint (Fase 4) | 0.757±0.036 | 0.630±0.039 | 0.729±0.051 | 0.814±0.038 | 0.757±0.036 | 30.395±8.579 |
| Rabbani test | unetplusplus | single-source (Fase 1) | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani test | unetplusplus | joint (Fase 4) | 0.631±0.056 | 0.504±0.059 | 0.683±0.088 | 0.701±0.038 | 0.631±0.056 | 60.076±10.358 |
| HemoSet test | deeplabv3plus | single-source (Fase 1) | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet test | deeplabv3plus | joint (Fase 4) | 0.721±0.053 | 0.590±0.055 | 0.759±0.077 | 0.734±0.075 | 0.721±0.053 | 29.856±9.249 |
| Rabbani test | deeplabv3plus | single-source (Fase 1) | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani test | deeplabv3plus | joint (Fase 4) | 0.640±0.034 | 0.511±0.031 | 0.724±0.022 | 0.668±0.039 | 0.640±0.034 | 56.070±7.804 |
