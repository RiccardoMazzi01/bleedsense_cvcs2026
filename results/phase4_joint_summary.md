# Fase 4/4b - Joint training (naturale vs bilanciato) vs single-source

Media +/- deviazione standard sui 5 fold HemoSet; Rabbani single-source e' un run singolo.

| Test set | Architettura | Training | DICE | IOU | PRECISION | RECALL | F1 | HD95 |
|---|---|---|---|---|---|---|---|---|
| HemoSet test | unet | single-source (Fase 1) | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet test | unet | joint naturale (Fase 4) | 0.747±0.041 | 0.620±0.044 | 0.756±0.045 | 0.769±0.051 | 0.747±0.041 | 25.614±8.944 |
| HemoSet test | unet | joint bilanciato (Fase 4b) | 0.742±0.047 | 0.613±0.048 | 0.753±0.071 | 0.766±0.064 | 0.742±0.047 | 29.085±9.659 |
| Rabbani test | unet | single-source (Fase 1) | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani test | unet | joint naturale (Fase 4) | 0.638±0.053 | 0.511±0.055 | 0.688±0.081 | 0.717±0.052 | 0.638±0.053 | 59.293±9.693 |
| Rabbani test | unet | joint bilanciato (Fase 4b) | 0.658±0.008 | 0.532±0.012 | 0.734±0.075 | 0.700±0.061 | 0.658±0.008 | 58.447±4.808 |
| HemoSet test | unetplusplus | single-source (Fase 1) | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet test | unetplusplus | joint naturale (Fase 4) | 0.757±0.036 | 0.630±0.039 | 0.729±0.051 | 0.814±0.038 | 0.757±0.036 | 30.395±8.579 |
| HemoSet test | unetplusplus | joint bilanciato (Fase 4b) | 0.743±0.036 | 0.612±0.038 | 0.721±0.064 | 0.795±0.050 | 0.743±0.036 | 35.014±11.039 |
| Rabbani test | unetplusplus | single-source (Fase 1) | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani test | unetplusplus | joint naturale (Fase 4) | 0.631±0.056 | 0.504±0.059 | 0.683±0.088 | 0.701±0.038 | 0.631±0.056 | 60.076±10.358 |
| Rabbani test | unetplusplus | joint bilanciato (Fase 4b) | 0.658±0.027 | 0.532±0.030 | 0.736±0.044 | 0.692±0.017 | 0.658±0.027 | 55.882±10.242 |
| HemoSet test | deeplabv3plus | single-source (Fase 1) | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet test | deeplabv3plus | joint naturale (Fase 4) | 0.721±0.053 | 0.590±0.055 | 0.759±0.077 | 0.734±0.075 | 0.721±0.053 | 29.856±9.249 |
| HemoSet test | deeplabv3plus | joint bilanciato (Fase 4b) | 0.731±0.048 | 0.599±0.047 | 0.746±0.069 | 0.748±0.051 | 0.731±0.048 | 29.748±10.083 |
| Rabbani test | deeplabv3plus | single-source (Fase 1) | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani test | deeplabv3plus | joint naturale (Fase 4) | 0.640±0.034 | 0.511±0.031 | 0.724±0.022 | 0.668±0.039 | 0.640±0.034 | 56.070±7.804 |
| Rabbani test | deeplabv3plus | joint bilanciato (Fase 4b) | 0.653±0.017 | 0.525±0.019 | 0.722±0.044 | 0.687±0.033 | 0.653±0.017 | 51.383±4.134 |
