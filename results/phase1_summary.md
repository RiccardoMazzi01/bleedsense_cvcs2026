# Phase 1 - Baseline results (in-domain vs cross-dataset)

Mean +/- standard deviation over the 5 HemoSet folds; single run for Rabbani.

| Setting | Architecture | DICE | IOU | PRECISION | RECALL | F1 | HD95 |
|---|---|---|---|---|---|---|---|
| HemoSet -> HemoSet (in-domain) | unet | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet -> Rabbani (cross) | unet | 0.218±0.033 | 0.149±0.025 | 0.638±0.081 | 0.179±0.036 | 0.218±0.033 | 134.346±20.055 |
| Rabbani -> Rabbani (in-domain) | unet | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani -> HemoSet (cross) | unet | 0.588 | 0.442 | 0.505 | 0.752 | 0.588 | 62.789 |
| HemoSet -> HemoSet (in-domain) | unetplusplus | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet -> Rabbani (cross) | unetplusplus | 0.236±0.035 | 0.159±0.023 | 0.418±0.096 | 0.223±0.022 | 0.236±0.035 | 104.057±8.325 |
| Rabbani -> Rabbani (in-domain) | unetplusplus | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani -> HemoSet (cross) | unetplusplus | 0.597 | 0.455 | 0.529 | 0.728 | 0.597 | 56.133 |
| HemoSet -> HemoSet (in-domain) | deeplabv3plus | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet -> Rabbani (cross) | deeplabv3plus | 0.196±0.032 | 0.133±0.025 | 0.665±0.074 | 0.165±0.035 | 0.196±0.032 | 151.249±20.629 |
| Rabbani -> Rabbani (in-domain) | deeplabv3plus | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani -> HemoSet (cross) | deeplabv3plus | 0.569 | 0.420 | 0.476 | 0.762 | 0.569 | 57.188 |
