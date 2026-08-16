# BleedSense - Riepilogo risultati (in-domain vs cross-dataset)

Media +/- deviazione standard sui 5 fold per HemoSet; run singolo per Rabbani.
Righe 'aggressive' assenti se la Fase 2 non e' ancora stata eseguita.

| Setting | Architettura | Augmentation | DICE | IOU | PRECISION | RECALL | F1 | HD95 |
|---|---|---|---|---|---|---|---|---|
| HemoSet -> HemoSet (in-domain) | unet | light | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet -> Rabbani (cross) | unet | light | 0.218±0.033 | 0.149±0.025 | 0.638±0.081 | 0.179±0.036 | 0.218±0.033 | 134.346±20.055 |
| Rabbani -> Rabbani (in-domain) | unet | light | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani -> HemoSet (cross) | unet | light | 0.588 | 0.442 | 0.505 | 0.752 | 0.588 | 62.789 |
| HemoSet -> HemoSet (in-domain) | unet | aggressive | 0.756±0.033 | 0.629±0.035 | 0.752±0.049 | 0.794±0.025 | 0.756±0.033 | 26.835±5.965 |
| HemoSet -> Rabbani (cross) | unet | aggressive | 0.323±0.032 | 0.227±0.027 | 0.396±0.071 | 0.370±0.031 | 0.323±0.032 | 90.396±4.508 |
| Rabbani -> Rabbani (in-domain) | unet | aggressive | 0.672 | 0.546 | 0.732 | 0.699 | 0.672 | 38.526 |
| Rabbani -> HemoSet (cross) | unet | aggressive | 0.603 | 0.461 | 0.539 | 0.747 | 0.603 | 51.224 |
| HemoSet -> HemoSet (in-domain) | unetplusplus | light | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet -> Rabbani (cross) | unetplusplus | light | 0.236±0.035 | 0.159±0.023 | 0.418±0.096 | 0.223±0.022 | 0.236±0.035 | 104.057±8.325 |
| Rabbani -> Rabbani (in-domain) | unetplusplus | light | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani -> HemoSet (cross) | unetplusplus | light | 0.597 | 0.455 | 0.529 | 0.728 | 0.597 | 56.133 |
| HemoSet -> HemoSet (in-domain) | unetplusplus | aggressive | 0.756±0.022 | 0.629±0.025 | 0.793±0.053 | 0.753±0.020 | 0.756±0.022 | 28.457±8.364 |
| HemoSet -> Rabbani (cross) | unetplusplus | aggressive | 0.304±0.028 | 0.212±0.022 | 0.349±0.019 | 0.367±0.051 | 0.304±0.028 | 97.351±7.258 |
| Rabbani -> Rabbani (in-domain) | unetplusplus | aggressive | 0.701 | 0.580 | 0.734 | 0.741 | 0.701 | 43.259 |
| Rabbani -> HemoSet (cross) | unetplusplus | aggressive | 0.615 | 0.475 | 0.546 | 0.749 | 0.615 | 60.491 |
| HemoSet -> HemoSet (in-domain) | deeplabv3plus | light | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet -> Rabbani (cross) | deeplabv3plus | light | 0.196±0.032 | 0.133±0.025 | 0.665±0.074 | 0.165±0.035 | 0.196±0.032 | 151.249±20.629 |
| Rabbani -> Rabbani (in-domain) | deeplabv3plus | light | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani -> HemoSet (cross) | deeplabv3plus | light | 0.569 | 0.420 | 0.476 | 0.762 | 0.569 | 57.188 |
| HemoSet -> HemoSet (in-domain) | deeplabv3plus | aggressive | 0.730±0.039 | 0.599±0.045 | 0.736±0.061 | 0.757±0.030 | 0.730±0.039 | 29.543±11.536 |
| HemoSet -> Rabbani (cross) | deeplabv3plus | aggressive | 0.337±0.021 | 0.239±0.019 | 0.401±0.047 | 0.436±0.045 | 0.337±0.021 | 93.988±7.533 |
| Rabbani -> Rabbani (in-domain) | deeplabv3plus | aggressive | 0.671 | 0.539 | 0.689 | 0.747 | 0.671 | 42.393 |
| Rabbani -> HemoSet (cross) | deeplabv3plus | aggressive | 0.585 | 0.438 | 0.471 | 0.835 | 0.585 | 51.325 |
