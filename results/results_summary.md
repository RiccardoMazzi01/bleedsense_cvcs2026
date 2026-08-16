# BleedSense - Riepilogo risultati (in-domain vs cross-dataset)

Media +/- deviazione standard sui 5 fold per HemoSet; run singolo per Rabbani.
Righe assenti se quella combinazione non e' ancora stata eseguita.

| Setting | Architettura | Augmentation | Adaptation | DICE | IOU | PRECISION | RECALL | F1 | HD95 |
|---|---|---|---|---|---|---|---|---|---|
| HemoSet -> HemoSet (in-domain) | unet | light | none | 0.738±0.035 | 0.607±0.038 | 0.779±0.056 | 0.726±0.037 | 0.738±0.035 | 25.852±5.863 |
| HemoSet -> Rabbani (cross) | unet | light | none | 0.218±0.033 | 0.149±0.025 | 0.638±0.081 | 0.179±0.036 | 0.218±0.033 | 134.346±20.055 |
| Rabbani -> Rabbani (in-domain) | unet | light | none | 0.689 | 0.561 | 0.745 | 0.711 | 0.689 | 49.819 |
| Rabbani -> HemoSet (cross) | unet | light | none | 0.588 | 0.442 | 0.505 | 0.752 | 0.588 | 62.789 |
| HemoSet -> HemoSet (in-domain) | unet | light | reinhard | 0.709±0.081 | 0.573±0.096 | 0.735±0.157 | 0.727±0.060 | 0.709±0.081 | 31.177±20.222 |
| HemoSet -> Rabbani (cross) | unet | light | reinhard | 0.259±0.044 | 0.176±0.032 | 0.397±0.111 | 0.299±0.110 | 0.259±0.044 | 97.635±14.636 |
| Rabbani -> Rabbani (in-domain) | unet | light | reinhard | 0.640 | 0.519 | 0.758 | 0.649 | 0.640 | 62.700 |
| Rabbani -> HemoSet (cross) | unet | light | reinhard | 0.589 | 0.453 | 0.541 | 0.718 | 0.589 | 57.643 |
| HemoSet -> HemoSet (in-domain) | unet | light | fda | 0.709±0.074 | 0.575±0.087 | 0.773±0.052 | 0.683±0.104 | 0.709±0.074 | 33.596±18.038 |
| HemoSet -> Rabbani (cross) | unet | light | fda | 0.241±0.038 | 0.165±0.028 | 0.474±0.079 | 0.221±0.069 | 0.241±0.038 | 108.179±15.021 |
| Rabbani -> Rabbani (in-domain) | unet | light | fda | 0.686 | 0.563 | 0.763 | 0.694 | 0.686 | 53.186 |
| Rabbani -> HemoSet (cross) | unet | light | fda | 0.578 | 0.445 | 0.533 | 0.680 | 0.578 | 62.265 |
| HemoSet -> HemoSet (in-domain) | unet | aggressive | none | 0.756±0.033 | 0.629±0.035 | 0.752±0.049 | 0.794±0.025 | 0.756±0.033 | 26.835±5.965 |
| HemoSet -> Rabbani (cross) | unet | aggressive | none | 0.323±0.032 | 0.227±0.027 | 0.396±0.071 | 0.370±0.031 | 0.323±0.032 | 90.396±4.508 |
| Rabbani -> Rabbani (in-domain) | unet | aggressive | none | 0.672 | 0.546 | 0.732 | 0.699 | 0.672 | 38.526 |
| Rabbani -> HemoSet (cross) | unet | aggressive | none | 0.603 | 0.461 | 0.539 | 0.747 | 0.603 | 51.224 |
| HemoSet -> HemoSet (in-domain) | unetplusplus | light | none | 0.749±0.026 | 0.617±0.031 | 0.732±0.034 | 0.791±0.046 | 0.749±0.026 | 32.914±18.234 |
| HemoSet -> Rabbani (cross) | unetplusplus | light | none | 0.236±0.035 | 0.159±0.023 | 0.418±0.096 | 0.223±0.022 | 0.236±0.035 | 104.057±8.325 |
| Rabbani -> Rabbani (in-domain) | unetplusplus | light | none | 0.675 | 0.550 | 0.731 | 0.712 | 0.675 | 54.827 |
| Rabbani -> HemoSet (cross) | unetplusplus | light | none | 0.597 | 0.455 | 0.529 | 0.728 | 0.597 | 56.133 |
| HemoSet -> HemoSet (in-domain) | unetplusplus | light | reinhard | 0.735±0.036 | 0.602±0.042 | 0.760±0.050 | 0.739±0.040 | 0.735±0.036 | 31.047±12.715 |
| HemoSet -> Rabbani (cross) | unetplusplus | light | reinhard | 0.283±0.040 | 0.190±0.027 | 0.365±0.055 | 0.338±0.094 | 0.283±0.040 | 99.263±11.659 |
| Rabbani -> Rabbani (in-domain) | unetplusplus | light | reinhard | 0.621 | 0.501 | 0.759 | 0.622 | 0.621 | 63.865 |
| Rabbani -> HemoSet (cross) | unetplusplus | light | reinhard | 0.580 | 0.440 | 0.521 | 0.711 | 0.580 | 54.977 |
| HemoSet -> HemoSet (in-domain) | unetplusplus | light | fda | 0.724±0.044 | 0.591±0.050 | 0.759±0.070 | 0.728±0.095 | 0.724±0.044 | 27.940±9.329 |
| HemoSet -> Rabbani (cross) | unetplusplus | light | fda | 0.252±0.026 | 0.171±0.020 | 0.374±0.102 | 0.277±0.064 | 0.252±0.026 | 99.942±10.204 |
| Rabbani -> Rabbani (in-domain) | unetplusplus | light | fda | 0.672 | 0.539 | 0.664 | 0.757 | 0.672 | 48.767 |
| Rabbani -> HemoSet (cross) | unetplusplus | light | fda | 0.513 | 0.367 | 0.392 | 0.802 | 0.513 | 74.231 |
| HemoSet -> HemoSet (in-domain) | unetplusplus | aggressive | none | 0.756±0.022 | 0.629±0.025 | 0.793±0.053 | 0.753±0.020 | 0.756±0.022 | 28.457±8.364 |
| HemoSet -> Rabbani (cross) | unetplusplus | aggressive | none | 0.304±0.028 | 0.212±0.022 | 0.349±0.019 | 0.367±0.051 | 0.304±0.028 | 97.351±7.258 |
| Rabbani -> Rabbani (in-domain) | unetplusplus | aggressive | none | 0.701 | 0.580 | 0.734 | 0.741 | 0.701 | 43.259 |
| Rabbani -> HemoSet (cross) | unetplusplus | aggressive | none | 0.615 | 0.475 | 0.546 | 0.749 | 0.615 | 60.491 |
| HemoSet -> HemoSet (in-domain) | deeplabv3plus | light | none | 0.748±0.027 | 0.617±0.026 | 0.775±0.050 | 0.751±0.055 | 0.748±0.027 | 27.977±11.522 |
| HemoSet -> Rabbani (cross) | deeplabv3plus | light | none | 0.196±0.032 | 0.133±0.025 | 0.665±0.074 | 0.165±0.035 | 0.196±0.032 | 151.249±20.629 |
| Rabbani -> Rabbani (in-domain) | deeplabv3plus | light | none | 0.689 | 0.558 | 0.740 | 0.709 | 0.689 | 39.340 |
| Rabbani -> HemoSet (cross) | deeplabv3plus | light | none | 0.569 | 0.420 | 0.476 | 0.762 | 0.569 | 57.188 |
| HemoSet -> HemoSet (in-domain) | deeplabv3plus | light | reinhard | 0.713±0.051 | 0.575±0.061 | 0.730±0.105 | 0.731±0.060 | 0.713±0.051 | 34.398±14.164 |
| HemoSet -> Rabbani (cross) | deeplabv3plus | light | reinhard | 0.247±0.034 | 0.169±0.026 | 0.394±0.106 | 0.262±0.072 | 0.247±0.034 | 106.050±7.321 |
| Rabbani -> Rabbani (in-domain) | deeplabv3plus | light | reinhard | 0.571 | 0.452 | 0.774 | 0.568 | 0.571 | 83.954 |
| Rabbani -> HemoSet (cross) | deeplabv3plus | light | reinhard | 0.561 | 0.416 | 0.469 | 0.763 | 0.561 | 51.960 |
| HemoSet -> HemoSet (in-domain) | deeplabv3plus | light | fda | 0.728±0.029 | 0.594±0.028 | 0.760±0.050 | 0.723±0.027 | 0.728±0.029 | 30.026±10.805 |
| HemoSet -> Rabbani (cross) | deeplabv3plus | light | fda | 0.234±0.032 | 0.160±0.025 | 0.526±0.068 | 0.204±0.045 | 0.234±0.032 | 115.712±13.686 |
| Rabbani -> Rabbani (in-domain) | deeplabv3plus | light | fda | 0.666 | 0.537 | 0.692 | 0.704 | 0.666 | 41.395 |
| Rabbani -> HemoSet (cross) | deeplabv3plus | light | fda | 0.560 | 0.421 | 0.480 | 0.713 | 0.560 | 61.675 |
| HemoSet -> HemoSet (in-domain) | deeplabv3plus | aggressive | none | 0.730±0.039 | 0.599±0.045 | 0.736±0.061 | 0.757±0.030 | 0.730±0.039 | 29.543±11.536 |
| HemoSet -> Rabbani (cross) | deeplabv3plus | aggressive | none | 0.337±0.021 | 0.239±0.019 | 0.401±0.047 | 0.436±0.045 | 0.337±0.021 | 93.988±7.533 |
| Rabbani -> Rabbani (in-domain) | deeplabv3plus | aggressive | none | 0.671 | 0.539 | 0.689 | 0.747 | 0.671 | 42.393 |
| Rabbani -> HemoSet (cross) | deeplabv3plus | aggressive | none | 0.585 | 0.438 | 0.471 | 0.835 | 0.585 | 51.325 |
