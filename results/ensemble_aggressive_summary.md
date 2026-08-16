# Ensemble delle 3 architetture (augmentation=aggressive)

| Setting | DICE | IOU | PRECISION | RECALL | F1 | HD95 |
|---|---|---|---|---|---|---|
| HemoSet -> HemoSet (in-domain, ensemble) | 0.770±0.025 | 0.647±0.028 | 0.798±0.054 | 0.773±0.015 | 0.770±0.025 | 24.637±5.623 |
| HemoSet -> Rabbani (cross, ensemble) | 0.343±0.014 | 0.245±0.012 | 0.433±0.030 | 0.379±0.029 | 0.343±0.014 | 91.457±4.161 |
| Rabbani -> Rabbani (in-domain, ensemble) | 0.706 | 0.583 | 0.757 | 0.739 | 0.706 | 39.571 |
| Rabbani -> HemoSet (cross, ensemble) | 0.623 | 0.481 | 0.544 | 0.777 | 0.623 | 49.629 |
