# Test-time BatchNorm recalibration (AdaBN) on top of augmentation=aggressive

| Direction | Phase | DICE | IOU | PRECISION | RECALL | F1 | HD95 |
|---|---|---|---|---|---|---|---|
| HemoSet -> Rabbani (cross) | without recalibration | 0.321±0.029 | 0.226±0.024 | 0.382±0.053 | 0.391±0.052 | 0.321±0.029 | 93.912±6.761 |
| HemoSet -> Rabbani (cross) | with test-time BN recalibration | 0.318±0.025 | 0.218±0.020 | 0.283±0.033 | 0.546±0.036 | 0.318±0.025 | 104.253±6.685 |
| Rabbani -> HemoSet (cross) | without recalibration | 0.601±0.015 | 0.458±0.019 | 0.519±0.041 | 0.777±0.051 | 0.601±0.015 | 54.347±5.321 |
| Rabbani -> HemoSet (cross) | with test-time BN recalibration | 0.570±0.003 | 0.433±0.004 | 0.641±0.033 | 0.574±0.027 | 0.570±0.003 | 61.863±4.515 |
