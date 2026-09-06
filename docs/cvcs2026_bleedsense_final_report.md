# BleedSense — Robustezza cross-dataset per la segmentazione del sangue in chirurgia robotica

**Report finale — Corso di Computer Vision and Cognitive Systems (CVCS), A.A. 2025/2026, UNIMORE**

Riccardo Mazzi (294437)
Supervisione: Dott.ssa Livia Del Gaudio (AImageLab, progetto TRAMIS)

---

## Abstract

La gestione automatica dell'emostasi in chirurgia robotica richiede la segmentazione affidabile del sangue nel campo operatorio. I modelli di segmentazione allenati su un singolo dataset chirurgico generalizzano tuttavia male su altri dataset con caratteristiche visive diverse (illuminazione, tessuto, strumentazione). Questo lavoro studia il problema su due dataset pubblici, **HemoSet** (chirurgia robotica su modello suino) e **Rabbani et al.** (laparoscopia ginecologica su umani), confermando un **domain gap asimmetrico** già osservato in lavori precedenti del gruppo di ricerca, e confronta sistematicamente sei famiglie di strategie per ridurlo: data augmentation aggressiva, appearance/domain adaptation (color transfer di Reinhard, Fourier Domain Adaptation), joint training multi-dominio (con e senza bilanciamento tra domini), ensembling di più architetture, un canale di input invariante all'illuminazione e style-mixing a livello di feature (MixStyle), oltre a una ricalibrazione delle statistiche BatchNorm a inferenza. Su tre architetture di segmentazione standard (UNet, UNet++, DeepLabV3+, encoder `resnet34`), la strategia più efficace risulta la **data augmentation aggressiva**, che migliora la robustezza cross-dataset in entrambe le direzioni senza costi apprezzabili in-domain; combinata con un **ensemble delle tre architetture** in fase di inferenza, produce il miglior risultato complessivo su tutte le condizioni testate. Le tre estensioni più originali tentate in un secondo momento non hanno migliorato ulteriormente questo risultato, ma la loro analisi fornisce indicazioni utili sul motivo per cui interventi più mirati falliscono in questo setting.

---

## 1. Introduzione e motivazione

L'automazione della gestione dell'emostasi in chirurgia robotica — l'identificazione e il controllo del sanguinamento durante l'intervento — richiede come primo passo la segmentazione affidabile del sangue nel campo visivo endoscopico. Questo progetto si inserisce nell'iniziativa di ricerca **TRAMIS**, orientata alla rilevazione in tempo reale del sanguinamento in chirurgia robotica; poiché al momento di questo lavoro non erano ancora disponibili dati chirurgici reali annotati del progetto, lo studio è stato condotto su due dataset pubblici di riferimento nel dominio, usati come proxy:

- **HemoSet** (Miao et al., ISMR 2024): sanguinamenti indotti durante chirurgia robotica teleoperata su modello suino, 962 coppie immagine-maschera, 10 soggetti.
- **Rabbani et al.** (MIDL 2022): laparoscopia ginecologica su pazienti umani, 751 immagini annotate.

Un lavoro precedente del gruppo di ricerca (Giusti & Marzo, report CVCS 2025/2026, si veda Sezione 2) aveva osservato un calo di performance **asimmetrico** quando un modello allenato su uno dei due dataset viene valutato sull'altro: **HemoSet → Rabbani** mostra un forte calo generale delle performance, mentre **Rabbani → HemoSet** mostra un aumento marcato dei falsi positivi (regioni non ematiche segmentate come sangue).

**Obiettivo del progetto**: quantificare questa asimmetria in modo rigoroso e riproducibile, e studiare sistematicamente quali strategie aumentano effettivamente la robustezza cross-dataset dei modelli di segmentazione, confrontando più condizioni sperimentali senza vincolo di convergere su un'unica soluzione.

---

## 2. Lavori correlati

- **Miao et al., "HemoSet: The First Blood Segmentation Dataset for Automation of Hemostasis Management"**, ISMR 2024 — dataset HemoSet; benchmark di modelli di segmentazione standard, metriche IoU/F1/Hausdorff Distance.
- **Rabbani, Seve, Bourdel & Bartoli, "Video-based Computer-aided Laparoscopic Bleeding Management..."**, MIDL 2022 — dataset e sistema di segmentazione del sanguinamento in laparoscopia ginecologica, con una componente di adversarial domain adaptation; metriche IoU e F-Score.
- **Yang & Soatto, "FDA: Fourier Domain Adaptation for Semantic Segmentation"**, CVPR 2020 — adattamento di dominio non supervisionato tramite scambio delle componenti a bassa frequenza nello spettro di Fourier, usato in questo lavoro come metodo di appearance adaptation (Sezione 6).
- **Reinhard, Ashikhmin, Gooch & Shirley, "Color Transfer between Images"**, IEEE CG&A 2001 — color transfer statistico nello spazio L\*a\*b\*, secondo metodo di appearance adaptation (Sezione 6).
- **Zhou et al., "Domain Generalization with MixStyle"**, ICLR 2021 — style mixing a livello di feature per la domain generalization single-source, usato in questo lavoro come estensione (Sezione 8.2).
- **Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation" (AdaBN)**, 2016 — ricalibrazione delle statistiche BatchNorm sul dominio target a inferenza, usata in questo lavoro come estensione (Sezione 8.2).
- **Su et al.**, AAAI 2023 — domain generalization single-source tramite augmentation, riferimento per l'interpretazione dei risultati di data augmentation (Sezione 5).
- **Giusti & Marzo, report di progetto CVCS 2025/2026** — prima quantificazione del domain gap asimmetrico tra HemoSet e Rabbani (Dice zero-shot: HemoSet→Rabbani 0.29, Rabbani→HemoSet 0.56), risultati qualitativamente coerenti con quelli di Sezione 4 nonostante split e iperparametri diversi.

---

## 3. Approccio

### 3.1 Dataset e split

- **HemoSet**: 962 coppie immagine-maschera, 10 soggetti (pig). Split **Stratified Group K-Fold** a 5 fold, con il soggetto come gruppo, per evitare che lo stesso animale compaia sia in training sia in validation nello stesso fold.
- **Rabbani**: 751 immagini/maschere. Split fisso **70/15/15** (train/val/test, seed 42), riusato identico in tutte le fasi del progetto.

### 3.2 Architetture e protocollo di training

Tre architetture standard tramite `segmentation_models_pytorch`, stesso encoder (`resnet34`) per un confronto equo: **UNet**, **UNet++**, **DeepLabV3+**. Loss Dice+BCE, ottimizzatore AdamW (lr 1e-4), scheduler `ReduceLROnPlateau`, batch size 8, immagini 256×256, dataset completo per ogni split. Early stopping su Dice di validazione (patience 8, cap 40 epoche) — verificato empiricamente su tutti gli 87 run di Fasi 1-4 che il budget fosse sufficiente per la convergenza (solo 2/87 run hanno raggiunto il cap, entrambi con un residuo di miglioramento trascurabile, <0.01 Dice). Stessa configurazione identica per ogni condizione confrontata, per isolare l'effetto della sola variabile in esame di volta in volta.

### 3.3 Metriche e protocollo di valutazione

Dice, IoU, Precision, Recall, F1, **HD95** (Hausdorff Distance al 95° percentile) — l'insieme copre per intero le metriche dei paper originali di entrambi i dataset (IoU/F1/HD per HemoSet, IoU/F-Score per Rabbani), con HD95 preferito all'HD massimo perché meno sensibile a singoli pixel outlier. Protocollo: un modello allenato su HemoSet è valutato in-domain sul fold di validazione e cross-dataset sul test set fisso di Rabbani; un modello allenato su Rabbani è valutato in-domain sul test set fisso di Rabbani e cross-dataset su tutto HemoSet (mai visto in training).

---

## 4. Risultati

### 4.1 Baseline e asimmetria del domain gap (Fase 1)

Le tre architetture allenate separatamente sui due dataset (augmentation minima) confermano e quantificano l'asimmetria osservata in lavori precedenti: **HemoSet→Rabbani** il Dice medio crolla di circa il 70% relativo (0.745 → 0.217), con HD95 che passa da ~29px a ~130px, e la causa è un crollo del **recall** (0.76→0.19): il modello diventa troppo conservativo sul dominio target, mancando la maggior parte del sangue reale. **Rabbani→HemoSet** il calo è più contenuto (0.684 → 0.585) ma la **precision** crolla (0.74→0.50) mentre il recall resta alto — coerente con l'osservazione originale sui falsi positivi. Le tre architetture si comportano in modo molto simile in ogni condizione: nessuna risolve da sola il domain gap, motivando lo spostamento del focus su strategie di training/adattamento piuttosto che sulla scelta architetturale.

### 4.2 Data augmentation, appearance adaptation, joint training (Fasi 2-4)

Un'augmentation "aggressiva" (hue/saturation, gamma, blur, rumore, compressione JPEG, via Albumentations) **migliora la robustezza cross-dataset su tutte e tre le architetture in entrambe le direzioni**, quasi senza costo in-domain (HemoSet→Rabbani: 0.217→0.321, +48% relativo; Rabbani→HemoSet: 0.585→0.601). Due metodi di appearance adaptation — color transfer di Reinhard (2001) e Fourier Domain Adaptation (Yang & Soatto, CVPR 2020), applicati usando immagini non annotate del dominio target — aiutano la direzione più critica (HemoSet→Rabbani: 0.217→0.263 con Reinhard, →0.242 con FDA) ma **a un costo in-domain non trascurabile** (es. Rabbani in-domain: 0.684→0.611 con Reinhard) e non battono l'augmentation aggressiva. L'ipotesi è che, a differenza dell'augmentation (probabilistica, applicata con una certa probabilità per immagine), l'adaptation qui sia **sempre attiva** durante il training: il modello non vede mai l'aspetto "pulito" del proprio dominio sorgente, e finisce diviso tra due stili invece di consolidarne bene uno con variazioni.

Allenare sull'unione di HemoSet e Rabbani (**joint training**, con test set held-out separati) non è risultato una soluzione: nessun beneficio sistematico su HemoSet, e un calo sistematico su Rabbani (0.684→0.636) dovuto allo sbilanciamento tra le dimensioni dei due dataset nel training set combinato (~59% HemoSet vs ~41% Rabbani). Un campionamento bilanciato tra domini (`WeightedRandomSampler`, ~50/50 per batch) recupera solo metà del calo (0.636→0.656), senza rendere il joint training competitivo con le altre strategie.

### 4.3 Ensemble delle tre architetture

Come contributo originale oltre la roadmap indicata, si è ensemblato — a costo computazionale nullo (nessun training aggiuntivo, solo media delle probabilità sigmoid dei checkpoint già allenati con augmentation aggressiva) — le predizioni delle tre architetture. Il risultato batte il migliore dei tre modelli singoli su **tutte e quattro** le condizioni testate, con guadagni modesti ma sistematici e senza eccezioni (es. HemoSet→Rabbani: 0.337 il migliore singolo → 0.343 ensemble; Rabbani→HemoSet: 0.615 → 0.623).

### 4.4 Sintesi comparativa

Media sulle tre architetture per ciascuna strategia (dettaglio completo per architettura nei materiali supplementari):

| Strategia | HemoSet (in-domain) | Rabbani (in-domain) | HemoSet→Rabbani (cross) | Rabbani→HemoSet (cross) |
|---|---|---|---|---|
| Baseline (source-only) | 0.745 | 0.684 | 0.217 | 0.585 |
| Augmentation aggressiva | 0.747 | 0.681 | 0.321 | 0.601 |
| Appearance adaptation (Reinhard) | 0.719 | 0.611 | 0.263 | 0.577 |
| Appearance adaptation (FDA) | 0.720 | 0.675 | 0.242 | 0.550 |
| Joint training naturale | 0.742 | 0.636 | n/a* | n/a* |
| Joint training bilanciato | 0.739 | 0.656 | n/a* | n/a* |
| **Ensemble 3 architetture (su aggressiva)** | **0.770** | **0.706** | **0.343** | **0.623** |

\* il joint training è valutato in-domain su entrambi i test set, non esiste una condizione "cross-dataset" comparabile per questa riga.

![Confronto Dice per strategia e condizione di valutazione](chart placeholder — vedi versione HTML/PDF del report)

**Configurazione consigliata: augmentation aggressiva in training + ensemble delle tre architetture in inferenza** — unica combinazione che migliora la robustezza cross-dataset in entrambe le direzioni senza penalizzare le performance in-domain.

---

## 5. Discussione

### 5.1 Perché l'augmentation batte l'adaptation e il joint training

Il fattore comune alle due strategie meno efficaci — appearance adaptation sempre-attiva, joint training sbilanciato — sembra essere la mancanza di un meccanismo che preservi comunque l'esposizione del modello alla distribuzione "pulita" del proprio dominio sorgente durante il training. Un intervento **leggero e probabilistico** sulla varietà visiva (l'augmentation) batte interventi più mirati ma **sempre-attivi** sullo stile o un semplice mescolamento dei dati.

### 5.2 Tre estensioni ulteriori (esito negativo, ma informativo)

Dopo aver individuato la configurazione consigliata, sono state tentate tre estensioni originali aggiuntive, per verificare se fosse possibile migliorarla ulteriormente:

1. **Canale di input "blood-index"** (rapporto R/(R+G+B), meno sensibile in teoria a differenze di camera/illuminazione): migliora leggermente HemoSet→Rabbani (0.321→0.331) ma **peggiora nettamente** Rabbani→HemoSet (0.601→0.526). L'indice, calcolato sui pixel grezzi, si porta dietro le differenze di white-balance tra i dataset invece di ignorarle, diventando una scorciatoia (*shortcut feature*) specifica del dominio sorgente invece che un segnale trasferibile.
2. **MixStyle** (Zhou et al., ICLR 2021): mescola le statistiche (media/varianza) delle feature map dell'encoder tra campioni dello stesso batch, senza richiedere immagini del dominio target. Risultato: peggioramento diffuso ma contenuto su entrambe le direzioni cross-dataset (0.217→0.203; 0.585→0.561), probabilmente perché applicato a livelli troppo bassi dell'encoder (dove si codificano bordi/texture rilevanti per contorni precisi) e perché i batch di training restano mono-dominio in questo setting, limitando la vera diversità di stile iniettata.
3. **Test-time BatchNorm recalibration** (AdaBN, Li et al. 2016): ricalibra le statistiche delle BatchNorm su immagini non annotate del dominio target, a inferenza, senza training aggiuntivo. Sposta il compromesso precision/recall in direzioni opposte nelle due direzioni cross-dataset (HemoSet→Rabbani: precision 0.382→0.283, recall 0.391→0.546; Rabbani→HemoSet: l'esatto contrario) ma il Dice netto non migliora in nessuna delle due (0.321→0.318; 0.601→0.570).

| Estensione | HemoSet→Rabbani (cross) | Rabbani→HemoSet (cross) | Esito |
|---|---|---|---|
| Canale blood-index | 0.321→0.331 | 0.601→0.526 | Negativo (trade-off sfavorevole) |
| MixStyle | 0.217→0.203 | 0.585→0.561 | Negativo (nessun guadagno) |
| Test-time BN recalibration | 0.321→0.318 | 0.601→0.570 | Neutro/negativo |

Nessuna delle tre batte la configurazione consigliata. Il pattern è comunque informativo: tre meccanismi molto diversi (canale di input, feature statistics, normalizzazione a inferenza) falliscono nello stesso modo — o introducendo una scorciatoia specifica del dominio sorgente, o disturbando informazione utile senza sostituirla con qualcosa di più trasferibile — rafforzando la conclusione che l'augmentation aggressiva combinata con l'ensemble non sia un ottimo locale facilmente migliorabile con interventi leggeri, almeno su questa coppia di dataset.

---

## 6. Limiti e lavori futuri

- Appearance/domain adaptation applicata con probabilità <1 (in stile augmentation) invece che sempre attiva, per isolare se il costo in-domain osservato è dovuto proprio a questo.
- Combinare augmentation aggressiva e appearance adaptation, per verificare se gli effetti si sommano.
- Self-training con pseudo-label sul dominio target (setting transduttivo, non ancora tentato in questo lavoro).
- Un ensemble esteso anche ai fold, oltre che alle sole architetture.
- Analizzare la causa della maggiore difficoltà osservata sistematicamente sul fold 3 di HemoSet.
- Quando disponibili, ripetere gli esperimenti più promettenti (augmentation aggressiva + ensemble in primis) sui dati reali del progetto TRAMIS, tenendo conto che mescolare dataset esterni di dimensioni molto diverse nel training non è automaticamente benefico senza un bilanciamento esplicito tra domini.
- Valutare estensioni basate su backbone Transformer pretrained (es. DINOv3, come nel BorDINO proposto da Giusti & Marzo), fuori dallo scope CNN-based di questo lavoro.

---

## 7. Conclusioni

Questo lavoro ha quantificato in modo sistematico il domain gap asimmetrico tra HemoSet e Rabbani, e ha confrontato sei famiglie di strategie per aumentare la robustezza cross-dataset dei modelli di segmentazione del sangue in chirurgia. La **data augmentation aggressiva** è risultata la strategia più efficace e a più basso costo tra quelle testate; la sua combinazione con un **ensemble di tre architetture** (UNet, UNet++, DeepLabV3+) produce il miglior risultato complessivo, senza training aggiuntivo oltre a quello già necessario per l'augmentation. Tre estensioni ulteriori (canale invariante all'illuminazione, MixStyle, test-time BatchNorm recalibration) non hanno migliorato questo risultato, ma la loro analisi supporta l'ipotesi che il fattore chiave sia preservare l'esposizione del modello alla distribuzione "pulita" del proprio dominio durante il training, piuttosto che imporre invarianza tramite segnali o normalizzazioni calcolate direttamente sui dati.

---

## 8. Riferimenti bibliografici

1. A. J. Miao, S. Lin, J. Lu, F. Richter, B. Ostrander, E. K. Funk, R. K. Orosco, M. C. Yip. "HemoSet: The First Blood Segmentation Dataset for Automation of Hemostasis Management." *International Symposium on Medical Robotics (ISMR)*, 2024.
2. N. Rabbani, C. Seve, N. Bourdel, A. Bartoli. "Video-based Computer-aided Laparoscopic Bleeding Management: a Space-time Memory Neural Network with Positional Encoding and Adversarial Domain Adaptation." *Medical Imaging with Deep Learning (MIDL)*, 2022.
3. Y. Yang, S. Soatto. "FDA: Fourier Domain Adaptation for Semantic Segmentation." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.
4. E. Reinhard, M. Ashikhmin, B. Gooch, P. Shirley. "Color Transfer between Images." *IEEE Computer Graphics and Applications*, 2001.
5. K. Zhou, Y. Yang, Y. Qiao, T. Xiang. "Domain Generalization with MixStyle." *International Conference on Learning Representations (ICLR)*, 2021.
6. Y. Li, N. Wang, J. Shi, J. Liu, X. Hou. "Revisiting Batch Normalization For Practical Domain Adaptation." *arXiv:1603.04779*, 2016.
7. Su et al. *AAAI Conference on Artificial Intelligence*, 2023. (domain generalization single-source via augmentation)
8. F. Giusti, F. Marzo. Report di progetto, Computer Vision and Cognitive Systems, UNIMORE, A.A. 2025/2026.
