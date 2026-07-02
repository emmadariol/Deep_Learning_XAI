# L'Illusione delle Saliency Maps

## 1. Sintesi del progetto

Il progetto costruisce una pipeline sperimentale in PyTorch per analizzare in modo critico due metodi post-hoc di Explainable AI applicati alla Computer Vision:

- Grad-CAM
- Integrated Gradients

Il caso studio usa il dataset Animals with Attributes 2, abbreviato AwA2, composto da immagini di animali appartenenti a 50 classi. L'obiettivo non e' soltanto classificare correttamente gli animali, ma stressare le spiegazioni prodotte dal modello per verificare se le saliency maps siano realmente legate alla morfologia dell'animale oppure a correlazioni spurie, in particolare allo sfondo.

La tesi sperimentale e':

> se una saliency map cambia drasticamente quando viene perturbato solo lo sfondo, pur mantenendo invariata la predizione del modello, allora la spiegazione non e' stabile rispetto al contenuto semantico principale dell'immagine.

In altre parole, il progetto vuole mostrare quantitativamente e visivamente che alcune spiegazioni post-hoc possono dare una falsa impressione di affidabilita': sembrano indicare l'oggetto corretto, ma possono essere sensibili a segnali non causali.

## 2. Motivazione

Le saliency maps vengono spesso usate per interpretare modelli deep learning in classificazione di immagini. In un caso ideale, se una ResNet predice "zebra", ci aspetteremmo che la mappa di salienza evidenzi:

- corpo dell'animale;
- testa;
- zampe;
- texture del mantello;
- pattern morfologici rilevanti.

Tuttavia, i modelli convoluzionali possono imparare correlazioni spurie:

- erba associata a erbivori;
- neve associata ad animali polari;
- acqua associata a mammiferi marini;
- foresta associata ad alcuni animali selvatici;
- colori e texture dello sfondo associati a una classe.

Questo e' particolarmente importante per AwA2, perche' molte classi animali sono fotografate in contesti naturali ricorrenti. Se il modello usa lo sfondo come scorciatoia, una saliency map puo' diventare una spiegazione esteticamente convincente ma concettualmente fragile.

## 3. Obiettivo tecnico

L'obiettivo tecnico e' costruire una pipeline completa che:

1. prepara il dataset AwA2;
2. addestra una baseline ResNet50 sulle 50 classi;
3. genera Grad-CAM e Integrated Gradients su immagini predette correttamente;
4. perturba selettivamente lo sfondo;
5. ricalcola predizioni e saliency maps;
6. misura quanto le spiegazioni cambiano;
7. produce report CSV e griglie visive per analisi o blog post.

Il punto centrale non e' ottenere la massima accuratezza possibile, ma avere una baseline sufficientemente buona da rendere significative le mappe XAI e lo stress test.

## 4. Struttura del progetto

La struttura prevista e':

```text
Deep_Learning_XAI/
  configs/
  data/
    AWA2/
      JPEGImages/
      awa2_manifest.csv
      class_to_idx.csv
      awa2_manifest_debug.csv
      class_to_idx_debug.csv
  docs/
    PROJECT_EXPLANATION.md
  outputs/
    checkpoints/
    figures/
    reports/
  scripts/
    prepare_awa2.py
    check_dataloader.py
    train_baseline.py
    run_xai.py
    run_stress_test.py
    generate_report.py
  src/
    __init__.py
    data.py
    model.py
    train.py
    xai.py
    perturb.py
    metrics.py
    utils.py
  README.md
  requirements.txt
```

Al momento e' stata implementata la FASE 1, cioe':

- preparazione manifest;
- Dataset PyTorch custom;
- trasformazioni ResNet standard;
- DataLoader;
- smoke test;
- modalita' subset debug.

Le fasi successive verranno implementate solo dopo conferma esplicita, per rispettare la modularita' rigida del progetto.

## 5. Dataset: AwA2

AwA2 contiene immagini JPEG organizzate per classe:

```text
data/AWA2/JPEGImages/
  antelope/
  grizzly+bear/
  killer+whale/
  ...
```

Ogni sottocartella rappresenta una classe. Lo script `scripts/prepare_awa2.py` scansiona queste directory e produce un manifest CSV con le colonne:

```text
filepath,label,class_name,split
```

Esempio concettuale:

```text
/path/to/JPEGImages/zebra/zebra_10001.jpg,49,zebra,train
```

Il mapping classe-indice viene salvato separatamente in:

```text
data/AWA2/class_to_idx.csv
```

Questo rende esplicita e riproducibile la codifica delle 50 classi.

## 6. Strategia subset debug

AwA2 pesa circa 13 GB. Usarlo subito a piena scala rallenterebbe sviluppo, debug e iterazione. Per questo la FASE 1 include una modalita' subset:

```bash
python scripts/prepare_awa2.py \
  --data-root data/AWA2 \
  --max-classes 10 \
  --max-images-per-class 200 \
  --manifest-name awa2_manifest_debug.csv \
  --class-map-name class_to_idx_debug.csv
```

Questa modalita' permette di:

- validare codice rapidamente;
- controllare trasformazioni e DataLoader;
- testare training su poche classi;
- sviluppare Grad-CAM e Integrated Gradients senza costi eccessivi.

La selezione e' deterministica, guidata da seed, quindi gli esperimenti debug sono riproducibili.

## 7. FASE 1: Data Preparation e DataLoader

### 7.1 Script di preparazione

File:

```text
scripts/prepare_awa2.py
```

Responsabilita':

- trovare `JPEGImages/`;
- opzionalmente scaricare AwA2;
- raccogliere immagini per classe;
- generare split train/val/test;
- creare manifest CSV;
- creare mapping classe-indice;
- supportare subset debug.

Split default:

```text
train: 70%
val:   15%
test:  15%
```

Lo split e' fatto dentro ogni classe, cosi' ogni split conserva una distribuzione bilanciata rispetto alle classi selezionate.

### 7.2 Dataset PyTorch

File:

```text
src/data.py
```

Classe principale:

```python
AwA2Dataset
```

Ogni elemento restituisce:

```python
image_tensor, label, class_name, filepath
```

Il `filepath` viene mantenuto perche' sara' utile nelle fasi XAI e stress test, dove bisognera' salvare immagini, heatmaps e report associati al campione originale.

### 7.3 Trasformazioni

Le trasformazioni seguono lo standard usato per ResNet pre-addestrata su ImageNet:

```text
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

Questa scelta e' importante perche' la FASE 2 usera' una ResNet50 pre-addestrata su ImageNet. Usare statistiche diverse introdurrebbe uno shift non necessario.

### 7.4 Smoke test

File:

```text
scripts/check_dataloader.py
```

Controlla:

- numero di immagini per split;
- numero classi;
- shape del batch;
- statistiche dei tensori normalizzati;
- statistiche dei tensori denormalizzati;
- range pre-normalizzazione.

Questo e' coerente con la regola "niente scatole nere": gia' dalla FASE 1 si ispezionano i tensori, non solo il codice.

## 8. FASE 2: Training e fine-tuning baseline

Questa fase verra' implementata dopo conferma.

Obiettivo:

- caricare ResNet50 pre-addestrata su ImageNet;
- sostituire il layer finale con una testa a 50 classi;
- congelare i blocchi iniziali;
- rendere addestrabili gli ultimi blocchi, in particolare `layer3`, `layer4` e `fc`;
- ottimizzare con Cross-Entropy;
- salvare il checkpoint migliore;
- usare Early Stopping.

Architettura prevista:

```text
src/model.py
src/train.py
scripts/train_baseline.py
```

Checkpoint previsto:

```text
outputs/checkpoints/best_resnet50_awa2.pt
```

Metriche minime:

- training loss;
- validation loss;
- training accuracy;
- validation accuracy;
- best epoch;
- early stopping counter.

## 9. FASE 3: Estrazione XAI

Questa fase verra' implementata dopo conferma.

Metodi:

- Grad-CAM;
- Integrated Gradients.

Per Grad-CAM il target layer sara':

```python
model.layer4[-1]
```

Per Integrated Gradients la baseline non sara' un'immagine nera. Useremo invece una baseline sfocata ottenuta applicando Gaussian Blur estremo all'immagine originale.

Motivo:

- una baseline nera introduce un riferimento artificiale;
- puo' alterare drasticamente luminosita' e distribuzione cromatica;
- per immagini naturali, una baseline sfocata preserva colore medio e illuminazione;
- il confronto diventa piu' coerente con il contenuto visivo dell'immagine.

Output:

```text
outputs/figures/xai_examples/
```

Ogni figura dovrebbe confrontare:

- immagine originale;
- Grad-CAM overlay;
- Integrated Gradients overlay;
- classe vera;
- classe predetta;
- confidenza.

## 10. FASE 4: Stress test

Questa fase verra' implementata dopo conferma.

L'obiettivo e' perturbare lo sfondo preservando, per quanto possibile, l'animale.

Approccio principale:

- usare `torchvision.models.detection.maskrcnn_resnet50_fpn`;
- tentare di ottenere una maschera approssimata dell'animale;
- applicare perturbazioni solo ai pixel fuori dalla maschera.

Perturbazioni previste:

1. Gaussian Noise sullo sfondo;
2. Color Shift con inversione canali RGB sullo sfondo;
3. Background Swap con rumore uniforme.

Problema noto:

Mask R-CNN e' addestrata su COCO, che non contiene tutte le classi AwA2. Alcuni animali potrebbero non essere segmentati correttamente.

Fallback:

- se la maschera fallisce, usare perturbazioni globali controllate;
- registrare nel report che il campione ha usato fallback;
- analizzare comunque la stabilita' della saliency rispetto a variazioni non semantiche.

Questa scelta non invalida il progetto: se la mappa cambia molto anche per perturbazioni non semantiche, il punto critico resta valido.

## 11. FASE 5: Metriche quantitative

Questa fase verra' implementata dopo conferma.

Per ogni immagine originale e perturbata:

1. calcolare predizione modello;
2. verificare se la predizione resta uguale;
3. calcolare saliency originale;
4. calcolare saliency perturbata;
5. confrontare le due mappe.

### 11.1 IoU della saliency

Si prendono i top 20% pixel piu' salienti nella mappa originale e nella mappa perturbata. Da ciascuna si ottiene una maschera binaria.

Formula:

```text
IoU = area(intersezione) / area(unione)
```

Interpretazione:

- IoU alta: la spiegazione resta spazialmente simile;
- IoU bassa: la spiegazione si sposta;
- IoU bassa con predizione invariata: possibile instabilita' della spiegazione.

### 11.2 Spearman Rank Correlation

Si appiattiscono i tensori di salienza e si calcola la correlazione di rango di Spearman.

Interpretazione:

- correlazione alta: l'ordine di importanza dei pixel resta simile;
- correlazione bassa o negativa: la gerarchia di importanza cambia;
- correlazione bassa con predizione invariata: la spiegazione non e' robusta.

## 12. Report finale

Output previsto:

```text
outputs/reports/stress_test_results.csv
```

Colonne previste:

```text
image_id
filepath
class_name
true_label
pred_original
pred_perturbed
confidence_original
confidence_perturbed
prediction_preserved
perturbation_type
mask_status
xai_method
saliency_iou_top20
spearman_correlation
notes
```

Figure previste:

```text
outputs/figures/stress_test_grids/
```

Ogni griglia dovrebbe mostrare:

- originale;
- immagine perturbata;
- Grad-CAM originale;
- Grad-CAM perturbata;
- Integrated Gradients originale;
- Integrated Gradients perturbata.

## 13. Criterio di successo

Il progetto ha successo se produce evidenza che:

- il modello mantiene spesso la stessa predizione dopo perturbazione dello sfondo;
- le mappe XAI cambiano sensibilmente;
- l'IoU tra saliency originali e perturbate diminuisce;
- la correlazione di Spearman tra saliency originali e perturbate diminuisce;
- le visualizzazioni mostrano spostamenti della salienza verso regioni non semantiche o instabili.

Non serve dimostrare che Grad-CAM o Integrated Gradients siano sempre inutili. L'obiettivo e' piu' preciso: mostrare che, in questo setting, possono essere fragili e fuorvianti se interpretati come spiegazioni causali.

## 14. Rischi sperimentali

### 14.1 Accuracy bassa

Se la baseline non impara abbastanza bene, le mappe XAI saranno poco informative. Soluzione:

- usare piu' immagini;
- aumentare epoche;
- sbloccare piu' layer;
- controllare learning rate;
- verificare class mapping e normalizzazione.

### 14.2 Segmentazione fallita

Mask R-CNN potrebbe non segmentare molti animali AwA2. Soluzione:

- usare fallback globale;
- salvare `mask_status`;
- non nascondere il fallimento, ma includerlo nell'analisi critica.

### 14.3 Integrated Gradients costoso

IG richiede molti forward pass. Soluzione:

- usare GPU;
- limitare numero immagini;
- ridurre temporaneamente `n_steps` in debug;
- usare batch piccoli;
- calcolare IG solo su immagini predette correttamente.

### 14.4 Interpretazione eccessiva delle saliency

Le saliency maps non sono automaticamente spiegazioni causali. Il progetto deve evitare claim troppo forti. La formulazione corretta e':

> le saliency maps osservate sono instabili sotto perturbazioni controllate, quindi non dovrebbero essere interpretate ingenuamente come prova che il modello stia usando la morfologia dell'animale.

## 15. Comandi operativi FASE 1

Preparazione manifest completo:

```bash
python scripts/prepare_awa2.py --data-root data/AWA2
```

Preparazione manifest debug:

```bash
python scripts/prepare_awa2.py \
  --data-root data/AWA2 \
  --max-classes 10 \
  --max-images-per-class 200 \
  --manifest-name awa2_manifest_debug.csv \
  --class-map-name class_to_idx_debug.csv
```

Smoke test manifest completo:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest.csv
```

Smoke test manifest debug:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest_debug.csv
```

## 16. Stato attuale

Implementato:

- setup directory;
- FASE 1;
- subset debug;
- documentazione iniziale.

Non ancora implementato:

- FASE 2 training baseline;
- FASE 3 XAI;
- FASE 4 stress test;
- FASE 5 metriche e report.

La prossima fase, dopo conferma, sara' la FASE 2.

