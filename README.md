# L'Illusione delle Saliency Maps

Stress test su Grad-CAM e Integrated Gradients con AwA2.

## Setup directory

```text
Deep_Learning_XAI/
  configs/
  data/
    AWA2/
      JPEGImages/
  outputs/
    checkpoints/
    figures/
    reports/
  scripts/
  src/
```

AwA2 richiede circa 13 GB. Puoi copiare manualmente `JPEGImages/` in
`data/AWA2/JPEGImages/`, oppure usare lo script con `--download`.

## FASE 1

Preparazione manifest:

```bash
python scripts/prepare_awa2.py --data-root data/AWA2
```

Download opzionale:

```bash
python scripts/prepare_awa2.py --data-root data/AWA2 --download
```

Smoke test DataLoader:

```bash
python scripts/check_dataloader.py --manifest data/AWA2/awa2_manifest.csv
```

