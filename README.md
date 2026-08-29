# AIGC Detector

A hackathon prototype for detecting AI-generated images, with an emphasis on
robustness to compression, blur, resizing, noise, colour adjustment, and
cropping.

## Label convention

- `0`: authentic/real image
- `1`: AI-generated image

This convention is used by every dataset manifest, model, metric, and JSON
prediction produced by the project.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## CIFAKE smoke-test data

CIFAKE is used only to verify that the end-to-end training and inference
pipeline works. Its images are only 32 x 32 pixels, so results on CIFAKE are
not treated as evidence of real-world robustness.

Download the dataset into `data/raw/cifake`, then create deterministic,
balanced manifests:

```bash
python -m scripts.prepare_cifake
```

This creates:

- `data/processed/train.csv`: 5,000 real and 5,000 AI-generated images
- `data/processed/val.csv`: 1,000 real and 1,000 AI-generated images
- `data/processed/test.csv`: 1,000 real and 1,000 AI-generated images
- `data/processed/dataset_summary.json`: split and label audit information

The validation split is drawn only from CIFAKE's official training split. The
test manifest remains a subset of CIFAKE's official test split.

## Current status

- Section 1A: repository and Python environment setup complete
- Section 1B: CIFAKE download and integrity audit complete
- Section 1C: reproducible manifest preparation complete
