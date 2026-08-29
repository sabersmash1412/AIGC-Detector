# AIGC Detector

A hackathon prototype for detecting AI-generated images, with an emphasis on
robustness to compression, blur, resizing, noise, colour adjustment, and
cropping.

## Documentation

- [Section 1 technical report](docs/section-1-technical-report.md): detailed
  environment, dataset, split, model, training, evaluation, testing, and
  limitation notes for the CIFAKE smoke-test pipeline.

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

## Directory inference

The inference command accepts an image directory and writes one AIGC
probability per readable image:

```bash
python -m src.predict \
  --input-dir example_images \
  --checkpoint checkpoints/cifake_cnn.pt \
  --output outputs/predictions.json
```

The output is a JSON array with the required `image_path` and `pred` fields.
`pred` is in the range `[0, 1]`, where a higher value means more likely
AI-generated. Section 1E produces the local smoke-test checkpoint used by this
command.

## CIFAKE smoke-test training and evaluation

Train the compact CNN with seed `42` and save the best validation ROC-AUC
checkpoint:

```bash
python -m src.train
```

Evaluate that checkpoint on the untouched CIFAKE test manifest:

```bash
python -m src.evaluate
```

The resulting metrics demonstrate that the pipeline works end to end. They do
not establish robustness or generalisation to modern generators.

### Smoke-test result

The best five-epoch checkpoint was selected using validation ROC-AUC and then
evaluated once on the 2,000-image test manifest using the default threshold of
`0.5`.

| Metric | Result |
| --- | ---: |
| ROC-AUC | 0.9570 |
| Average precision | 0.9558 |
| Balanced accuracy | 0.8940 |
| Precision | 0.8752 |
| Recall | 0.9190 |
| F1 | 0.8966 |

These numbers are only a pipeline check on low-resolution CIFAKE data. The
research baseline begins with frozen CLIP features in Section 2.

## Frozen CLIP environment check

Section 2 uses OpenCLIP's `ViT-B-32-quickgelu` model with the original
`openai` pretrained weights. The QuickGELU architecture matches the activation
used to train those weights. CLIP is frozen and supplies one L2-normalized
512-dimensional image embedding; a separate linear classifier is trained in a
later step.

Run the two-image environment and feature sanity check with:

```bash
python -m scripts.check_clip --device auto
```

The command downloads the pretrained weights once into the ignored
`checkpoints/open_clip` cache and writes diagnostics to
`reports/clip_environment_check.json`. Passing this check proves that the
pretrained feature extractor loads and produces valid features. It does not
train or evaluate a fake-image classifier.

## Current status

- Section 1A: repository and Python environment setup complete
- Section 1B: CIFAKE download and integrity audit complete
- Section 1C: reproducible manifest preparation complete
- Section 1D: image-directory JSON inference implemented
- Section 1E: CIFAKE smoke-test training and evaluation complete
- Section 2A: frozen CLIP dependency and feature sanity check implemented
