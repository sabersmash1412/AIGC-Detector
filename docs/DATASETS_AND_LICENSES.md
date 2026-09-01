# Datasets, Models, Licenses, and Citations

This project is a non-commercial hackathon research prototype. This document
records the external assets used, how they were used, and the obligations or
limitations identified during the submission audit. It is not legal advice.

## Redistribution boundary

The repository does **not** contain CIFAKE, SID-Set, or AIGIBench image bytes.
Raw images, CLIP feature caches, prediction exports, and downloaded OpenCLIP
weights remain ignored by Git. The repository does contain deterministic
selection manifests with repository-relative paths and dataset-derived
filenames/identifiers, provenance hashes, aggregate metrics, non-image plots,
and the small trained E5 linear head.

## Asset summary

| Asset | Use in this project | Declared license | Repository redistribution |
| --- | --- | --- | --- |
| CIFAKE | Smoke test, E1–E5 training/validation, internal test | MIT (as declared by the official Kaggle dataset page) | No images; manifests and aggregate results only |
| SID-Set | E4/E5 development and held-out FLUX audit | CC-BY-4.0 | No images; selection manifests, provenance and aggregate results only |
| AIGIBench Midjourney V6 | Frozen E5 external test only | CC-BY-NC-SA-4.0 | No images; selection manifest, provenance and aggregate results only |
| OpenCLIP | Feature-extraction implementation | MIT | Dependency only |
| OpenAI CLIP | `ViT-B-32-quickgelu`, `openai` pretrained weights | MIT software release | Downloaded weights are not committed |

## CIFAKE

- Official dataset: [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Paper: Bird, J. J. and Lotfi, A. (2024), *CIFAKE: Image Classification and
  Explainable Identification of AI-Generated Synthetic Images*, IEEE Access,
  DOI [10.1109/ACCESS.2024.3356122](https://doi.org/10.1109/ACCESS.2024.3356122).
- Real-image source citation: Krizhevsky, A. and Hinton, G. (2009), *Learning
  Multiple Layers of Features from Tiny Images* (CIFAR-10).
- Generated-image source: Stable Diffusion v1.4 images released as part of
  CIFAKE.
- License record: the official Kaggle page declares the dataset under the same
  MIT license as CIFAR-10 and requires citation of both CIFAKE and CIFAR-10.

CIFAKE's 32 × 32 resolution and single Stable Diffusion family make it useful
for pipeline verification, but insufficient evidence of modern, real-world
generator generalisation.

## SID-Set and SIDA

- Dataset: [saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)
- Project repository: [hzlsaber/SIDA](https://github.com/hzlsaber/SIDA)
- Paper: Huang, Z. et al. (2025), *SIDA: Social Media Image Deepfake Detection,
  Localization and Explanation with Large Multimodal Model*, CVPR 2025.
- License record: CC-BY-4.0.

The current SID-Set card describes real images as Open Images V7 and notes
incorporated material from COCO and Flickr30k. The project protocol also
records a source-description inconsistency between the dataset card and paper.
We therefore cite SID-Set/SIDA and the upstream sources, retain the exact
selection provenance, and make no claim that these source domains are
equivalent. Open Images images carry per-image attribution requirements; no
SID-Set image bytes are redistributed here.

## AIGIBench

- Dataset: [HorizonTEL/AIGIBench](https://huggingface.co/datasets/HorizonTEL/AIGIBench)
- Paper: Li, Z. et al. (2025), *Is Artificial Intelligence Generated Image
  Detection a Solved Problem?*, NeurIPS 2025,
  [arXiv:2505.12335](https://arxiv.org/abs/2505.12335).
- License record: CC-BY-NC-SA-4.0.

Only the frozen, deduplicated 1,000-real/1,000-Midjourney-V6 external subset
was used. It was never used for training, model selection, or threshold
selection. Because its license is non-commercial and share-alike, commercial
reuse requires a separate review. The repository publishes no AIGIBench image
bytes.

## CLIP and OpenCLIP

- Radford, A. et al. (2021), *Learning Transferable Visual Models From Natural
  Language Supervision*, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020).
- OpenAI implementation and license: [openai/CLIP](https://github.com/openai/CLIP)
- OpenCLIP implementation and license:
  [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

Both software repositories use the MIT license. This project uses OpenCLIP to
load the original OpenAI `ViT-B-32-quickgelu` weights. The encoder remains
frozen; the committed checkpoint contains the project-specific 512-weight
linear head, one bias term, and audit metadata. Downloaded encoder weights are
not committed.

## Organiser validation isolation

The organiser-provided validation subset, including the forbidden COCO
validation and DALL·E Advanced assets identified in the frozen protocols, was
never used for training, threshold selection, model selection, debugging, or
the external evaluation. This is enforced by checked-in protocol locks, tests,
and `python -m scripts.audit_submission`.

## Project code license

Original project code is released under the repository's MIT `LICENSE` file.
Third-party datasets, dependencies and pretrained assets remain governed by
their respective licenses described above; the project MIT license does not
replace or override those terms.
