# Vision-Model-Bias-Eval

Final project for COMP-5230 (Spring 2026) by Matthew Shoup and Dan Nguyen.

## Overview

This project investigates bias in vision models on a gender identification task using the FairFace dataset. The main question we aim to explore is how well a model trained on one race transfers to other race groups, and whether performance gaps show up in out-of-distribution evaluation.

We compare three model families:

- ResNet-50
- VGG16
- Vision Transformer (ViT)

Each experiment trains a binary gender classifier on a single race group, evaluates on the matching in-distribution split plus all remaining race groups, and then records the resulting accuracy gaps, confusion matrices, and Grad-CAM visualizations.

## Project outputs

The main outputs are:

- Per-race bias reports with in-distribution accuracy, mean OOD accuracy, worst OOD accuracy, and gaps between metrics
- Confusion matrix summaries and plots for each evaluation group
- Grad-CAM feature importance visualizations to inspect which image regions the model uses for predictions across race groups

Example outputs can be found in the `results/` directory, in model-specific folders such as `results/resnet/`, `results/vgg16/`, and `results/vit/`.

## Main project files

- `experiments/compare_race_resnet.py` trains and evaluates a ResNet-50 on one race at a time
- `experiments/compare_race_vgg16.py` does the same for VGG16
- `experiments/compare_race_vit.py` does the same for ViT
- `experiments/run_compare_race_resnet.py` runs the ResNet experiment once per race and aggregates the results
- `experiments/run_compare_race_vgg16.py` runs the VGG16 experiment once per race and aggregates the results
- `experiments/run_compare_race_vit.py` runs the ViT experiment once per race and aggregates the results
- `experiments/confusion_matrix_utils.py` builds confusion-matrix metrics, plots, and tables
- `experiments/visualize_features.py` generates Grad-CAM visualizations for a trained model

## Setup

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

## Running the Experiments

The default dataset is `HuggingFaceM4/FairFace` with config `1.25`.

Train a single model (Resnet, VGG16, ViT) on a single race in FairFace:

```bash
python experiments/compare_race_resnet.py --train-race White --pretrained
python experiments/compare_race_vgg16.py --train-race White --pretrained
python experiments/compare_race_vit.py --train-race White --pretrained
```

Train a single model (Resnet, VGG16, ViT) on the entire FairFace dataset:

```bash
python experiments/run_compare_race_resnet.py --pretrained
python experiments/run_compare_race_vgg16.py --pretrained
python experiments/run_compare_race_vit.py --pretrained
```

The available races to train on are:

- White
- Black
- Indian
- Middle Eastern
- Latino_Hispanic
- East Asian
- Southeast Asian

## Grad-CAM Visualizations

To generate Grad-CAM overlays for a trained checkpoint. For example, generating visualizations for ViT:

```bash
python experiments/visualize_features.py --model-type vit --checkpoint results/vit/vit_best.pt
```

You can also use `--model-type resnet50` or `--model-type vgg16`. Pass `--checkpoint` explicitly if your checkpoint is stored elsewhere.

## Output Structure

The race-specific training scripts create a folder named like `gender_from_<Race>/` inside the chosen output directory. Each folder contains:

- `gender_bias_report.json`
- `confusion_matrices.json`
- `confusion_matrices_summary.txt`
- `confusion_matrices_comparison.png`
- per-group confusion matrix plots

The sweep scripts also write:

- `race_comparison_summary.json`
- `race_comparison_summary.txt`

Grad-CAM runs create race folders such as `race_0_White/` and saves sample visualizations there.
