# Comparative Analysis of Deep Learning Architectures for CIFAR-10

A reproducible image-classification study comparing four neural-network architecture families—**MLP, residual CNN, row-wise LSTM, and Vision Transformer (ViT)**—on the CIFAR-10 dataset.

> **Project status:** The notebook contains the implementation and experiment workflow. Do not publish performance claims until the full tuning and held-out test-evaluation runs have completed and their generated results have been reviewed.

## Overview

This project investigates how different architectural choices affect image classification. All four models use the same CIFAR-10 data protocol, and model selection is based on validation macro-F1.

### Architectures

| Model | Implementation |
|---|---|
| MLP | Flattened 3,072-pixel input followed by dense layers (1,024, 512, 256), BatchNorm, ReLU and dropout |
| CNN | Convolutional stem, residual blocks at 64/128/256 channels, global average pooling and a classification head |
| LSTM | Treats each 32-pixel image row as a sequence of 96 features; uses a two-layer LSTM |
| ViT | 4×4 patch embedding, class token, positional embeddings and a six-layer Transformer encoder |

Each network produces logits for the 10 CIFAR-10 classes.

## Dataset

**CIFAR-10** contains 60,000 colour images of size 32×32 across 10 classes:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

- 50,000 official training images
- 10,000 official test images
- The notebook creates a stratified split of the 50,000 training images into 45,000 training and 5,000 validation images (seed 42).
- The official 10,000-image test split is kept separate from hyperparameter tuning and is used only for final evaluation.

### Download the dataset

**You do not need to manually upload CIFAR-10 to GitHub or Google Colab.** The notebook uses `torchvision.datasets.CIFAR10(..., download=True)`, which downloads and extracts the dataset when it is first needed.

Official dataset page: https://www.cs.toronto.edu/~kriz/cifar.html

The dataset archive is approximately 170 MB. Keep it out of the Git repository; the notebook downloads it locally/into the Colab runtime. The notebook's Google Drive workflow stores run outputs under `MyDrive/CIFAR10_Final/`.

## Experimental protocol

- **Preprocessing:** convert images to tensors and normalize using channel means `(0.4914, 0.4822, 0.4465)` and standard deviations `(0.2470, 0.2435, 0.2616)`.
- **Training augmentation:** random crop (32×32, padding 4) and random horizontal flip.
- **Batch size:** 128 by default.
- **Split seed:** 42.
- **Tuning:** three learning-rate trials per architecture, using 0.3×, 1× and 3× the architecture's base learning rate.
- **Selection metric:** validation macro-F1.
- **Optimization:** CNN uses SGD with momentum; MLP, LSTM and ViT use AdamW. The implementation uses weight decay, gradient clipping and learning-rate scheduling.
- **Final evaluation:** selected checkpoints are evaluated on the untouched official test split after tuning is complete.

Base learning rates in the implementation:

| Architecture | Base learning rate |
|---|---:|
| MLP | 0.001 |
| CNN | 0.1 |
| LSTM | 0.001 |
| ViT | 0.0003 |

## Repository contents

```text
.
├── CIFAR10_All_In_One_Colab.ipynb   # Complete Google Colab notebook
├── README.md                        # Project documentation
└── .gitignore                       # Recommended generated-file exclusions
```

The notebook writes a standalone Python implementation (`CIFAR10_All_In_One.py`) during execution and runs the workflow from Colab.

## Getting started

### Option 1: Google Colab (recommended)

1. Upload `CIFAR10_All_In_One_Colab.ipynb` to Google Colab or open it from your GitHub repository.
2. In Colab, select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the notebook cells from top to bottom.
4. Connect Google Drive when prompted. Outputs are saved in a timestamped folder under `MyDrive/CIFAR10_Final/`.
5. The notebook runs a synthetic smoke test, trains/tunes the models, evaluates the selected checkpoints on the official test split, and generates a PDF report.

The notebook's default tuning budget is 20 epochs per trial and three trials per model. Running all four models therefore involves 12 training trials and can take a substantial amount of time.

### Option 2: Run the generated Python script locally

The notebook generates `CIFAR10_All_In_One.py`. After generating that file, install the dependencies:

```bash
python -m pip install torch torchvision numpy scikit-learn matplotlib reportlab
```

Run a synthetic model smoke test:

```bash
python CIFAR10_All_In_One.py smoke
```

Train/tune all four architectures:

```bash
python CIFAR10_All_In_One.py tune --model all --epochs 20 --out runs/tuning
```

Run final evaluation and generate the report:

```bash
python CIFAR10_All_In_One.py finalize --tuning runs/tuning --out runs/final
```

Use a **new output directory** for each run. The implementation intentionally refuses to overwrite existing experiment outputs.

## Outputs

Depending on the completed workflow, generated artifacts include:

- Model checkpoints (`best.pt`, `last.pt`)
- Training history (`history.csv`)
- Experiment configuration and dataset audit (`config.json`, `dataset_audit.json`)
- Selected validation results (`selected.json`, `validation_summary.csv`)
- Test comparison (`test_comparison.csv`, `test_comparison.json`)
- Confusion matrices and learning curves
- High-confidence error examples and prediction-level outputs
- `Final_Report.pdf`

Generated outputs are not included in this repository by default. Add selected plots, tables or report excerpts only after checking that they were produced by a completed run.

## Evaluation metrics

The implementation reports:

- Accuracy
- Macro precision
- Macro recall
- Macro-F1
- Confusion matrix
- Number of model parameters
- Training duration

Macro-averaged metrics give each class equal weight. Validation metrics are used for model selection; final test metrics are reserved for reporting the selected models' held-out performance.

## Reproducibility and limitations

- The training/validation split is stratified and saved with a SHA-256 audit hash.
- Random seeds are set for Python, NumPy and PyTorch.
- Model configurations, checkpoints and training histories are saved.
- The smoke test uses **synthetic random inputs**. It checks model execution and gradients; it is not a CIFAR-10 accuracy test.
- A single seed does not measure variability across independent runs.
- Equal epoch counts do not mean equal computational cost across architectures.
- Results depend on the hardware, software versions, hyperparameters and training budget. Report measured results from your own completed run rather than inserting assumed benchmark values.

## Team

Project contributors (as listed in the notebook):

- Harsh Kushwaha
- Sanket Tippimani
- Sanyam Satija
- Devansh Sharma

## References

1. Alex Krizhevsky. *Learning Multiple Layers of Features from Tiny Images*. 2009. [CIFAR-10 dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).
2. [PyTorch](https://pytorch.org/)
3. [Torchvision CIFAR-10 dataset documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)

## License

Add a repository license after confirming the licensing and redistribution requirements for your source code and any included materials. The CIFAR-10 dataset has its own terms and acknowledgement guidance; refer to the [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).
