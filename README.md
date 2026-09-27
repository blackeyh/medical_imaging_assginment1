# ENG 2440 Assignment 1: Pneumonia Classification Challenge

End-to-end PyTorch pipeline that classifies frontal chest radiographs from the RSNA Pneumonia Detection Challenge as lung opacity consistent with possible pneumonia (positive) or no such opacity (negative). This is a learning exercise, not a clinical diagnostic system.

The pipeline covers data inspection, patient-grouped (leakage-safe) splits, preprocessing and augmentation, a trivial baseline, a logistic-regression baseline on image-summary features, an ImageNet-pretrained ResNet-18 (feature extraction then fine-tuning, three seeds), evaluation (ROC/PR curves, operating thresholds, calibration), subgroup analysis by view, and Grad-CAM.

## Repository contents

| File | Purpose |
| --- | --- |
| `25011143_ENG2440_A1.ipynb` | Fully executed notebook with all code and outputs |
| `requirements.txt` | Python packages used |
| `ai_use_declaration.txt` | Declaration of AI-tool use |
| `README.md` | This file |

**No data is included.** The RSNA/NIH images, the label and mapping CSV files, cached image arrays, model weights and figures are excluded (see `.gitignore`), in line with the assignment rules and the dataset terms.

## Data setup

1. Download the image archive from the official RSNA Pneumonia Detection Challenge (2018) page and the assignment supporting files from the course LMS.
2. Arrange the folder as follows:

```
ENG2440_Assignment1/
|-- data/
|   |-- images/            # extracted DICOM folders and files
|-- assignment1_labels.csv
|-- rsna_to_nih_mapping.csv
|-- 25011143_ENG2440_A1.ipynb
```

3. In the first configuration cell, set `DATA_ROOT` if your image folder is elsewhere.

## Environment

Python 3 with the packages in `requirements.txt`. The main ones are `torch`, `torchvision`, `pydicom`, `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `captum`.

Note: `requirements.txt` was exported from a Linux machine with an NVIDIA GPU (CUDA 12.8 builds of `torch` and `torchvision`, plus `nvidia-*` packages). On macOS or a CPU-only machine, install `torch` and `torchvision` for your platform from pytorch.org and skip the `nvidia-*`, `cuda-*` and `triton` lines. The notebook picks `cuda`, `mps` or `cpu` automatically.

## Running

Run the notebook top to bottom. The notebook:

1. loads the labels and mapping, collapses them to one label per examination and joins the NIH patient IDs;
2. creates patient-grouped train/validation/test splits (`GroupShuffleSplit`, seed 42) and checks they are disjoint;
3. reads DICOM headers and pixels, extracts baseline features and caches resized 224 x 224 images to `data/`;
4. trains the baselines and the ResNet-18 (two stages, seeds 42, 43, 44), saving weights, histories and predicted probabilities to `outputs/`;
5. evaluates and interprets the models from the saved outputs.

Slow steps (feature extraction, image caching, CNN training) write their results to `data/` and `outputs/`; later cells load these files.

## Reproducibility

Random seeds are set for Python, NumPy, PyTorch and the training DataLoader. GPU backends may not be fully deterministic, so re-runs can differ very slightly.

## Data acknowledgement

Images and annotations come from the RSNA Pneumonia Detection Challenge (2018) and the NIH ChestX-ray dataset. The NIH Clinical Center is acknowledged as the provider of the source chest-radiograph data.
