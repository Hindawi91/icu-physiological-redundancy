# ICU Physiological Redundancy

> **Characterizing Physiological Information Redundancy Among ICU Vital Signs Through Deep Learning-Based Cross-Modal Reconstruction**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: PhysioNet 2019](https://img.shields.io/badge/Data-PhysioNet%202019-lightblue.svg)](https://physionet.org/content/challenge-2019/1.0.0/)

> 📄 **Paper coming soon**

---

## Graphical Abstract

<!-- To add your graphical abstract, place the image in the repo and replace the line above with:
![Graphical Abstract](figures/graphical_abstract.png) -->

<img width="1918" height="1004" alt="graph_abstract" src="https://github.com/user-attachments/assets/83f52256-f283-490c-bf63-742debb254cf" />


---

Official code for:

> **Characterizing Physiological Information Redundancy Among ICU Vital Signs
> Through Deep Learning-Based Cross-Modal Reconstruction**
> *Firas Al-Hindawi*

---

## Overview

This repository investigates **physiological information redundancy** in ICU
vital sign monitoring — the extent to which one physiological modality can be
reconstructed from the remaining monitored signals while preserving downstream
clinical information.

Rather than treating reconstruction as an engineering goal, we use it as a
**probe**: reconstruction fidelity (Pearson *r*) quantifies how much information
about each vital sign is encoded in the others. We evaluate six deep learning
architectures across 15 reconstruction configurations and assess whether
reconstructed signals preserve sepsis classification performance.

**Key findings:**
- Blood pressure modalities (MAP, SBP, DBP) are **highly redundant** with
  each other (*r* = 0.86–0.94 with one additional BP sensor)
- BP **cannot** be inferred from non-invasive sensors alone (*r* ≈ 0.23)
- Respiratory rate and SpO₂ are **physiologically independent** from
  haemodynamic vitals (*r* ≤ 0.30)
- MAP reconstruction with *r* = 0.86 recovers **95% of lost sepsis
  classification performance**

---

## Repository Structure

```
icu-physiological-redundancy/
│
├── models.py                        # Reconstruction architectures (TCN, BiLSTM,
│                                    #   LSTM S2S, U-Net 1D, Transformer, Conv-LSTM)
├── classification_models.py         # Sepsis classifier architectures
├── train.py                         # Train reconstruction models
├── test.py                          # Evaluate reconstruction models
├── train_sepsis_classifier.py       # Train sepsis classifier for clinical utility
├── evaluate_clinical_utility.py     # Clinical utility evaluation
├── preprocess.py                    # Data preprocessing pipeline
│
├── scripts/                         # SLURM job submission scripts (HPC)
│   └── submit_all_experiments.sh    # Submit all 15 reconstruction experiments
│
├── analysis/                        # Results aggregation and plotting
│   ├── combine_results.py           # Aggregate per-experiment CSVs → ALL_metrics.csv
│   ├── plot_all_experiments.py      # Waveform figure (Appendix in paper)
│   └── plot_reconstruction_results.py  # Summary figures (Figs. 2–3 in paper)
│
├── notebooks/                       # Exploratory data analysis
│   ├── 1_Prepare_datav3.ipynb       # Data loading, missingness analysis, EDA
│   └── 2_combine_and_split_datasets.ipynb  # Dataset combination and splitting
│
├── results/                         # Pre-computed results
│   └── ALL_metrics.csv              # All 15 experiments × 6 models
│
├── environment.yml                  # Conda environment
├── requirements.txt                 # pip dependencies
└── README.md
```

---

## Experiments

### Reconstruction Experiments (15 configurations)

| Group | Target(s) | Input set | Configs |
|-------|-----------|-----------|---------|
| G1 | MAP, SBP, DBP | Non-invasive only (HR, O₂Sat, Resp, Temp) | 3 |
| G2 | MAP, SBP, DBP | Non-invasive + 1 or 2 BP vitals | 9 |
| G3 | Temp | All remaining 6 vitals | 1 |
| G4 | Resp, O₂Sat | All remaining 6 vitals | 2 |

### Clinical Utility Experiment

A frozen TCN sepsis classifier is evaluated under 5 MAP conditions:
full supervision, MAP removed (mean imputed), and MAP reconstructed at
3 fidelity levels (G1: *r*=0.225, G2+: *r*=0.862, G2++: *r*=0.936).

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Hindawi91/icu-physiological-redundancy.git
cd icu-physiological-redundancy
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate icu-vital
```

Or with pip:

```bash
pip install -r requirements.txt
```

### 3. Download and preprocess the data

This study uses the
[PhysioNet Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
dataset (free, requires registration).

> **Note:** The preprocessed data splits used in this study cannot be
> redistributed due to the PhysioNet data use agreement. Run the
> preprocessing script below with the raw data to reproduce the exact
> splits (seed = 42).

```bash
# After downloading the PhysioNet 2019 data:
python preprocess.py \
    --data_dir /path/to/physionet2019/training/ \
    --out_dir  checkpoints/splits/
```

This produces `checkpoints/splits/train.csv`, `val.csv`, and `test.csv`
with the same patient-stratified splits used in the paper.

---

## Usage

### Train a reconstruction model

```bash
# Example: TCN reconstructing MAP from NI + DBP (Group 2, r=0.862)
python train.py \
    --model tcn \
    --inputs HR,O2Sat,Resp,Temp,DBP \
    --target MAP \
    --epochs 200 \
    --seed 42
```

### Evaluate reconstruction

```bash
python test.py \
    --model tcn \
    --run_tag HR-O2Sat-Resp-Temp-DBP_to_MAP
```

Results are saved to `results/metrics_<run_tag>.csv`.

### Reproduce all 15 experiments (SLURM cluster)

```bash
chmod +x scripts/submit_all_experiments.sh
./scripts/submit_all_experiments.sh

# Aggregate results
python analysis/combine_results.py \
    --results_dir results/ \
    --out results/ALL_metrics.csv
```

### Train the sepsis classifier

```bash
python train_sepsis_classifier.py \
    --pos_weight 5 \
    --epochs 200
```

### Clinical utility evaluation

```bash
python evaluate_clinical_utility.py --pos_weight 5
```

### Reproduce paper figures

```bash
# Figure 2: Progressive sensor availability
# Figure 3: Clinical utility of MAP reconstruction
python analysis/plot_reconstruction_results.py \
    --metrics results/ALL_metrics.csv \
    --outdir  figures/

# Appendix figures: Waveform grid
python analysis/plot_all_experiments.py --seeds 42
```

---

## Pre-computed Results

`results/ALL_metrics.csv` contains reconstruction metrics for all
15 experimental configurations × 6 deep learning models × 5 baselines,
allowing reproduction of all tables and figures without retraining.

| Column | Description |
|--------|-------------|
| `Experiment` | Input→target configuration (e.g. `MAP ← HR,O2Sat,Resp,Temp`) |
| `Group` | Experimental group (1–4) |
| `Target` | Reconstructed vital sign |
| `Inputs` | Input vital signs |
| `RMSE` | Root mean squared error |
| `MAE` | Mean absolute error |
| `MAPE` | Mean absolute percentage error (%) |
| `Pearson_r` | Pearson correlation coefficient |

---

## Model Architecture Summary

| Model | Family | Key configuration |
|-------|--------|-------------------|
| TCN | Convolutional | Channels (32,32,64,64), dilations (1,2,4,8), kernel 5 |
| BiLSTM | Recurrent | Hidden=64, 2 layers, bidirectional |
| LSTM S2S | Recurrent | Hidden=64, 2 layers, bidirectional encoder |
| U-Net 1D | Convolutional | Depth=3, base channels=32, skip connections |
| Transformer | Attention | d_model=64, 4 heads, 3 encoder layers |
| Conv-LSTM | Hybrid | CNN 32ch + BiLSTM hidden=64 |

All models receive input `(B, C_in, T)` and output `(B, C_out, T)` where
`T = 12` (12-hour window, 4-hour step) and `C_out = 1`.

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{hindawi2025icu,
  title   = {Characterizing Physiological Information Redundancy Among
             {ICU} Vital Signs Through Deep Learning-Based
             Cross-Modal Reconstruction},
  author  = {Al-Hindawi, Firas},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The PhysioNet 2019 dataset is subject to its own
[data use agreement](https://physionet.org/content/challenge-2019/1.0.0/).

---

## Contact

Firas Al-Hindawi — King Fahd University of Petroleum and Minerals
📧 firas.hindawi@kfupm.edu.sa
🔗 [GitHub](https://github.com/Hindawi91)


---
