# Physiological Information Redundancy in ICU Vital Signs

> **Systematic Investigation of Cross-Modal Reconstruction and Clinical Utility**

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: PhysioNet 2019](https://img.shields.io/badge/Data-PhysioNet%202019-lightblue.svg)](https://physionet.org/content/challenge-2019/1.0.0/)

Official code for the paper:

> **Physiological Information Redundancy in ICU Vital Signs: A Systematic
> Investigation of Cross-Modal Reconstruction and Clinical Utility**
> *Firas Hindawi — IEEE Journal of Biomedical and Health Informatics (under review)*

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
icu-vital-redundancy/
│
├── models.py                        # Reconstruction architectures (TCN, BiLSTM,
│                                    #   LSTM S2S, U-Net 1D, Transformer, Conv-LSTM)
├── classification_models.py         # Sepsis classifier architectures
├── train.py                         # Train reconstruction models
├── test.py                          # Evaluate reconstruction models
├── train_sepsis_classifier_v2.py    # Train sepsis classifiers (FS/IS/RAS conditions)
├── evaluate_clinical_utility.py     # Clinical utility evaluation (Study 1)
├── evaluate_clinical_utility_v2.py  # Clinical utility evaluation (Study 2)
│
├── scripts/                         # SLURM job submission scripts (HPC)
│   ├── submit_all_experiments.sh    # Submit all 15 reconstruction experiments
│   ├── submit_classifier_v2.sh      # Submit IS/RAS classifier jobs
│   └── submit_oracle_search.sh      # Submit oracle architecture search
│
├── analysis/                        # Results aggregation and plotting
│   ├── combine_results.py           # Aggregate per-experiment CSVs → ALL_metrics.csv
│   ├── plot_all_experiments.py      # Waveform figure (Fig. A1–A3 in paper)
│   └── plot_reconstruction_results.py  # Summary figures (Fig. 2–3 in paper)
│
├── results/                         # Pre-computed results (CSV)
│   ├── ALL_metrics.csv              # All 15 experiments × 6 models (window-level)
│   ├── ALL_metrics_per_patient.csv  # Patient-level aggregation
│   └── ALL_metrics_subgroup.csv     # Sepsis vs non-sepsis subgroup analysis
│
├── environment.yml                  # Conda environment
├── requirements.txt                 # pip dependencies
└── README.md
```

> **Note:** Model checkpoints (`checkpoints/models/*.pt`) and patient data
> splits (`checkpoints/splits/`) are **not included** in this repository.
> See [Data & Checkpoints](#data--checkpoints) below.

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
git clone https://github.com/<your-username>/icu-vital-redundancy.git
cd icu-vital-redundancy
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

### 3. Download the data

This study uses the
[PhysioNet Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/)
dataset (free, requires registration).

```bash
# After downloading, place files under data/physionet2019/
# then run preprocessing:
python preprocess.py --data_dir data/physionet2019/ --out_dir checkpoints/splits/
```

> **Important:** The preprocessing script applies the same train/val/test
> splits (stratified by hospital + SepsisLabel) used in the paper. Fix
> `seed=42` to reproduce exact splits.

---

## Usage

### Train a reconstruction model

```bash
# Example: TCN reconstructing MAP from NI + DBP (G2+, r=0.862)
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

### Reproduce all 15 experiments

On a SLURM cluster:

```bash
chmod +x scripts/submit_all_experiments.sh
./scripts/submit_all_experiments.sh
```

Then aggregate results:

```bash
python analysis/combine_results.py --results_dir results/ --out results/ALL_metrics.csv
```

### Train the sepsis classifier

```bash
# Full Supervision (oracle)
python train_sepsis_classifier_v2.py \
    --condition FS \
    --model tcn \
    --pos_weight 7 \
    --epochs 100

# Incomplete Supervision (MAP missing)
python train_sepsis_classifier_v2.py \
    --condition IS \
    --missing_vital MAP \
    --model tcn \
    --pos_weight 7 \
    --epochs 100
```

### Clinical utility evaluation

```bash
python evaluate_clinical_utility.py --pos_weight 5
```

### Reproduce paper figures

```bash
# Figure 2: Progressive sensor availability
python analysis/plot_reconstruction_results.py \
    --metrics results/ALL_metrics.csv \
    --outdir figures/

# Figure A1–A3: Waveform grid (Appendix)
python analysis/plot_all_experiments.py --seeds 42
```

---

## Data & Checkpoints

### Data splits

The train/val/test patient splits used in the paper are available as CSV
files at:

```
checkpoints/splits/train.csv
checkpoints/splits/val.csv
checkpoints/splits/test.csv
```

These contain Patient_IDs and preprocessed hourly vital sign values.
Due to PhysioNet data use agreements, we cannot redistribute the raw
patient data. Download from
[PhysioNet](https://physionet.org/content/challenge-2019/1.0.0/)
and run `preprocess.py`.

### Model checkpoints

Pre-trained reconstruction model checkpoints (`.pt` files, ~2 GB total)
are available at:

> 🔗 **[Zenodo — DOI: 10.5281/zenodo.XXXXXXX]** *(link added after publication)*

Download and place under `checkpoints/models/`.

---

## Results

Pre-computed results are included in `results/`:

| File | Contents |
|------|----------|
| `ALL_metrics.csv` | 15 experiments × 6 models, window-level RMSE/MAE/MAPE/r |
| `ALL_metrics_per_patient.csv` | Patient-level aggregation |
| `ALL_metrics_subgroup.csv` | Sepsis vs non-sepsis subgroup |

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
`T=12` (12-hour window) and `C_out=1` for single-target reconstruction.

---

## Citation

If you use this code or results in your research, please cite:

```bibtex
@article{hindawi2025icu,
  title   = {Physiological Information Redundancy in {ICU} Vital Signs:
             A Systematic Investigation of Cross-Modal Reconstruction
             and Clinical Utility},
  author  = {Hindawi, Firas},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The PhysioNet 2019 dataset is subject to its own
[data use agreement](https://physionet.org/content/challenge-2019/1.0.0/).

---

## Contact

Firas Hindawi — King Fahd University of Petroleum and Minerals
📧 firas.hindawi@kfupm.edu.sa
