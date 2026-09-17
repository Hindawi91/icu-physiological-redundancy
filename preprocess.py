"""
preprocess.py

Preprocessing pipeline for the PhysioNet Computing in Cardiology
Challenge 2019 dataset used in:

    "Physiological Information Redundancy in ICU Vital Signs:
     A Systematic Investigation of Cross-Modal Reconstruction
     and Clinical Utility"

Steps:
    1. Load and combine raw PSV files from both hospital cohorts
    2. Separate Hospital A (7-digit Patient IDs) and Hospital B (6-digit)
    3. Remove EtCO2 (>99% missing in Hospital A)
    4. Forward-fill imputation within each patient stay
    5. Drop rows where all vital signs are still missing
    6. Save combined preprocessed CSV

Usage:
    python preprocess.py \
        --data_dir /path/to/physionet2019/training/ \
        --out_dir  checkpoints/splits/

    # The PhysioNet 2019 data has two subdirectories:
    #   training/training_setA/   ← Hospital A (.psv files)
    #   training/training_setB/   ← Hospital B (.psv files)
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ========================================================================
# Config
# ========================================================================
VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
DEMO_COLS  = ["Age", "Gender", "ICULOS", "HospAdmTime"]
LABEL_COL  = "SepsisLabel"
SEED       = 42


# ========================================================================
# Loading
# ========================================================================
def load_psv_files(data_dir, hospital_label):
    """
    Load all .psv files from a PhysioNet 2019 hospital directory.
    Adds Patient_ID and Hospital columns.
    """
    records = []
    data_path = Path(data_dir)
    psv_files = sorted(data_path.glob("*.psv"))

    if not psv_files:
        raise FileNotFoundError(
            f"No .psv files found in {data_dir}. "
            "Check your PhysioNet data directory.")

    print(f"  Loading {len(psv_files)} patients from {hospital_label}...")
    for f in psv_files:
        pid = f.stem.lstrip("p")   # remove leading 'p' from filename
        df  = pd.read_csv(f, sep="|")
        df["Patient_ID"] = pid
        df["Hospital"]   = hospital_label
        df["Hour"]       = range(len(df))
        records.append(df)

    combined = pd.concat(records, ignore_index=True)
    print(f"    {combined['Patient_ID'].nunique():,} patients, "
          f"{len(combined):,} rows")
    return combined


# ========================================================================
# Imputation
# ========================================================================
def forward_fill_within_patient(df, cols):
    """
    Forward-fill missing values within each patient stay,
    then back-fill for leading NaNs.
    """
    df = df.copy()
    df[cols] = (
        df.groupby("Patient_ID")[cols]
          .transform(lambda x: x.ffill().bfill())
    )
    return df


def fill_remaining_with_mean(df, cols):
    """
    Fill any remaining NaNs (patients with all-NaN channels)
    with the training set column mean.
    """
    df = df.copy()
    for col in cols:
        mean_val = df[col].mean()
        df[col]  = df[col].fillna(mean_val)
    return df


# ========================================================================
# Splitting
# ========================================================================
def stratified_patient_split(df, val_frac=0.15, test_frac=0.15, seed=42):
    """
    Patient-level stratified split by Hospital + SepsisLabel.
    Returns train, val, test DataFrames.
    """
    import random
    rng = random.Random(seed)

    # Identify sepsis patients per hospital
    patient_info = (
        df.groupby("Patient_ID")
          .agg(hospital=("Hospital", "first"),
               sepsis=("SepsisLabel", "max"))
          .reset_index()
    )

    train_ids, val_ids, test_ids = [], [], []

    for (hosp, sep), group in patient_info.groupby(["hospital", "sepsis"]):
        pids = group["Patient_ID"].tolist()
        rng.shuffle(pids)
        n      = len(pids)
        n_test = max(1, int(n * test_frac))
        n_val  = max(1, int(n * val_frac))
        test_ids.extend(pids[:n_test])
        val_ids.extend(pids[n_test:n_test + n_val])
        train_ids.extend(pids[n_test + n_val:])

    df_train = df[df["Patient_ID"].isin(train_ids)].reset_index(drop=True)
    df_val   = df[df["Patient_ID"].isin(val_ids)].reset_index(drop=True)
    df_test  = df[df["Patient_ID"].isin(test_ids)].reset_index(drop=True)

    return df_train, df_val, df_test


# ========================================================================
# Main
# ========================================================================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # -- 1. Load raw data ----------------------------------------------
    print("\nStep 1: Loading raw PhysioNet 2019 data...")
    df_a = load_psv_files(
        os.path.join(args.data_dir, "training_setA"), "A")
    df_b = load_psv_files(
        os.path.join(args.data_dir, "training_setB"), "B")
    df   = pd.concat([df_a, df_b], ignore_index=True)
    print(f"\n  Combined: {df['Patient_ID'].nunique():,} patients, "
          f"{len(df):,} rows")

    # -- 2. Drop EtCO2 (>99% missing in Hospital A) -------------------
    print("\nStep 2: Dropping EtCO2 (>99% missing in Hospital A)...")
    if "EtCO2" in df.columns:
        df = df.drop(columns=["EtCO2"])

    # -- 3. Report missingness -----------------------------------------
    print("\nStep 3: Missingness in vital sign columns:")
    keep_cols = VITAL_COLS + DEMO_COLS + [LABEL_COL, "Patient_ID", "Hour", "Hospital"]
    df        = df[[c for c in keep_cols if c in df.columns]]

    for col in VITAL_COLS:
        pct = df[col].isna().mean() * 100
        print(f"  {col:<10}: {pct:.1f}% missing")

    # -- 4. Forward-fill imputation ------------------------------------
    print("\nStep 4: Forward-fill imputation within patient stays...")
    df = forward_fill_within_patient(df, VITAL_COLS)

    # -- 5. Fill remaining with column mean ----------------------------
    print("\nStep 5: Filling remaining NaNs with column mean...")
    df = fill_remaining_with_mean(df, VITAL_COLS)

    # Verify no missing vital signs remain
    remaining = df[VITAL_COLS].isna().sum().sum()
    assert remaining == 0, f"Still {remaining} missing values after imputation!"
    print(f"  No missing values remain in vital sign columns")

    # -- 6. Patient-level stratified split ----------------------------
    print("\nStep 6: Splitting into train/val/test...")
    df_train, df_val, df_test = stratified_patient_split(
        df, val_frac=0.15, test_frac=0.15, seed=SEED)

    for split, name in [(df_train, "train"), (df_val, "val"), (df_test, "test")]:
        n_pat    = split["Patient_ID"].nunique()
        n_sepsis = split[split["SepsisLabel"] == 1]["Patient_ID"].nunique()
        pct_sep  = n_sepsis / n_pat * 100
        out_path = os.path.join(args.out_dir, f"{name}.csv")
        split.to_csv(out_path, index=False)
        print(f"  {name:5s}: {n_pat:6,} patients  "
              f"({n_sepsis:,} sepsis, {pct_sep:.1f}%)  → {out_path}")

    # -- 7. Summary ----------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  Preprocessing complete.")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Splits saved:     train.csv, val.csv, test.csv")
    print(f"  Vital columns:    {', '.join(VITAL_COLS)}")
    print(f"  Random seed:      {SEED}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess PhysioNet 2019 data for ICU vital sign reconstruction"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to PhysioNet 2019 training directory "
             "(containing training_setA/ and training_setB/)")
    parser.add_argument(
        "--out_dir", type=str, default="checkpoints/splits/",
        help="Output directory for train/val/test CSV splits")
    args = parser.parse_args()
    main(args)
