"""
train_sepsis_classifier.py

Trains a TCNClassifier to predict sepsis risk from a 12-hour window of
vital signs + demographics.

Labelling strategy: patient-level (Option C) - all windows from a sepsis
patient are labelled positive. This gives ~13% window-level prevalence
and a tractable 6.7:1 imbalance ratio.

Architecture:
    Vitals (7 × T) → TCN backbone → last timestep
    Concatenate [Age, Gender, ICULOS, HospAdmTime]
    → MLP → logit → sigmoid → P(sepsis)

Loss     : BCEWithLogitsLoss with pos_weight
Stopping : No early stopping - train for full --epochs,
           save best checkpoint based on val balanced accuracy

Usage:
    python train_sepsis_classifier.py --pos_weight 7 --epochs 200
    python train_sepsis_classifier.py --no_demographics --epochs 200
"""

import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score,
    f1_score, confusion_matrix,
)

from models import TCNClassifier

# ========================================================================
# Config
# ========================================================================
TRAIN_PATH  = "checkpoints/splits/train.csv"
VAL_PATH    = "checkpoints/splits/val.csv"
SAVE_DIR    = "classification_results"
WINDOW_SIZE = 12
STEP        = 4
SEED        = 42

VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
DEMO_COLS  = ["Age", "Gender", "ICULOS", "HospAdmTime"]

# ========================================================================
# Reproducibility
# ========================================================================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ========================================================================
# Data
# ========================================================================
def make_windows(df, vital_cols, demo_cols, window_size=WINDOW_SIZE, step=STEP):
    """
    Patient-level labelling (Option C):
    All windows from a sepsis patient are labelled 1, all others 0.
    Gives ~13% window-level prevalence (6.7:1 imbalance).
    """
    X_list, D_list, y_list = [], [], []
    sepsis_pids = set(df[df["SepsisLabel"] == 1]["Patient_ID"].unique())

    for pid, group in df.groupby("Patient_ID"):
        group     = group.sort_values("Hour").reset_index(drop=True)
        n         = len(group)
        if n < window_size:
            continue
        vitals    = group[vital_cols].values.astype(np.float32)
        pat_label = float(pid in sepsis_pids)
        demo      = group[demo_cols].iloc[0].values.astype(np.float32) \
                    if demo_cols else None

        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            X_list.append(vitals[start:end].T)   # (n_vitals, T)
            y_list.append(pat_label)
            if demo is not None:
                D_list.append(demo)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    D = np.stack(D_list) if D_list \
        else np.zeros((len(X), 0), dtype=np.float32)
    return X, D, y


class SepsisDataset(Dataset):
    def __init__(self, X, D, y):
        self.X = torch.from_numpy(X).float()
        self.D = torch.from_numpy(D).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.D[idx], self.y[idx]

# ========================================================================
# Comprehensive metrics
# ========================================================================
def compute_all_metrics(probs, labels, loss):
    """
    Compute all relevant metrics at optimal F1 threshold.
    Returns a dict with every metric we might want to report.
    """
    # Threshold-free metrics
    auroc = float(roc_auc_score(labels, probs)) \
            if labels.sum() > 0 else float("nan")
    auprc = float(average_precision_score(labels, probs)) \
            if labels.sum() > 0 else float("nan")

    # Find optimal threshold by F1
    thresholds  = np.linspace(0.01, 0.99, 200)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    preds = (probs >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    sensitivity  = tp / (tp + fn + 1e-8)   # recall / TPR
    specificity  = tn / (tn + fp + 1e-8)   # TNR
    precision    = tp / (tp + fp + 1e-8)   # PPV
    npv          = tn / (tn + fn + 1e-8)   # NPV
    accuracy     = float(accuracy_score(labels, preds))
    bal_accuracy = float(balanced_accuracy_score(labels, preds))

    return {
        "loss":         loss,
        "auroc":        auroc,
        "auprc":        auprc,
        "bal_accuracy": bal_accuracy,
        "accuracy":     accuracy,
        "f1":           best_f1,
        "sensitivity":  sensitivity,
        "specificity":  specificity,
        "precision":    precision,
        "npv":          npv,
        "threshold":    best_t,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for x, d, y in loader:
            x, d, y = x.to(device), d.to(device), y.to(device)
            logits  = model(x, d if d.shape[1] > 0 else None)
            loss    = criterion(logits, y)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    avg_loss = total_loss / max(len(loader), 1)

    return compute_all_metrics(probs, labels, avg_loss), probs, labels

# ========================================================================
# Main
# ========================================================================
def main(args):
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device        : {device}")
    print(f"Seed          : {SEED}")
    print(f"pos_weight    : {args.pos_weight}")
    print(f"Epochs        : {args.epochs}")
    print(f"Demographics  : {'Yes' if not args.no_demographics else 'No'}")

    demo_cols = [] if args.no_demographics else DEMO_COLS

    # -- Load data ---------------------------------------------------------
    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)

    X_train, D_train, y_train = make_windows(df_train, VITAL_COLS, demo_cols)
    X_val,   D_val,   y_val   = make_windows(df_val,   VITAL_COLS, demo_cols)

    print(f"\nTrain windows : {len(y_train):,}  "
          f"(sepsis: {int(y_train.sum()):,}, {y_train.mean()*100:.1f}%)")
    print(f"Val windows   : {len(y_val):,}  "
          f"(sepsis: {int(y_val.sum()):,}, {y_val.mean()*100:.1f}%)")

    train_ds     = SepsisDataset(X_train, D_train, y_train)
    val_ds       = SepsisDataset(X_val,   D_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              worker_init_fn=lambda _: set_seed(SEED))
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False)

    # -- Build model -------------------------------------------------------
    model = TCNClassifier(
        n_vitals   = len(VITAL_COLS),
        n_demo     = len(demo_cols),
        channels   = (32, 32, 64, 64),
        kernel_size= 5,
        dropout    = 0.2,      # increased from 0.1 to reduce overfitting
        mlp_hidden = 64,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params  : {n_params:,}")

    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10)

    pw_tag       = str(args.pos_weight).replace(".", "p")
    ckpt_path    = os.path.join(SAVE_DIR, f"tcn_classifier_pw{pw_tag}_best.pt")
    history_path = os.path.join(SAVE_DIR, f"tcn_classifier_pw{pw_tag}_history.json")

    best_bal_acc = 0.0
    history      = []

    # Header
    print(f"\n{'-'*105}")
    print(f"{'Ep':>4}  {'TrLoss':>8}  {'VaLoss':>8}  "
          f"{'AUROC':>7}  {'AUPRC':>7}  {'BalAcc':>7}  "
          f"{'Acc':>7}  {'F1':>7}  {'Sens':>7}  {'Spec':>7}  {'Prec':>7}")
    print(f"{'-'*105}")

    for epoch in range(1, args.epochs + 1):

        # -- Train ----------------------------------------------------------
        model.train()
        train_loss = 0.0
        for x, d, y in train_loader:
            x, d, y = x.to(device), d.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x, d if d.shape[1] > 0 else None)
            loss   = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        # -- Validate -------------------------------------------------------
        val_m, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_m["bal_accuracy"])

        print(f"{epoch:4d}  {train_loss:8.4f}  {val_m['loss']:8.4f}  "
              f"{val_m['auroc']:7.4f}  {val_m['auprc']:7.4f}  "
              f"{val_m['bal_accuracy']:7.4f}  {val_m['accuracy']:7.4f}  "
              f"{val_m['f1']:7.4f}  {val_m['sensitivity']:7.4f}  "
              f"{val_m['specificity']:7.4f}  {val_m['precision']:7.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss, **val_m})
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best based on balanced accuracy
        if val_m["bal_accuracy"] > best_bal_acc + 1e-4:
            best_bal_acc = val_m["bal_accuracy"]
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_bal_accuracy":    best_bal_acc,
                "val_metrics":          val_m,
                "args":                 vars(args),
                "vital_cols":           VITAL_COLS,
                "demo_cols":            demo_cols,
                "seed":                 SEED,
            }, ckpt_path)
            print(f"      Best saved  "
                  f"(BalAcc={best_bal_acc:.4f}  "
                  f"AUROC={val_m['auroc']:.4f}  "
                  f"AUPRC={val_m['auprc']:.4f})")

    print(f"\nDone. Best val balanced accuracy: {best_bal_acc:.4f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"History    : {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--pos_weight",      type=float, default=7.0)
    parser.add_argument("--no_early_stopping", action="store_true",
                        help="Kept for compatibility - early stopping is "
                             "disabled by default in this version")
    parser.add_argument("--no_demographics", action="store_true")
    args = parser.parse_args()
    main(args)