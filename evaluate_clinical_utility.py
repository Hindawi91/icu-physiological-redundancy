"""
evaluate_clinical_utility.py

Evaluates how well reconstructed vital signs preserve sepsis prediction
performance compared to true signals.

Runs 23 conditions:
    - Full (baseline)
    - Drop each vital (7 conditions)
    - Replace each dropped vital with reconstruction (15 conditions)

For MAP/SBP/DBP: tests G1 (non-invasive only), G2+ (+1 BP), G2++ (+2 BP)
For Temp, Resp, O2Sat: tests single reconstruction condition

Outputs:
    classification_results/clinical_utility_results.csv
    classification_results/clinical_utility_summary.txt

Usage:
    python evaluate_clinical_utility.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, confusion_matrix,
                              accuracy_score, balanced_accuracy_score)

from models import TCNClassifier, TCN, build_model

# ════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════
TEST_PATH      = "checkpoints/splits/test.csv"
TRAIN_PATH     = "checkpoints/splits/train.csv"
RECON_DIR      = "checkpoints/models"
CLASSIFIER_DIR = "classification_results"
SAVE_DIR       = "classification_results"
WINDOW_SIZE    = 12
STEP           = 4
SEED           = 42

VITAL_COLS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
DEMO_COLS  = ["Age", "Gender", "ICULOS", "HospAdmTime"]

# All 23 evaluation conditions
# Format: (label, vital_to_replace, reconstruction_run_tag or None)
# vital_to_replace=None means use true signal for all vitals
CONDITIONS = [
    # ── Full baseline ──────────────────────────────────────────────────
    ("Full (all true vitals)", None, None),

    # ── Drop one vital (mean imputation) ──────────────────────────────
    ("Remove HR",    "HR",    None),
    ("Remove O2Sat", "O2Sat", None),
    ("Remove Temp",  "Temp",  None),
    ("Remove SBP",   "SBP",   None),
    ("Remove MAP",   "MAP",   None),
    ("Remove DBP",   "DBP",   None),
    ("Remove Resp",  "Resp",  None),

    # ── Reconstruct MAP ───────────────────────────────────────────────
    ("MAP recon G1 (NI only, r=0.225)",   "MAP", "HR-O2Sat-Resp-Temp_to_MAP"),
    ("MAP recon G2+ (+DBP, r=0.863)",     "MAP", "HR-O2Sat-Resp-Temp-DBP_to_MAP"),
    ("MAP recon G2++ (+DBP+SBP, r=0.936)","MAP", "HR-O2Sat-Resp-Temp-DBP-SBP_to_MAP"),

    # ── Reconstruct SBP ───────────────────────────────────────────────
    ("SBP recon G1 (NI only, r=0.231)",   "SBP", "HR-O2Sat-Resp-Temp_to_SBP"),
    ("SBP recon G2+ (+DBP, r=0.603)",     "SBP", "HR-O2Sat-Resp-Temp-DBP_to_SBP"),
    ("SBP recon G2++ (+DBP+MAP, r=0.886)","SBP", "HR-O2Sat-Resp-Temp-DBP-MAP_to_SBP"),

    # ── Reconstruct DBP ───────────────────────────────────────────────
    ("DBP recon G1 (NI only, r=0.273)",   "DBP", "HR-O2Sat-Resp-Temp_to_DBP"),
    ("DBP recon G2+ (+SBP, r=0.599)",     "DBP", "HR-O2Sat-Resp-Temp-SBP_to_DBP"),
    ("DBP recon G2++ (+SBP+MAP, r=0.915)","DBP", "HR-O2Sat-Resp-Temp-SBP-MAP_to_DBP"),

    # ── Reconstruct Temp ──────────────────────────────────────────────
    ("Temp recon (r=0.383)",   "Temp",  "HR-O2Sat-Resp-MAP-SBP-DBP_to_Temp"),

    # ── Reconstruct Resp ──────────────────────────────────────────────
    ("Resp recon (r=0.301)",   "Resp",  "HR-O2Sat-Temp-MAP-SBP-DBP_to_Resp"),

    # ── Reconstruct O2Sat ─────────────────────────────────────────────
    ("O2Sat recon (r=0.207)",  "O2Sat", "HR-Resp-Temp-MAP-SBP-DBP_to_O2Sat"),
]


# ════════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════════
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════════════
def make_windows(df, vital_cols, demo_cols, window_size=WINDOW_SIZE, step=STEP):
    X_list, D_list, y_list = [], [], []
    sepsis_pids = set(df[df["SepsisLabel"] == 1]["Patient_ID"].unique())
    for pid, group in df.groupby("Patient_ID"):
        group  = group.sort_values("Hour").reset_index(drop=True)
        n      = len(group)
        if n < window_size:
            continue
        vitals    = group[vital_cols].values.astype(np.float32)
        pat_label = float(pid in sepsis_pids)
        demo      = group[demo_cols].iloc[0].values.astype(np.float32) if demo_cols else None
        for start in range(0, n - window_size + 1, step):
            end = start + window_size
            X_list.append(vitals[start:end].T)
            y_list.append(pat_label)
            if demo is not None:
                D_list.append(demo)
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    D = np.stack(D_list) if D_list else np.zeros((len(X), 0), dtype=np.float32)
    return X, D, y


# ════════════════════════════════════════════════════════════════════════
# Reconstruction inference
# ════════════════════════════════════════════════════════════════════════
def load_recon_model(run_tag, in_ch, device):
    """Load best TCN reconstruction checkpoint for a given run_tag.
    in_ch is derived from run_tag (inputs before '_to_') to match
    the exact architecture the checkpoint was trained with.
    """
    path = os.path.join(RECON_DIR, f"tcn_{run_tag}_best.pt")
    if not os.path.exists(path):
        print(f"   ⚠️  Reconstruction checkpoint not found: {path}")
        return None
    # Parse in_ch from run_tag: "HR-O2Sat-Resp-Temp_to_MAP" -> 4 inputs
    input_part = run_tag.split("_to_")[0]
    actual_in_ch = len(input_part.split("-"))
    model = build_model("tcn", in_channels=actual_in_ch, out_channels=1)
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def reconstruct_vital(X_test, vital_idx, run_tag, vital_cols, device,
                      batch_size=512):
    """
    Replace one channel in X_test with reconstructed values.
    X_test   : (N, n_vitals, T)  — true test windows (all 7 vitals)
    run_tag  : encodes exact input columns, e.g. HR-O2Sat-Resp-Temp_to_MAP
    Returns a copy of X_test with vital_idx channel replaced.
    """
    # Parse exact input columns from run_tag
    # e.g. "HR-O2Sat-Resp-Temp_to_MAP" -> ["HR","O2Sat","Resp","Temp"]
    input_part  = run_tag.split("_to_")[0]
    input_names = input_part.split("-")
    input_indices = [vital_cols.index(c) for c in input_names
                     if c in vital_cols]
    X_input = X_test[:, input_indices, :]   # (N, n_inputs, T)

    model = load_recon_model(run_tag, X_input.shape[1], device)
    if model is None:
        return None

    X_t    = torch.from_numpy(X_input).float()
    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size,
                        shuffle=False)
    preds  = []
    with torch.no_grad():
        for (x,) in loader:
            out = model(x.to(device))    # (B, 1, T)
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # (N, 1, T)

    X_replaced = X_test.copy()
    X_replaced[:, vital_idx, :] = preds[:, 0, :]
    return X_replaced


# ════════════════════════════════════════════════════════════════════════
# Classifier evaluation
# ════════════════════════════════════════════════════════════════════════
def compute_metrics(probs, labels):
    """
    Compute all relevant metrics at the optimal F1 threshold.
    """
    auroc = float(roc_auc_score(labels, probs))             if labels.sum() > 0 else float("nan")
    auprc = float(average_precision_score(labels, probs))             if labels.sum() > 0 else float("nan")

    # Find optimal threshold by F1
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (probs >= t).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t

    opt_preds    = (probs >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, opt_preds, labels=[0,1]).ravel()

    sensitivity  = tp / (tp + fn + 1e-8)
    specificity  = tn / (tn + fp + 1e-8)
    precision    = tp / (tp + fp + 1e-8)
    npv          = tn / (tn + fn + 1e-8)
    accuracy     = float(accuracy_score(labels, opt_preds))
    bal_accuracy = float(balanced_accuracy_score(labels, opt_preds))

    return {
        "AUROC":        auroc,
        "AUPRC":        auprc,
        "F1":           best_f1,
        "Sensitivity":  sensitivity,
        "Specificity":  specificity,
        "Precision":    precision,
        "NPV":          npv,
        "Accuracy":     accuracy,
        "Bal_Accuracy": bal_accuracy,
        "Threshold":    best_t,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def run_classifier(model, X, D, device, batch_size=512):
    """Run frozen classifier on given windows, return probabilities."""
    model.eval()
    X_t = torch.from_numpy(X).float()
    D_t = torch.from_numpy(D).float()
    ds  = TensorDataset(X_t, D_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs  = []
    with torch.no_grad():
        for x, d in loader:
            x, d = x.to(device), d.to(device)
            logits = model(x, d if d.shape[1] > 0 else None)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load classifier ───────────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_weight", type=float, default=7.0,
                        help="pos_weight used during training — selects checkpoint")
    cli = parser.parse_args()
    pw_tag    = str(cli.pos_weight).replace(".", "p")
    ckpt_path = os.path.join(CLASSIFIER_DIR, f"tcn_classifier_pw{pw_tag}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Classifier checkpoint not found at {ckpt_path}. "
            f"Run train_sepsis_classifier.py --pos_weight {cli.pos_weight} first.")

    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
    demo_cols = ckpt["demo_cols"]
    vital_cols_saved = ckpt["vital_cols"]
    assert vital_cols_saved == VITAL_COLS, \
        f"Vital cols mismatch: {vital_cols_saved} vs {VITAL_COLS}"

    best_metric = ckpt.get("best_bal_accuracy", ckpt.get("best_auroc", float("nan")))
    print(f"Classifier loaded — best val balanced accuracy: {best_metric:.4f}")
    print(f"Demographics: {demo_cols if demo_cols else 'None'}")

    classifier = TCNClassifier(
        n_vitals=len(VITAL_COLS),
        n_demo=len(demo_cols),
        channels=(32, 32, 64, 64),
        kernel_size=5, dropout=0.2, mlp_hidden=64,
    ).to(device)
    classifier.load_state_dict(ckpt["model_state_dict"])
    classifier.eval()
    # Freeze classifier — weights never change across conditions
    for p in classifier.parameters():
        p.requires_grad = False

    # ── Load test data ────────────────────────────────────────────────────
    print("\nLoading test data...")
    df_test = pd.read_csv(TEST_PATH)

    X_true, D_test, y_test = make_windows(df_test, VITAL_COLS, demo_cols)
    print(f"Test windows  : {len(y_test):,}  "
          f"(sepsis: {int(y_test.sum()):,}, {y_test.mean()*100:.1f}%)")

    # Precompute per-vital train means for mean imputation
    df_train = pd.read_csv(TRAIN_PATH)
    vital_means = df_train[VITAL_COLS].mean().values.astype(np.float32)

    # ── Run all conditions ─────────────────────────────────────────────────
    print(f"\nRunning {len(CONDITIONS)} conditions...\n")
    results = []

    # Store baseline metrics for computing % recovered
    baseline_metrics = None

    for label, vital_to_replace, recon_run_tag in CONDITIONS:
        print(f"  {label}")

        if vital_to_replace is None:
            # Full baseline — use true signals
            X_eval = X_true.copy()

        elif recon_run_tag is None:
            # Drop condition — replace with training mean
            vital_idx = VITAL_COLS.index(vital_to_replace)
            X_eval    = X_true.copy()
            X_eval[:, vital_idx, :] = vital_means[vital_idx]

        else:
            # Reconstruction condition
            vital_idx = VITAL_COLS.index(vital_to_replace)
            X_eval    = reconstruct_vital(
                X_true, vital_idx, recon_run_tag, VITAL_COLS, device)
            if X_eval is None:
                print(f"    ⚠️  Skipping — reconstruction model not available")
                continue

        # Run frozen classifier
        probs   = run_classifier(classifier, X_eval, D_test, device)
        metrics = compute_metrics(probs, y_test)
        metrics["Condition"] = label
        metrics["Vital_replaced"] = vital_to_replace if vital_to_replace else "None"
        metrics["Reconstruction"] = recon_run_tag if recon_run_tag else "N/A"
        results.append(metrics)

        if baseline_metrics is None:
            baseline_metrics = metrics

        print(f"    AUROC={metrics['AUROC']:.4f}  AUPRC={metrics['AUPRC']:.4f}  "
              f"BalAcc={metrics['Bal_Accuracy']:.4f}  Acc={metrics['Accuracy']:.4f}  "
              f"F1={metrics['F1']:.4f}  Sens={metrics['Sensitivity']:.4f}  "
              f"Spec={metrics['Specificity']:.4f}  Prec={metrics['Precision']:.4f}  "
              f"NPV={metrics['NPV']:.4f}")

    # ── Compute % AUROC recovered ─────────────────────────────────────────
    bl_auroc = baseline_metrics["AUROC"]

    # For each reconstruction condition, find the corresponding drop condition
    for r in results:
        if r["Reconstruction"] != "N/A" and r["Vital_replaced"] != "None":
            # Find drop-one condition for same vital
            drop_label = f"Remove {r['Vital_replaced']}"
            drop_row   = next((x for x in results if x["Condition"] == drop_label), None)
            if drop_row:
                lost    = bl_auroc - drop_row["AUROC"]
                gained  = r["AUROC"] - drop_row["AUROC"]
                r["AUROC_drop"]      = drop_row["AUROC"]
                r["AUROC_lost"]      = lost
                r["AUROC_recovered"] = gained
                r["Pct_recovered"]   = (gained / lost * 100) if lost > 1e-6 else float("nan")
            else:
                r["AUROC_drop"] = float("nan")
                r["AUROC_lost"] = float("nan")
                r["AUROC_recovered"] = float("nan")
                r["Pct_recovered"]   = float("nan")
        else:
            r["AUROC_drop"]      = float("nan")
            r["AUROC_lost"]      = float("nan")
            r["AUROC_recovered"] = float("nan")
            r["Pct_recovered"]   = float("nan")

    # ── Save results ──────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    csv_path   = os.path.join(SAVE_DIR, "clinical_utility_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n✅ Saved: {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("\n" + "="*95)
    summary_lines.append("  Clinical Utility — Information Redundancy Map")
    summary_lines.append(f"  Baseline AUROC (all true vitals): {bl_auroc:.4f}")
    summary_lines.append("="*95)
    summary_lines.append(
        f"\n  {'Condition':<45} {'AUROC':>7} {'AUPRC':>7} "
        f"{'BalAcc':>7} {'Acc':>7} {'F1':>6} "
        f"{'Sens':>6} {'Spec':>6} {'Prec':>6} {'NPV':>6} {'%Recov':>8}")
    summary_lines.append("  " + "-"*115)

    for r in results:
        pct = f"{r['Pct_recovered']:.1f}%"               if not np.isnan(r.get("Pct_recovered", float("nan"))) else ""
        line = (f"  {r['Condition']:<45} "
                f"{r['AUROC']:>7.4f} {r['AUPRC']:>7.4f} "
                f"{r['Bal_Accuracy']:>7.4f} {r['Accuracy']:>7.4f} "
                f"{r['F1']:>6.4f} {r['Sensitivity']:>6.4f} "
                f"{r['Specificity']:>6.4f} {r['Precision']:>6.4f} "
                f"{r['NPV']:>6.4f} {pct:>8}")
        summary_lines.append(line)

    summary = "\n".join(summary_lines)
    print(summary)

    txt_path = os.path.join(SAVE_DIR, "clinical_utility_summary.txt")
    with open(txt_path, "w") as f:
        f.write(summary)
    print(f"\n✅ Summary saved: {txt_path}")


if __name__ == "__main__":
    main()