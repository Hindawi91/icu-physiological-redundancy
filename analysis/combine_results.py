"""
combine_results.py

Aggregates all 15 experiment results into:
    1. results/ALL_metrics.csv          - window-level metrics, all experiments
    2. results/ALL_metrics_per_patient.csv - per-patient metrics, all experiments
    3. results/ALL_metrics_subgroup.csv - sepsis vs healthy, all experiments
    4. paper_figures/ALL_experiments.png - 15 rows × 6 cols comprehensive figure

Run after all test.py calls have completed.

Usage:
    python combine_results.py
"""

import os
import re
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader

from models import build_model, MODEL_REGISTRY

# ========================================================================
# Config
# ========================================================================

RESULTS_DIR    = "results"
FIGURES_DIR    = "paper_figures"
TEST_PATH      = "checkpoints/splits/test.csv"
TRAIN_PATH     = "checkpoints/splits/train.csv"
CHECKPOINT_DIR = "checkpoints/models"
WINDOW_SIZE    = 12
STEP           = 4
SEED           = 42

# All 15 experiments in order
EXPERIMENTS = [
    # Group 1
    {"group": 1, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp"},
    {"group": 1, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp"},
    {"group": 1, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp"},
    # Group 2
    {"group": 2, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp,DBP"},
    {"group": 2, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp,SBP"},
    {"group": 2, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp,DBP,SBP"},
    {"group": 2, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp,DBP"},
    {"group": 2, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp,MAP"},
    {"group": 2, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp,DBP,MAP"},
    {"group": 2, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp,SBP"},
    {"group": 2, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp,MAP"},
    {"group": 2, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp,SBP,MAP"},
    # Group 3
    {"group": 3, "target": "Temp",  "inputs": "HR,O2Sat,Resp,MAP,SBP,DBP"},
    # Group 4
    {"group": 4, "target": "Resp",  "inputs": "HR,O2Sat,Temp,MAP,SBP,DBP"},
    {"group": 4, "target": "O2Sat", "inputs": "HR,Resp,Temp,MAP,SBP,DBP"},
]

DL_COLORS = {
    "unet1d":       "#2196F3",
    "bilstm":       "#F44336",
    "lstm_seq2seq": "#FF9800",
    "tcn":          "#4CAF50",
    "transformer":  "#9C27B0",
    "conv_lstm":    "#00BCD4",
}
DL_LABELS = {
    "unet1d":       "U-Net",
    "bilstm":       "BiLSTM",
    "lstm_seq2seq": "LSTM S2S",
    "tcn":          "TCN",
    "transformer":  "Transformer",
    "conv_lstm":    "Conv-LSTM",
}
BASELINE_COLOR = "#795548"
BASELINE_LABEL = "Lin. Regr."

UNITS = {
    "MAP":   "MAP (mmHg)",
    "SBP":   "SBP (mmHg)",
    "DBP":   "DBP (mmHg)",
    "Resp":  "Resp (br/min)",
    "O2Sat": "O2Sat (%)",
    "Temp":  "Temp (°C)",
    "HR":    "HR (bpm)",
}

GROUP_LABELS = {
    1: "G1: Non-Invasive → BP",
    2: "G2: Non-Invasive + Partial → BP",
    3: "G3: Temperature",
    4: "G4: Resp & O2Sat",
}


# ========================================================================
# Step 1: Combine per-experiment CSVs into ALL_* files
# ========================================================================

def combine_csvs():
    print("\n-- Combining CSVs --------------------------------------")

    all_window, all_patient, all_subgroup = [], [], []

    for exp in EXPERIMENTS:
        target     = exp["target"]
        inputs     = exp["inputs"]
        input_tag  = inputs.replace(",", "-")
        target_tag = target
        run_tag    = f"{input_tag}_to_{target_tag}"

        label = f"{target} ← {inputs}"

        # Window-level
        path = os.path.join(RESULTS_DIR, f"metrics_{run_tag}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df.insert(0, "Inputs", inputs)
            df.insert(0, "Target", target)
            df.insert(0, "Group", exp["group"])
            df.insert(0, "Experiment", label)
            all_window.append(df)
            print(f"   {run_tag}")
        else:
            print(f"   Missing: metrics_{run_tag}.csv")

        # Per-patient
        path = os.path.join(RESULTS_DIR, f"metrics_per_patient_{run_tag}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            df.insert(0, "Inputs", inputs)
            df.insert(0, "Target", target)
            df.insert(0, "Group", exp["group"])
            df.insert(0, "Experiment", label)
            all_patient.append(df)

        # Subgroup
        path = os.path.join(RESULTS_DIR, f"metrics_subgroup_{run_tag}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.insert(0, "Inputs", inputs)
            df.insert(0, "Target", target)
            df.insert(0, "Group", exp["group"])
            df.insert(0, "Experiment", label)
            all_subgroup.append(df)

    # Save combined files
    if all_window:
        out = pd.concat(all_window)
        out.to_csv(os.path.join(RESULTS_DIR, "ALL_metrics.csv"))
        print(f"\n   Saved: results/ALL_metrics.csv "
              f"({len(out)} rows)")

    if all_patient:
        out = pd.concat(all_patient)
        out.to_csv(os.path.join(RESULTS_DIR, "ALL_metrics_per_patient.csv"))
        print(f"   Saved: results/ALL_metrics_per_patient.csv "
              f"({len(out)} rows)")

    if all_subgroup:
        out = pd.concat(all_subgroup)
        out.to_csv(os.path.join(RESULTS_DIR, "ALL_metrics_subgroup.csv"),
                   index=False)
        print(f"   Saved: results/ALL_metrics_subgroup.csv "
              f"({len(out)} rows)")


# ========================================================================
# Step 2: Comprehensive figure - 15 rows × 6 cols
# ========================================================================

def make_windows(df, input_cols, target_cols):
    X_list, y_list, meta = [], [], []
    for pid, group in df.groupby("Patient_ID"):
        group = group.sort_values("Hour").reset_index(drop=True)
        n = len(group)
        if n < WINDOW_SIZE:
            continue
        x_vals = group[input_cols].values.astype(np.float32)
        y_vals = group[target_cols].values.astype(np.float32)
        sepsis = int(group["SepsisLabel"].max())
        for start in range(0, n - WINDOW_SIZE + 1, STEP):
            end = start + WINDOW_SIZE
            X_list.append(x_vals[start:end])
            y_list.append(y_vals[start:end])
            meta.append({"Patient_ID": pid,
                          "start_hour": int(group["Hour"].iloc[start]),
                          "sepsis_label": sepsis})
    X    = np.stack(X_list)
    y    = np.stack(y_list)
    meta = pd.DataFrame(meta)
    return X, y, meta


def load_dl_model(model_name, run_tag, in_ch, out_ch, device):
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_{run_tag}_best.pt")
    if not os.path.exists(path):
        return None
    model = build_model(model_name, in_channels=in_ch, out_channels=out_ch)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def run_inference(model, X, device, batch_size=256):
    X_t    = torch.from_numpy(X).float().permute(0, 2, 1)
    loader = DataLoader(torch.utils.data.TensorDataset(X_t),
                        batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (x,) in loader:
            preds.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0).transpose(0, 2, 1)


class NumpyLinReg:
    def fit(self, X, y):
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        res, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.coef_      = res[:-1].T
        self.intercept_ = res[-1]
        return self

    def predict(self, X):
        return X @ self.coef_.T + self.intercept_


def plot_comprehensive_figure(device):
    print("\n-- Generating comprehensive figure (15 × 6) ------------")

    df_test  = pd.read_csv(TEST_PATH)
    df_train = pd.read_csv(TRAIN_PATH)

    n_rows = len(EXPERIMENTS)
    n_cols = 6   # 3 healthy + 3 sepsis

    fig = plt.figure(figsize=(26, n_rows * 2.8))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                             hspace=0.45, wspace=0.25,
                             left=0.07, right=0.98,
                             top=0.97, bottom=0.05)

    # Column headers
    col_titles = ["Healthy 1", "Healthy 2", "Healthy 3",
                  "Sepsis 1",  "Sepsis 2",  "Sepsis 3"]

    hours = np.arange(WINDOW_SIZE)
    rng   = random.Random(SEED)

    current_group = None

    for row_idx, exp in enumerate(EXPERIMENTS):
        target     = exp["target"]
        inputs     = exp["inputs"]
        input_cols = inputs.split(",")
        target_col = [target]
        in_ch      = len(input_cols)
        out_ch     = 1
        input_tag  = inputs.replace(",", "-")
        run_tag    = f"{input_tag}_to_{target}"

        print(f"   [{row_idx+1:02d}/15] {target} ← {inputs}")

        # Data
        X_test,  y_test,  meta = make_windows(df_test,  input_cols, target_col)
        X_train, y_train, _    = make_windows(df_train, input_cols, target_col)

        # Select 3 healthy + 3 sepsis
        healthy = meta[meta["sepsis_label"] == 0].index.tolist()
        sepsis  = meta[meta["sepsis_label"] == 1].index.tolist()
        rng2    = random.Random(SEED + row_idx)
        sel_h   = rng2.sample(healthy, min(3, len(healthy)))
        sel_s   = rng2.sample(sepsis,  min(3, len(sepsis)))
        indices = sel_h + sel_s

        # DL predictions
        dl_preds = {}
        for name in MODEL_REGISTRY.keys():
            m = load_dl_model(name, run_tag, in_ch, out_ch, device)
            if m is None:
                continue
            dl_preds[name] = run_inference(m, X_test, device)

        # Linear regression
        lr = NumpyLinReg()
        lr.fit(X_train.reshape(-1, in_ch), y_train.reshape(-1, out_ch))
        bl_pred = lr.predict(
            X_test.reshape(-1, in_ch)).reshape(len(X_test), WINDOW_SIZE, out_ch)

        # Group separator line
        if exp["group"] != current_group:
            current_group = exp["group"]
            # Group change noted - spacing handled by gridspec hspace

        for col, (win_idx, is_sepsis) in enumerate(
                zip(indices, [False]*3 + [True]*3)):
            ax = fig.add_subplot(gs[row_idx, col])

            # Column titles on first row only
            if row_idx == 0:
                color = "green" if col < 3 else "darkred"
                ax.set_title(col_titles[col], fontsize=8,
                             fontweight="bold", color=color, pad=3)

            # Ground truth
            ax.plot(hours, y_test[win_idx, :, 0],
                    color="black", linewidth=2.0, zorder=10)

            # DL models
            for name, preds in dl_preds.items():
                ax.plot(hours, preds[win_idx, :, 0],
                        color=DL_COLORS[name], linewidth=0.9, alpha=0.8)

            # Best baseline
            ax.plot(hours, bl_pred[win_idx, :, 0],
                    color=BASELINE_COLOR, linewidth=0.9,
                    linestyle="--", alpha=0.8)

            ax.set_facecolor("#fff5f5" if is_sepsis else "#f5fff5")
            ax.tick_params(labelsize=6)
            ax.set_xticks([0, 3, 6, 9, 11])
            ax.grid(True, alpha=0.2, linestyle="--")

            # Vertical separator between healthy and sepsis
            if col == 2:
                ax.spines["right"].set_linewidth(1.5)
                ax.spines["right"].set_color("#555555")

            # Row label on leftmost column
            if col == 0:
                group_label = GROUP_LABELS[exp["group"]]
                short_inputs = inputs.replace("HR,O2Sat,Resp,Temp", "NI")
                row_label = f"{target}←{short_inputs}"
                ax.set_ylabel(row_label, fontsize=6.5,
                              labelpad=3, rotation=0,
                              ha="right", va="center")

    # Legend at bottom
    legend_elements = [
        Line2D([0], [0], color="black", linewidth=2.0, label="Ground Truth"),
    ]
    for name in MODEL_REGISTRY.keys():
        legend_elements.append(
            Line2D([0], [0], color=DL_COLORS[name], linewidth=1.2,
                   label=DL_LABELS[name])
        )
    legend_elements.append(
        Line2D([0], [0], color=BASELINE_COLOR, linewidth=1.2,
               linestyle="--", label=BASELINE_LABEL)
    )
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=8, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    # Group annotations on right side
    group_rows = {1: [], 2: [], 3: [], 4: []}
    for i, exp in enumerate(EXPERIMENTS):
        group_rows[exp["group"]].append(i)

    for g, rows in group_rows.items():
        mid_row = (rows[0] + rows[-1]) / 2
        fig.text(0.995, 1 - (mid_row + 0.5) / n_rows,
                 GROUP_LABELS[g], fontsize=7, fontweight="bold",
                 ha="right", va="center", rotation=270,
                 color="#333333")

    save_path = os.path.join(FIGURES_DIR, "ALL_experiments.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   Saved: {save_path}")


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR,  exist_ok=True)
    os.makedirs(FIGURES_DIR,  exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Combine CSVs
    combine_csvs()

    # Step 2: Comprehensive figure
    plot_comprehensive_figure(device)

    print("\nAll combined outputs generated:")
    print("   results/ALL_metrics.csv")
    print("   results/ALL_metrics_per_patient.csv")
    print("   results/ALL_metrics_subgroup.csv")
    print("   paper_figures/ALL_experiments.png")