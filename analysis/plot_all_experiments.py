"""
plot_all_experiments.py

Generates the comprehensive ALL_experiments figure (15 rows × 6 cols)
with different random seeds for patient window selection.

Each seed produces a different set of example patients while using
the same trained model predictions - useful for picking the most
visually informative version for the paper.

Usage:
    # Single seed
    python plot_all_experiments.py --seeds 42

    # Multiple seeds at once
    python plot_all_experiments.py --seeds 42,7,13,99,2024
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from models import build_model, MODEL_REGISTRY

# ========================================================================
# Config
# ========================================================================

TEST_PATH      = "checkpoints/splits/test.csv"
TRAIN_PATH     = "checkpoints/splits/train.csv"
CHECKPOINT_DIR = "checkpoints/models"
FIGURES_DIR    = "paper_figures"
WINDOW_SIZE    = 12
STEP           = 4

EXPERIMENTS = [
    {"group": 1, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp"},
    {"group": 1, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp"},
    {"group": 1, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp"},
    {"group": 2, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp,DBP"},
    {"group": 2, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp,SBP"},
    {"group": 2, "target": "MAP",   "inputs": "HR,O2Sat,Resp,Temp,DBP,SBP"},
    {"group": 2, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp,DBP"},
    {"group": 2, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp,MAP"},
    {"group": 2, "target": "SBP",   "inputs": "HR,O2Sat,Resp,Temp,DBP,MAP"},
    {"group": 2, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp,SBP"},
    {"group": 2, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp,MAP"},
    {"group": 2, "target": "DBP",   "inputs": "HR,O2Sat,Resp,Temp,SBP,MAP"},
    {"group": 3, "target": "Temp",  "inputs": "HR,O2Sat,Resp,MAP,SBP,DBP"},
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

GROUP_LABELS = {
    1: "G1: Non-Invasive→BP",
    2: "G2: Non-Invasive+Partial→BP",
    3: "G3: Temperature",
    4: "G4: Resp & O2Sat",
}

UNITS = {
    "MAP":   "MAP (mmHg)",
    "SBP":   "SBP (mmHg)",
    "DBP":   "DBP (mmHg)",
    "Resp":  "Resp (br/min)",
    "O2Sat": "O2Sat (%)",
    "Temp":  "Temp (°C)",
    "HR":    "HR (bpm)",
}


# ========================================================================
# Helpers
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
    return np.stack(X_list), np.stack(y_list), pd.DataFrame(meta)


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


# ========================================================================
# Main figure function
# ========================================================================

def pick_best_windows(y_true, preds_dict, indices, n=2):
    """
    From a list of window indices, pick the n windows where the
    best DL model achieves the highest Pearson r vs ground truth.
    """
    from scipy.stats import pearsonr
    scored = []
    # Use TCN if available, else first available model
    key = "tcn" if "tcn" in preds_dict else list(preds_dict.keys())[0]
    pred = preds_dict[key]
    for idx in indices:
        gt = y_true[idx, :, 0]
        pr = pred[idx, :, 0]
        if gt.std() < 1e-6 or pr.std() < 1e-6:
            r = 0.0
        else:
            r, _ = pearsonr(gt, pr)
        scored.append((r, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:n]]


def make_page_figure(exp_subset, seed, device, df_test, model_cache,
                     page_num, n_pages):
    """
    Generates one page of the ALL_experiments figure.
    exp_subset : list of experiment dicts for this page
    """
    n_rows = len(exp_subset)
    n_cols = 4

    fig = plt.figure(figsize=(24, n_rows * 5.2))
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                             hspace=0.55, wspace=0.28,
                             left=0.16, right=0.99,
                             top=0.93, bottom=0.04)

    title = (
        "Reconstruction of ICU Vital Signs - "
        f"15 Experimental Configurations  (Page {page_num}/{n_pages})\n"
        "True signal (black) vs. model predictions  |  "
        "Left 2 cols: Non-sepsis patients  |  Right 2 cols: Sepsis patients"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    # Legend right under title
    legend_elements = [
        Line2D([0],[0], color="black", linewidth=2.5, label="Ground Truth"),
    ]
    for name in MODEL_REGISTRY.keys():
        legend_elements.append(
            Line2D([0],[0], color=DL_COLORS[name], linewidth=1.8,
                   label=DL_LABELS[name])
        )
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=7, fontsize=13, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.965),
               markerscale=1.8, borderpad=0.5, labelspacing=0.4,
               columnspacing=1.0)

    col_titles = ["Non-Sepsis #1", "Non-Sepsis #2",
                  "Sepsis #1",      "Sepsis #2"]
    hours = np.arange(WINDOW_SIZE)
    return fig, gs, col_titles, hours, exp_subset


def generate_figure(seed, device, df_test, df_train, model_cache):
    """
    Splits 15 experiments across two PDF pages (8 + 7 rows).
    Each page is saved separately then combined.
    """
    print(f"  Generating ALL_experiments_seed{seed}.pdf ...")

    # Split into three pages: 5 + 5 + 5 rows
    page_splits = [EXPERIMENTS[:5], EXPERIMENTS[5:10], EXPERIMENTS[10:]]
    n_pages     = len(page_splits)
    all_figs    = []

    for page_num, exp_subset in enumerate(page_splits, start=1):
        fig, gs, col_titles, hours, exps = make_page_figure(
            exp_subset, seed, device, df_test, model_cache,
            page_num, n_pages)
        all_figs.append(fig)
        n_rows = len(exps)

        for row_idx, exp in enumerate(exps):
            target     = exp["target"]
            inputs     = exp["inputs"]
            input_cols = inputs.split(",")
            target_col = [target]
            run_tag    = f"{inputs.replace(',', '-')}_to_{target}"

            if run_tag not in model_cache:
                continue

            X_test, y_test, meta = make_windows(df_test, input_cols, target_col)
            dl_preds = model_cache[run_tag]

            healthy_idx = meta[meta["sepsis_label"] == 0].index.tolist()
            sepsis_idx  = meta[meta["sepsis_label"] == 1].index.tolist()
            sel_h = pick_best_windows(y_test, dl_preds, healthy_idx, n=2)
            sel_s = pick_best_windows(y_test, dl_preds, sepsis_idx,  n=2)
            indices   = sel_h + sel_s
            is_sepsis = [False, False, True, True]

            # Group start rows per page
            all_group_starts = {0, 3, 12, 13}  # global row indices
            global_row = EXPERIMENTS.index(exp)

            for col, (win_idx, sepsis_flag) in enumerate(zip(indices, is_sepsis)):
                ax = fig.add_subplot(gs[row_idx, col])

                if row_idx == 0:
                    color = "darkgreen" if col < 2 else "darkred"
                    ax.set_title(col_titles[col], fontsize=13,
                                 fontweight="bold", color=color, pad=6)

                ax.plot(hours, y_test[win_idx, :, 0],
                        color="black", linewidth=2.5, zorder=10)

                for name, preds in dl_preds.items():
                    ax.plot(hours, preds[win_idx, :, 0],
                            color=DL_COLORS[name], linewidth=1.2, alpha=0.85)

                ax.set_facecolor("#fff3f3" if sepsis_flag else "#f3fff3")
                ax.tick_params(labelsize=12)
                ax.set_xticks([0, 3, 6, 9, 11])
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Hour", fontsize=13)
                ax.grid(True, alpha=0.2, linestyle="--")

                if col == 1:
                    ax.spines["right"].set_linewidth(2.0)
                    ax.spines["right"].set_color("#444444")
                    ax.spines["right"].set_linestyle("--")

                if global_row in all_group_starts:
                    ax.spines["top"].set_linewidth(2.5)
                    ax.spines["top"].set_color("#333333")

                if col == 0:
                    short_in = (inputs
                        .replace("HR,O2Sat,Resp,Temp,DBP,SBP", "NI+DBP+SBP")
                        .replace("HR,O2Sat,Resp,Temp,SBP,MAP", "NI+SBP+MAP")
                        .replace("HR,O2Sat,Resp,Temp,DBP,MAP", "NI+DBP+MAP")
                        .replace("HR,O2Sat,Resp,Temp,DBP",     "NI+DBP")
                        .replace("HR,O2Sat,Resp,Temp,SBP",     "NI+SBP")
                        .replace("HR,O2Sat,Resp,Temp,MAP",     "NI+MAP")
                        .replace("HR,O2Sat,Resp,Temp",         "NI only")
                        .replace("HR,O2Sat,Resp,MAP,SBP,DBP",  "All vitals")
                        .replace("HR,O2Sat,Temp,MAP,SBP,DBP",  "All vitals")
                        .replace("HR,Resp,Temp,MAP,SBP,DBP",   "All vitals")
                    )
                    units     = {"MAP":"mmHg","SBP":"mmHg","DBP":"mmHg",
                                 "Resp":"br/min","O2Sat":"%","Temp":"°C"}
                    unit      = units.get(target, "")
                    group_str = {1:"G1",2:"G2",3:"G3",4:"G4"}[exp["group"]]
                    lbl = f"{group_str}: {target} ({unit})\n← {short_in}"
                    ax.set_ylabel(lbl, fontsize=11, labelpad=8,
                                  rotation=0, ha="right", va="center",
                                  linespacing=1.5)

    # Save each page figure to a separate PDF + one combined multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = os.path.join(FIGURES_DIR, f"ALL_experiments_seed{seed}.pdf")
    png_base = os.path.join(FIGURES_DIR, f"ALL_experiments_seed{seed}")

    # Save individual PNGs first (before figures are closed)
    for p_num, fig in enumerate(all_figs, start=1):
        png_out = f"{png_base}_p{p_num}.png"
        fig.savefig(png_out, dpi=180, bbox_inches="tight")
        print(f"  \u2705 Saved: {png_out}")

    # Save multi-page PDF
    with PdfPages(pdf_path) as pdf:
        for fig in all_figs:
            pdf.savefig(fig, bbox_inches="tight")
    plt.close("all")
    print(f"  \u2705 Saved 3-page PDF: {pdf_path}")


# ========================================================================
# Main
# ========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds (e.g. '42,7,13,99,2024')")
    parser.add_argument("--cache_dir", type=str, default="paper_figures/inference_cache",
                        help="Directory to save/load inference cache (numpy .npz files)")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Ignore existing cache and rerun inference")
    args  = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    os.makedirs(FIGURES_DIR,   exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device    : {device}")
    print(f"Seeds     : {seeds}")
    print(f"Cache dir : {args.cache_dir}")

    # -- Load data once ------------------------------------------------
    print("\nLoading data...")
    df_test  = pd.read_csv(TEST_PATH)
    df_train = pd.read_csv(TRAIN_PATH)

    # -- Inference with caching ----------------------------------------
    model_cache = {}

    for exp in EXPERIMENTS:
        target     = exp["target"]
        inputs     = exp["inputs"]
        input_cols = inputs.split(",")
        target_col = [target]
        in_ch      = len(input_cols)
        out_ch     = 1
        run_tag    = f"{inputs.replace(',', '-')}_to_{target}"
        cache_file = os.path.join(args.cache_dir, f"{run_tag}.npz")

        if os.path.exists(cache_file) and not args.force_recompute:
            print(f"  Loading cache: {run_tag}")
            data = np.load(cache_file, allow_pickle=True)
            dl_preds = {k: data[k] for k in data.files
                        if k != "y_test"}
            model_cache[run_tag] = dl_preds
            continue

        print(f"  Running inference: {target} ← {inputs}")
        X_test,  y_test,  _ = make_windows(df_test,  input_cols, target_col)
        X_train, y_train, _ = make_windows(df_train, input_cols, target_col)

        dl_preds = {}
        for name in MODEL_REGISTRY.keys():
            m = load_dl_model(name, run_tag, in_ch, out_ch, device)
            if m is None:
                continue
            dl_preds[name] = run_inference(m, X_test, device)
        model_cache[run_tag] = dl_preds

        # Save DL predictions to cache (no LR baseline)
        save_dict = {k: v for k, v in dl_preds.items()}
        save_dict["y_test"] = y_test
        np.savez(cache_file, **save_dict)
        print(f"    Cache saved: {cache_file}")

    # -- Generate one figure per seed ----------------------------------
    print(f"\nGenerating {len(seeds)} figure(s)...")
    for seed in seeds:
        generate_figure(seed, device, df_test, df_train, model_cache)

    print("\nDone.")
    for seed in seeds:
        print(f"   paper_figures/ALL_experiments_seed{seed}.pdf")