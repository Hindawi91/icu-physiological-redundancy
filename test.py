"""
test.py
Evaluates a trained model on the test set. Mirrors train.py structure
with the same CLI args so experiments are easy to define and run.
Also includes simple baselines for direct comparison in the same run.

For multi-target experiments, per-channel metrics are also computed
and saved separately for fair comparison with single-target models.
"""
import argparse
import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
RESULTS_DIR    = "results"
WINDOW_SIZE    = 12
STEP           = 4
DEFAULT_SEED   = 42

MODEL_COLORS = {
    "unet1d":           "#2196F3",
    "bilstm":           "#F44336",
    "lstm_seq2seq":     "#FF9800",
    "tcn":              "#4CAF50",
    "transformer":      "#9C27B0",
    "conv_lstm":        "#00BCD4",
    "MeanImputation":   "#9E9E9E",
    "TargetMean":       "#607D8B",
    "ForwardFill":      "#FF5722",
    "LinearInterp":     "#795548",
    "LinearRegression": "#FFEB3B",
}
LINE_STYLES = {
    "unet1d":           "-",  "bilstm":           "-",
    "lstm_seq2seq":     "-",  "tcn":              "-",
    "transformer":      "-",  "conv_lstm":        "-",
    "MeanImputation":   "--", "TargetMean":       "--",
    "ForwardFill":      "--", "LinearInterp":     "--",
    "LinearRegression": "--",
}

# ========================================================================
# Reproducibility
# ========================================================================
def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ========================================================================
# Data
# ========================================================================
def make_windows_with_meta(df, input_cols, target_cols, demo_cols=None,
                            window_size=WINDOW_SIZE, step=STEP):
    X_list, y_list, meta = [], [], []
    n_demo = len(demo_cols) if demo_cols else 0
    for pid, group in df.groupby("Patient_ID"):
        group  = group.sort_values("Hour").reset_index(drop=True)
        n      = len(group)
        if n < window_size:
            continue
        x_vals = group[input_cols].values.astype(np.float32)
        y_vals = group[target_cols].values.astype(np.float32)
        sepsis = int(group["SepsisLabel"].max())
        if demo_cols:
            demo_vals = group[demo_cols].iloc[0].values.astype(np.float32)
        for start in range(0, n - window_size + 1, step):
            end   = start + window_size
            x_win = x_vals[start:end]
            if demo_cols:
                demo_tile = np.tile(demo_vals, (window_size, 1))
                x_win     = np.concatenate([x_win, demo_tile], axis=1)
            X_list.append(x_win)
            y_list.append(y_vals[start:end])
            meta.append({
                "Patient_ID":   pid,
                "start_hour":   int(group["Hour"].iloc[start]),
                "sepsis_label": sepsis,
            })
    total_in = len(input_cols) + n_demo
    X    = np.stack(X_list) if X_list else np.empty((0, window_size, total_in))
    y    = np.stack(y_list) if y_list else np.empty((0, window_size, len(target_cols)))
    meta = pd.DataFrame(meta)
    return X, y, meta

# ========================================================================
# Metrics
# ========================================================================
def compute_metrics(pred, true):
    """Compute metrics over all channels flattened together."""
    pred = pred.flatten()
    true = true.flatten()
    mse  = float(np.mean((pred - true) ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(pred - true)))
    mape = float(np.mean(np.abs((pred - true) / (np.abs(true) + 1e-6))) * 100)
    r    = float(np.corrcoef(pred, true)[0, 1]) \
           if pred.std() > 0 and true.std() > 0 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "Pearson_r": r}


def compute_per_channel_metrics(pred, true, channel_names):
    """
    Compute metrics separately for each output channel.
    pred, true : (N, T, C_out)
    Returns a dict: {channel_name: {metric: value}}
    """
    results = {}
    for c_idx, c_name in enumerate(channel_names):
        p = pred[:, :, c_idx].flatten()
        t = true[:, :, c_idx].flatten()
        mse  = float(np.mean((p - t) ** 2))
        rmse = float(np.sqrt(mse))
        mae  = float(np.mean(np.abs(p - t)))
        mape = float(np.mean(np.abs((p - t) / (np.abs(t) + 1e-6))) * 100)
        r    = float(np.corrcoef(p, t)[0, 1]) \
               if p.std() > 0 and t.std() > 0 else float("nan")
        results[c_name] = {
            "MSE": mse, "RMSE": rmse, "MAE": mae,
            "MAPE": mape, "Pearson_r": r
        }
    return results


def compute_per_patient_metrics(preds, y_true, meta):
    per_patient = {}
    for win_idx, row in meta.iterrows():
        pid = row["Patient_ID"]
        if pid not in per_patient:
            per_patient[pid] = {"pred": [], "true": []}
        per_patient[pid]["pred"].append(preds[win_idx])
        per_patient[pid]["true"].append(y_true[win_idx])
    results = []
    for data in per_patient.values():
        p = np.concatenate(data["pred"], axis=0)
        t = np.concatenate(data["true"], axis=0)
        results.append(compute_metrics(p, t))
    return results

# ========================================================================
# Deep learning inference
# ========================================================================
def load_model(model_name, run_tag, in_ch, out_ch, device):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_{run_tag}_best.pt")
    if not os.path.exists(ckpt_path):
        return None
    model = build_model(model_name, in_channels=in_ch, out_channels=out_ch)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def run_dl_inference(model, X, device, batch_size=256):
    X_t    = torch.from_numpy(X).float().permute(0, 2, 1)
    loader = DataLoader(torch.utils.data.TensorDataset(X_t),
                        batch_size=batch_size, shuffle=False)
    preds  = []
    with torch.no_grad():
        for (x,) in loader:
            preds.append(model(x.to(device)).cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds.transpose(0, 2, 1)   # (N, T, C_out)

# ========================================================================
# Simple baselines
# ========================================================================
class NumpyLinearRegression:
    def fit(self, X, y):
        X_b = np.hstack([X, np.ones((X.shape[0], 1))])
        result, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
        self.coef_      = result[:-1].T
        self.intercept_ = result[-1]
        return self
    def predict(self, X):
        return X @ self.coef_.T + self.intercept_


def run_baselines(X_train, y_train, X_test, y_test):
    train_mean = float(y_train.mean())
    N, T, C    = X_test.shape
    out_ch     = y_test.shape[2]
    lr_model   = NumpyLinearRegression()
    lr_model.fit(X_train.reshape(-1, C), y_train.reshape(-1, out_ch))
    return {
        "MeanImputation":
            X_test.mean(axis=2, keepdims=True).repeat(out_ch, axis=2),
        "TargetMean":
            np.full((N, T, out_ch), train_mean, dtype=np.float32),
        "ForwardFill":
            np.repeat(X_test[:, 0, :].mean(axis=1)[:, np.newaxis, np.newaxis],
                      T, axis=1).repeat(out_ch, axis=2),
        "LinearInterp": (lambda sv, ev:
            (sv[:, None] + (ev - sv)[:, None] *
             np.linspace(0, 1, T)[None, :])[:, :, np.newaxis]
            .repeat(out_ch, axis=2)
        )(X_test[:, 0, :].mean(axis=1), X_test[:, -1, :].mean(axis=1)),
        "LinearRegression":
            lr_model.predict(X_test.reshape(-1, C)).reshape(N, T, out_ch),
    }

# ========================================================================
# Print and save metrics
# ========================================================================
def print_and_save_metrics(all_metrics, all_preds, y_test, meta,
                            target_cols, target_tag, run_tag):
    metric_keys = ["MSE", "RMSE", "MAE", "MAPE", "Pearson_r"]
    is_multitarget = len(target_cols) > 1

    # -- Window-level (all channels combined) -----------------------------
    df = pd.DataFrame(all_metrics).T[metric_keys].round(4).sort_values("RMSE")
    print(f"\n{'='*65}")
    print(f"  Window-Level Metrics - Target: {target_tag}")
    print(f"  {'(combined across all channels)' if is_multitarget else ''}")
    print(f"{'='*65}")
    print(df.to_string())
    df.to_csv(os.path.join(RESULTS_DIR, f"metrics_{run_tag}.csv"))

    # -- Per-channel metrics (only for multi-target) -----------------------
    if is_multitarget:
        print(f"\n{'='*65}")
        print(f"  Per-Channel Metrics - Target: {target_tag}")
        print(f"{'='*65}")

        per_channel_rows = []
        for name, preds in all_preds.items():
            ch_metrics = compute_per_channel_metrics(preds, y_test, target_cols)
            for ch_name, m in ch_metrics.items():
                row = {"Model": name, "Channel": ch_name}
                row.update(m)
                per_channel_rows.append(row)
                print(f"  {name:<20} [{ch_name}]  "
                      f"RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
                      f"MAPE={m['MAPE']:.2f}%  r={m['Pearson_r']:.4f}")

        ch_df = pd.DataFrame(per_channel_rows)
        ch_df.to_csv(
            os.path.join(RESULTS_DIR, f"metrics_per_channel_{run_tag}.csv"),
            index=False)
        print(f"\nPer-channel metrics saved: "
              f"results/metrics_per_channel_{run_tag}.csv")

    # -- Per-patient (combined) --------------------------------------------
    summary_rows = []
    for name, preds in all_preds.items():
        results = compute_per_patient_metrics(preds, y_test, meta)
        row = {"Model": name}
        for k in metric_keys:
            vals = [r[k] for r in results if not np.isnan(r[k])]
            row[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{k}_std"]  = float(np.std(vals))  if vals else float("nan")
        summary_rows.append(row)

    pat_df = pd.DataFrame(summary_rows).set_index("Model").sort_values("RMSE_mean")
    print(f"\n{'='*65}")
    print(f"  Per-Patient Metrics (mean ± std) - Target: {target_tag}")
    print(f"{'='*65}")
    print(f"\n{'Model':<22}", end="")
    for k in metric_keys:
        print(f"  {k:>18}", end="")
    print()
    print("-" * (22 + 20 * len(metric_keys)))
    for model_name, row in pat_df.iterrows():
        print(f"{model_name:<22}", end="")
        for k in metric_keys:
            print(f"  {row[f'{k}_mean']:>8.4f}±{row[f'{k}_std']:<8.4f}", end="")
        print()
    pat_df.to_csv(os.path.join(RESULTS_DIR,
                               f"metrics_per_patient_{run_tag}.csv"))

    # -- Subgroup analysis -------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  Subgroup Analysis (Sepsis vs Healthy) - Target: {target_tag}")
    print(f"{'='*65}")
    subgroup_rows = []
    for label_sg, flag in [("Healthy", 0), ("Sepsis", 1)]:
        idx = meta[meta["sepsis_label"] == flag].index.tolist()
        if not idx:
            continue
        print(f"\n  {label_sg} ({len(idx)} windows):")
        for name, preds in all_preds.items():
            m = compute_metrics(preds[idx], y_test[idx])
            m["Model"] = name; m["Subgroup"] = label_sg
            subgroup_rows.append(m)
            print(f"    {name:<22} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
                  f"MAPE={m['MAPE']:.2f}%  r={m['Pearson_r']:.4f}")

    pd.DataFrame(subgroup_rows).to_csv(
        os.path.join(RESULTS_DIR, f"metrics_subgroup_{run_tag}.csv"),
        index=False)
    print(f"\nMetrics saved to results/")

# ========================================================================
# Visualization
# ========================================================================
def plot_samples(y_true, all_preds, meta, target_cols, target_tag, run_tag,
                 n_samples=9):
    hours = np.arange(WINDOW_SIZE)
    healthy_idx = meta[meta["sepsis_label"] == 0].index.tolist()
    sepsis_idx  = meta[meta["sepsis_label"] == 1].index.tolist()
    random.seed(DEFAULT_SEED)
    sel_healthy = random.sample(healthy_idx, min(n_samples, len(healthy_idx)))
    sel_sepsis  = random.sample(sepsis_idx,  min(n_samples, len(sepsis_idx)))

    for t_idx, t_name in enumerate(target_cols):
        fig = plt.figure(figsize=(36, 18))
        fig.suptitle(
            f"{t_name} Reconstruction - All Models vs Ground Truth\n"
            f"Solid: Deep Learning  |  Dashed: Simple Baselines\n"
            f"Left: Healthy  |  Right: Sepsis",
            fontsize=14, fontweight="bold", y=1.02
        )
        gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.50, wspace=0.35)
        for panel_idx, (indices, label) in enumerate([
            (sel_healthy, "Healthy"), (sel_sepsis, "Sepsis")
        ]):
            for i, win_idx in enumerate(indices):
                row = i // 3
                col = (i % 3) + (panel_idx * 3)
                ax  = fig.add_subplot(gs[row, col])
                pid        = meta.loc[win_idx, "Patient_ID"]
                start_hour = meta.loc[win_idx, "start_hour"]
                true_vals  = y_true[win_idx, :, t_idx]
                ax.plot(hours, true_vals, color="black", linewidth=2.5,
                        label="Ground Truth", zorder=10)
                for name, preds in all_preds.items():
                    ax.plot(hours, preds[win_idx, :, t_idx],
                            color=MODEL_COLORS.get(name, "gray"),
                            linewidth=1.0, alpha=0.85,
                            linestyle=LINE_STYLES.get(name, "-"),
                            label=name)
                ax.set_facecolor("#fff5f5" if label == "Sepsis" else "#f5fff5")
                ax.set_title(f"{label} | P:{pid} | Hr:{start_hour}",
                             fontsize=7, fontweight="bold",
                             color="red" if label == "Sepsis" else "green")
                ax.set_xlabel("Hour", fontsize=7)
                ax.set_ylabel(t_name, fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)
                if i == 0 and panel_idx == 0:
                    ax.legend(fontsize=5.5, loc="upper right",
                              framealpha=0.8, ncol=1)
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"visual_{t_name}_{run_tag}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visual saved to {save_path}")

# ========================================================================
# Main
# ========================================================================
def main(args):
    set_seed(DEFAULT_SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    input_cols  = args.inputs.split(",")
    target_cols = args.targets.split(",")
    demo_cols   = [d.strip() for d in args.demographics.split(",")]                   if args.demographics else []
    target_tag  = "-".join(target_cols)
    input_tag   = "-".join(input_cols)
    run_tag     = f"{input_tag}_to_{target_tag}"

    # Append modifier suffix to match training checkpoint naming
    modifiers = []
    if demo_cols:
        modifiers.append(f"demo-{'_'.join(demo_cols)}")
    if args.lambda_r > 0.0:
        modifiers.append(f"lr{args.lambda_r}".replace(".", "p"))
    if args.window_size != WINDOW_SIZE:
        modifiers.append(f"w{args.window_size}")
    if modifiers:
        run_tag = run_tag + "__" + "_".join(modifiers)

    in_ch  = len(input_cols) + len(demo_cols)
    out_ch = len(target_cols)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice       : {device}")
    print(f"Inputs       : {input_cols}")
    print(f"Targets      : {target_cols}")
    print(f"Demographics : {demo_cols if demo_cols else 'None'}")
    print(f"Lambda_r     : {args.lambda_r}")
    print(f"Window size  : {args.window_size}")
    print(f"Run tag      : {run_tag}")
    if out_ch > 1:
        print(f"Multi-target : per-channel metrics will also be saved")

    # -- Load data ---------------------------------------------------------
    df_train = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)

    X_train, y_train, _    = make_windows_with_meta(df_train, input_cols, target_cols, demo_cols,
                                                      window_size=args.window_size)
    X_test,  y_test,  meta = make_windows_with_meta(df_test,  input_cols, target_cols, demo_cols,
                                                      window_size=args.window_size)

    print(f"Test windows  : {X_test.shape[0]:,}")
    print(f"Test patients : {meta['Patient_ID'].nunique()}\n")

    all_preds   = {}
    all_metrics = {}

    # -- Deep learning models ----------------------------------------------
    if not args.baselines_only:
        models_to_run = list(MODEL_REGISTRY.keys()) if args.all_models \
                        else [args.model]
        print("-- Deep Learning Models -------------------------------------")
        for name in models_to_run:
            model = load_model(name, run_tag, in_ch, out_ch, device)
            if model is None:
                print(f"   {name}: checkpoint not found - skipping")
                continue
            preds = run_dl_inference(model, X_test, device)
            all_preds[name]   = preds
            all_metrics[name] = compute_metrics(preds, y_test)
            m = all_metrics[name]
            print(f"   {name:<20} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
                  f"MAPE={m['MAPE']:.2f}%  r={m['Pearson_r']:.4f}")

    # -- Simple baselines --------------------------------------------------
    if not args.no_baselines:
        print("\n-- Simple Baselines -----------------------------------------")
        for name, preds in run_baselines(X_train, y_train, X_test, y_test).items():
            all_preds[name]   = preds
            all_metrics[name] = compute_metrics(preds, y_test)
            m = all_metrics[name]
            print(f"   {name:<20} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
                  f"MAPE={m['MAPE']:.2f}%  r={m['Pearson_r']:.4f}")

    # -- Metrics tables ----------------------------------------------------
    print_and_save_metrics(all_metrics, all_preds, y_test, meta,
                            target_cols, target_tag, run_tag)

    # -- Visual ------------------------------------------------------------
    if not args.no_plots:
        plot_samples(y_test, all_preds, meta, target_cols, target_tag, run_tag)

    print(f"\nDone. Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",          type=str, default="unet1d",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--all_models",     action="store_true")
    parser.add_argument("--baselines_only", action="store_true")
    parser.add_argument("--no_baselines",   action="store_true")
    parser.add_argument("--no_plots",       action="store_true")
    parser.add_argument("--inputs",         type=str, default="HR,O2Sat,Resp,Temp")
    parser.add_argument("--targets",        type=str, default="MAP")
    parser.add_argument("--demographics",   type=str, default=None,
                        help="Comma-separated demographic columns, e.g. 'Age,ICULOS'")
    parser.add_argument("--lambda_r",       type=float, default=0.0,
                        help="Used only for run_tag construction to match checkpoint name")
    parser.add_argument("--window_size",    type=int, default=WINDOW_SIZE)
    parser.add_argument("--step",           type=int, default=STEP)
    args = parser.parse_args()
    main(args)