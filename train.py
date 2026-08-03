"""
train.py
Trains a seq2seq model to reconstruct one or more vital signs from one
or more other vitals. Defaults to predicting MAP (Mean Arterial
Pressure) from HR, O2Sat, Resp, Temp.

Usage:
    python train.py --model tcn --inputs HR,O2Sat,Resp,Temp --targets MAP
    python train.py --model tcn --inputs HR,O2Sat,Resp,Temp --targets MAP --lambda_r 0.1
    python train.py --model tcn --inputs HR,O2Sat,Resp,Temp --targets MAP --demographics Age,ICULOS
    python train.py --model tcn --inputs HR,O2Sat,Resp,Temp --targets MAP --window_size 24
    python train.py --model tcn --inputs HR,O2Sat,Resp,Temp --targets MAP \
        --lambda_r 0.1 --demographics Age,ICULOS --window_size 24
"""
import argparse
import json
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import build_model, MODEL_REGISTRY

# ════════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════════
DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════
TRAIN_PATH  = "checkpoints/splits/train.csv"
VAL_PATH    = "checkpoints/splits/val.csv"
WINDOW_SIZE = 12
STEP        = 4

# ════════════════════════════════════════════════════════════════════════
# Physics-aware loss: MSE + lambda_r * (1 - Pearson_r)
# ════════════════════════════════════════════════════════════════════════
def pearson_r_loss(pred, target):
    """
    Computes 1 - Pearson_r averaged over all output channels and batch.
    pred, target : (B, C_out, T)
    Returns a scalar tensor.
    """
    # Flatten over batch and time for each channel
    B, C, T = pred.shape
    pred_flat   = pred.reshape(B * C, T)
    target_flat = target.reshape(B * C, T)

    pred_m   = pred_flat   - pred_flat.mean(dim=1, keepdim=True)
    target_m = target_flat - target_flat.mean(dim=1, keepdim=True)

    num   = (pred_m * target_m).sum(dim=1)
    denom = pred_m.norm(dim=1) * target_m.norm(dim=1) + 1e-8
    r     = num / denom
    return (1 - r).mean()

# ════════════════════════════════════════════════════════════════════════
# Data: sliding windows + Dataset
# ════════════════════════════════════════════════════════════════════════
def make_windows(df, input_cols, target_cols, demo_cols=None,
                 window_size=WINDOW_SIZE, step=STEP):
    """
    Slices each patient's series into fixed-length windows.
    If demo_cols is provided, demographic values are repeated across
    all T timesteps and appended as extra input channels.
    """
    X_list, y_list = [], []
    n_demo = len(demo_cols) if demo_cols else 0

    for pid, group in df.groupby("Patient_ID"):
        group = group.sort_values("Hour").reset_index(drop=True)
        n = len(group)
        if n < window_size:
            continue

        x_vals = group[input_cols].values.astype(np.float32)
        y_vals = group[target_cols].values.astype(np.float32)

        # Demographics: one value per patient, repeated across T timesteps
        if demo_cols:
            demo_vals = group[demo_cols].iloc[0].values.astype(np.float32)  # (n_demo,)

        for start in range(0, n - window_size + 1, step):
            end   = start + window_size
            x_win = x_vals[start:end]    # (T, C_in)

            if demo_cols:
                # Tile demographics: (T, n_demo)
                demo_tile = np.tile(demo_vals, (window_size, 1))
                x_win     = np.concatenate([x_win, demo_tile], axis=1)

            X_list.append(x_win)
            y_list.append(y_vals[start:end])

    total_in = len(input_cols) + n_demo
    X = np.stack(X_list) if X_list else np.empty((0, window_size, total_in),    dtype=np.float32)
    y = np.stack(y_list) if y_list else np.empty((0, window_size, len(target_cols)), dtype=np.float32)
    return X, y


class VitalsDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float().permute(0, 2, 1)  # (N, C_in, T)
        self.y = torch.from_numpy(y).float().permute(0, 2, 1)  # (N, C_out, T)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ════════════════════════════════════════════════════════════════════════
# Train / validate loop
# ════════════════════════════════════════════════════════════════════════
def run_epoch(model, loader, criterion, optimizer, device,
              train: bool, lambda_r: float = 0.0):
    model.train(mode=train)
    total_loss, total_mae, n_batches = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred     = model(x)
            mse_loss = criterion(pred, y)

            # Physics-aware loss component
            if lambda_r > 0.0:
                loss = mse_loss + lambda_r * pearson_r_loss(pred, y)
            else:
                loss = mse_loss

            mae = torch.mean(torch.abs(pred - y))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += loss.item()
        total_mae  += mae.item()
        n_batches  += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_mae  = total_mae  / max(n_batches, 1)
    avg_rmse = avg_loss ** 0.5
    return avg_loss, avg_mae, avg_rmse

# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def main(args):
    set_seed(args.seed)
    print(f"Seed       : {args.seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")

    input_cols  = args.inputs.split(",")
    target_cols = args.targets.split(",")
    demo_cols   = args.demographics.split(",") if args.demographics else []

    # Normalize demo column names
    demo_cols = [d.strip() for d in demo_cols] if demo_cols else []

    print(f"Inputs       : {input_cols}")
    print(f"Targets      : {target_cols}")
    print(f"Demographics : {demo_cols if demo_cols else 'None'}")
    print(f"Lambda_r     : {args.lambda_r}")
    print(f"Window size  : {args.window_size}")

    total_in_channels = len(input_cols) + len(demo_cols)
    print(f"Total input channels: {total_in_channels}")

    # ── Load data ────────────────────────────────────────────────────────
    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)

    print(f"Train patients: {df_train['Patient_ID'].nunique():,}, rows: {df_train.shape[0]:,}")
    print(f"Val patients  : {df_val['Patient_ID'].nunique():,}, rows: {df_val.shape[0]:,}")

    X_train, y_train = make_windows(df_train, input_cols, target_cols, demo_cols,
                                     window_size=args.window_size, step=args.step)
    X_val,   y_val   = make_windows(df_val,   input_cols, target_cols, demo_cols,
                                     window_size=args.window_size, step=args.step)

    print(f"Train windows: {X_train.shape}, Val windows: {X_val.shape}")

    train_ds     = VitalsDataset(X_train, y_train)
    val_ds       = VitalsDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    # ── Build model ──────────────────────────────────────────────────────
    model = build_model(args.model,
                        in_channels=total_in_channels,
                        out_channels=len(target_cols)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} ({n_params:,} parameters)")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    # ── Checkpoint naming ─────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    target_tag = "-".join(target_cols)
    input_tag  = "-".join(input_cols)
    run_tag    = f"{input_tag}_to_{target_tag}"

    # Append modifiers to run_tag so ablation checkpoints don't overwrite base
    modifiers = []
    if demo_cols:
        modifiers.append(f"demo-{'_'.join(demo_cols)}")
    if args.lambda_r > 0.0:
        modifiers.append(f"lr{args.lambda_r}".replace(".", "p"))
    if args.window_size != WINDOW_SIZE:
        modifiers.append(f"w{args.window_size}")
    if modifiers:
        run_tag = run_tag + "__" + "_".join(modifiers)

    ckpt_path    = os.path.join(args.checkpoint_dir, f"{args.model}_{run_tag}_best.pt")
    history_path = os.path.join(args.checkpoint_dir, f"{args.model}_{run_tag}_history.json")

    print(f"Run tag: {run_tag}")

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch       = 1
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    history           = []

    resume_from = args.resume if args.resume else (ckpt_path if args.auto_resume else None)
    if resume_from and os.path.exists(resume_from):
        print(f"\n🔄 Resuming from: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch       = ckpt["epoch"] + 1
        best_val_loss     = ckpt["best_val_loss"]
        epochs_no_improve = ckpt["epochs_no_improve"]
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
        print(f"   Resuming at epoch {start_epoch} (best: {best_val_loss:.4f})\n")

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_mae, train_rmse = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, lambda_r=args.lambda_r)

        val_loss, val_mae, val_rmse = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False, lambda_r=args.lambda_r)

        scheduler.step(val_loss)

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f} train_MAE={train_mae:.4f} | "
              f"val_loss={val_loss:.4f} val_MAE={val_mae:.4f} val_RMSE={val_rmse:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_MAE": train_mae, "train_RMSE": train_rmse,
            "val_loss":   val_loss,   "val_MAE":   val_mae,   "val_RMSE":  val_rmse,
        })
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        is_best = val_loss < best_val_loss - 1e-5
        if is_best:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss":      best_val_loss,
                "epochs_no_improve":  epochs_no_improve,
                "args":               vars(args),
            }, ckpt_path)
            print(f"   ✅ New best saved (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience and not args.no_early_stopping:
            print(f"Early stopping at epoch {epoch}.")
            break

    print(f"\nDone. Best val_loss : {best_val_loss:.4f}")
    print(f"Run tag             : {run_tag}")
    print(f"Checkpoint saved    : {ckpt_path}")
    print(f"History saved       : {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       type=str,   default="unet1d",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--inputs",      type=str,   default="HR,O2Sat,Resp,Temp")
    parser.add_argument("--targets",     type=str,   default="MAP")
    parser.add_argument("--demographics",type=str,   default=None,
                        help="Comma-separated demographic columns to append as static "
                             "input channels, e.g. 'Age,ICULOS' or "
                             "'Age,Gender,ICULOS,HospAdmTime'")
    parser.add_argument("--lambda_r",   type=float,  default=0.0,
                        help="Weight for the Pearson-r loss term. "
                             "Loss = MSE + lambda_r * (1 - r). "
                             "0.0 = pure MSE (default). Try 0.1 or 1.0.")
    parser.add_argument("--window_size", type=int,   default=WINDOW_SIZE)
    parser.add_argument("--step",        type=int,   default=STEP)
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/models")
    parser.add_argument("--resume",      type=str,   default=None)
    parser.add_argument("--auto_resume", action="store_true")
    args = parser.parse_args()
    main(args)