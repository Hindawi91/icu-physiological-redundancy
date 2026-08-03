"""
plot_reconstruction_results.py

Generates a publication-quality summary figure for all 15 reconstruction
experiments. Two figures are produced:

Figure 1 (fig_reconstruction_overview.pdf):
    Grid of bar charts — one per experiment group.
    X-axis: architectures. Y-axis: Pearson r.
    Clearly labelled titles, axes, and experiment descriptions.

Figure 2 (fig_progressive_sensor.pdf):
    Line plot showing progressive BP reconstruction fidelity
    as sensor availability increases (NI → NI+1BP → NI+2BP).

Usage:
    python plot_reconstruction_results.py \
        --metrics results/ALL_metrics.csv \
        --outdir  paper_figures/
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ════════════════════════════════════════════════════════════════════════
# Style
# ════════════════════════════════════════════════════════════════════════
FONT_SIZE   = 18
TITLE_SIZE  = 18
LABEL_SIZE  = 14
TICK_SIZE   = 16
LEGEND_SIZE = 18

def apply_style():
    """Call at the start of each plot function to apply global style."""
    plt.rcParams.update({
        "font.family":        "DejaVu Sans",
        "font.size":          FONT_SIZE,
        "axes.titlesize":     TITLE_SIZE,
        "axes.labelsize":     LABEL_SIZE,
        "xtick.labelsize":    TICK_SIZE,
        "ytick.labelsize":    TICK_SIZE,
        "legend.fontsize":    LEGEND_SIZE,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linestyle":     "--",
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
    })

# Model display names and colours
MODEL_ORDER = [
    "tcn", "conv_lstm", "bilstm", "lstm_seq2seq",
    "transformer", "unet1d",
    "LinearRegression",
]
MODEL_LABELS = {
    "tcn":             "TCN",
    "conv_lstm":       "Conv-LSTM",
    "bilstm":          "BiLSTM",
    "lstm_seq2seq":    "LSTM S2S",
    "transformer":     "Transformer",
    "unet1d":          "U-Net 1D",
    "LinearRegression":"Linear Reg.",
}
MODEL_COLORS = {
    "tcn":             "#2196F3",
    "conv_lstm":       "#4CAF50",
    "bilstm":          "#FF9800",
    "lstm_seq2seq":    "#9C27B0",
    "transformer":     "#F44336",
    "unet1d":          "#009688",
    "LinearRegression":"#757575",
}

# Experiment configurations
EXPERIMENTS = [
    # (label, group_title, experiment_str)
    ("G1: MAP\n← NI only",
     "Group 1 — Fully Non-Invasive → Blood Pressure",
     "MAP ← HR,O2Sat,Resp,Temp"),
    ("G1: SBP\n← NI only",
     "Group 1 — Fully Non-Invasive → Blood Pressure",
     "SBP ← HR,O2Sat,Resp,Temp"),
    ("G1: DBP\n← NI only",
     "Group 1 — Fully Non-Invasive → Blood Pressure",
     "DBP ← HR,O2Sat,Resp,Temp"),
    ("G2: MAP\n← NI+DBP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "MAP ← HR,O2Sat,Resp,Temp,DBP"),
    ("G2: MAP\n← NI+SBP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "MAP ← HR,O2Sat,Resp,Temp,SBP"),
    ("G2: MAP\n← NI+DBP+SBP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "MAP ← HR,O2Sat,Resp,Temp,DBP,SBP"),
    ("G2: SBP\n← NI+DBP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "SBP ← HR,O2Sat,Resp,Temp,DBP"),
    ("G2: SBP\n← NI+MAP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "SBP ← HR,O2Sat,Resp,Temp,MAP"),
    ("G2: SBP\n← NI+DBP+MAP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "SBP ← HR,O2Sat,Resp,Temp,DBP,MAP"),
    ("G2: DBP\n← NI+SBP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "DBP ← HR,O2Sat,Resp,Temp,SBP"),
    ("G2: DBP\n← NI+MAP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "DBP ← HR,O2Sat,Resp,Temp,MAP"),
    ("G2: DBP\n← NI+SBP+MAP",
     "Group 2 — Partial BP Sensor → Missing BP",
     "DBP ← HR,O2Sat,Resp,Temp,SBP,MAP"),
    ("G3: Temp\n← 6 vitals",
     "Group 3 — Temperature",
     "Temp ← HR,O2Sat,Resp,MAP,SBP,DBP"),
    ("G4: Resp\n← 6 vitals",
     "Group 4 — Resp & O₂Sat",
     "Resp ← HR,O2Sat,Temp,MAP,SBP,DBP"),
    ("G4: O₂Sat\n← 6 vitals",
     "Group 4 — Resp & O₂Sat",
     "O2Sat ← HR,Resp,Temp,MAP,SBP,DBP"),
]


def load_data(metrics_path):
    df = pd.read_csv(metrics_path)
    # Normalise experiment strings
    df["Experiment"] = df["Experiment"].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


# ════════════════════════════════════════════════════════════════════════
# Figure 2 — Progressive sensor availability (all 3 targets on one plot)
# ════════════════════════════════════════════════════════════════════════
def plot_progressive(df, outdir):
    """
    One line per target (MAP/SBP/DBP) connecting three points:
      x=0: NI only
      x=1: best single-sensor config (highest r)
      x=2: dual-sensor config

    All other single-sensor configs plotted as hollow markers at x=1
    but NOT connected — the main line uses only the best single config.

    r= labels stacked vertically below the lowest point at each x,
    aligned in a column for readability.
    """

    # Best single-sensor configs per target
    best_single = {
        "MAP": ("NI + DBP",     0.8616),
        "SBP": ("NI + MAP",     0.8051),
        "DBP": ("NI + MAP",     0.8607),
    }
    alt_single = {
        "MAP": ("NI + SBP",     0.7943),
        "SBP": ("NI + DBP",     0.6026),
        "DBP": ("NI + SBP",     0.5986),
    }
    dual = {
        "MAP": ("NI + DBP+SBP", 0.9363),
        "SBP": ("NI + DBP+MAP", 0.8862),
        "DBP": ("NI + SBP+MAP", 0.9151),
    }
    ni_only = {
        "MAP": 0.2252,
        "SBP": 0.2306,
        "DBP": 0.2730,
    }

    targets = ["MAP", "SBP", "DBP"]
    colors  = {"MAP": "#1565C0", "SBP": "#2E7D32", "DBP": "#6A1B9A"}
    markers = {"MAP": "o",       "SBP": "s",        "DBP": "^"}
    LWIDTH  = 3.0
    MSIZE   = 160

    apply_style()
    fig, ax = plt.subplots(figsize=(13, 8))

    # ── Draw main lines and markers ───────────────────────────────────
    for target in targets:
        r0  = ni_only[target]
        r1b = best_single[target][1]
        r1a = alt_single[target][1]
        r2  = dual[target][1]

        # Main line: x=0 → x=1 (best) → x=2
        ax.plot([0, 1, 2], [r0, r1b, r2],
                color=colors[target], linewidth=LWIDTH,
                alpha=0.9, zorder=2)

        # Main markers (filled)
        for xi, yi in [(0, r0), (1, r1b), (2, r2)]:
            ax.scatter(xi, yi, color=colors[target],
                       marker=markers[target], s=MSIZE, zorder=4,
                       edgecolors="white", linewidths=1.2)

        # Alt single-sensor: hollow marker at x=1, not on line
        ax.scatter(1, r1a, color="none",
                   marker=markers[target], s=MSIZE, zorder=4,
                   edgecolors=colors[target], linewidths=2.0)

        # Target label at right end — stored for later vertical spacing
        pass  # labels drawn after loop

    # ── Target labels at right end with vertical spacing ─────────────
    right_labels = sorted([(dual[t][1], t) for t in targets], key=lambda x: x[0])
    used_ys, min_gap = [], 0.04
    for r2, target in right_labels:
        y_pos = r2
        for prev_y in used_ys:
            if abs(y_pos - prev_y) < min_gap:
                y_pos = prev_y + min_gap
        used_ys.append(y_pos)
        ax.text(2.08, y_pos, target, fontsize=LABEL_SIZE, fontweight="bold",
                color=colors[target], va="center")

    # Source labels for each config
    # Format: (r_value, target, source_description)
    x_entries = {
        0: [
            (ni_only["MAP"], "MAP", "NI only"),
            (ni_only["SBP"], "SBP", "NI only"),
            (ni_only["DBP"], "DBP", "NI only"),
        ],
        1: [
            (best_single["MAP"][1], "MAP", "MAP←NI+DBP"),
            (best_single["SBP"][1], "SBP", "SBP←NI+MAP"),
            (best_single["DBP"][1], "DBP", "DBP←NI+MAP"),
            (alt_single["MAP"][1],  "MAP", "MAP←NI+SBP"),
            (alt_single["SBP"][1],  "SBP", "SBP←NI+DBP"),
            (alt_single["DBP"][1],  "DBP", "DBP←NI+SBP"),
        ],
        2: [
            (dual["MAP"][1], "MAP", "MAP←NI+DBP+SBP"),
            (dual["SBP"][1], "SBP", "SBP←NI+DBP+MAP"),
            (dual["DBP"][1], "DBP", "DBP←NI+SBP+MAP"),
        ],
    }

    label_gap    = 0.048   # vertical gap between stacked labels
    below_margin = 0.065   # gap below lowest data point

    for xi, entries in x_entries.items():
        min_r = min(r for r, t, src in entries)
        # Sort ascending so smallest r is at bottom
        entries_sorted = sorted(entries, key=lambda x: x[0])
        y_start = min_r - below_margin
        for i, (r_val, target, src) in enumerate(entries_sorted):
            y_lbl = y_start - i * label_gap
            ax.text(xi, y_lbl, f"r={r_val:.3f}  {src}",
                    ha="center", va="top",
                    fontsize=LABEL_SIZE, fontweight="bold",
                    color=colors[target])

    # ── Threshold and shading ─────────────────────────────────────────
    # Background shading removed — no threshold line

    # ── Axes ──────────────────────────────────────────────────────────
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(
        ["No BP sensor\n(NI vitals only)",
         "1 additional\nBP sensor",
         "2 additional\nBP sensors"],
        fontsize=TICK_SIZE)
    ax.set_xlim(-0.5, 2.75)
    ax.set_ylim(-0.32, 1.08)
    ax.set_ylabel("Pearson Correlation ($r$)", fontsize=FONT_SIZE, fontweight="bold")
    ax.set_xlabel("Sensor availability", fontsize=FONT_SIZE, fontweight="bold")
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_title(
        "Progressive Blood Pressure Reconstruction Fidelity\n"
        "vs. Available Sensor Combinations\n"
        "(NI = HR, O\u2082Sat, Resp, Temp)",
        fontsize=TITLE_SIZE, fontweight="bold")

    # ── Legend (lower right) ──────────────────────────────────────────
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    handles = [
        Line2D([0],[0], color=colors[t], marker=markers[t],
               markersize=9, linewidth=2,
               label=f"{t} (primary input combination)")
        for t in targets
    ] + [
        Line2D([0],[0], color=colors[t], marker=markers[t],
               markersize=9, linewidth=0,
               markerfacecolor="none", markeredgewidth=1.5,
               label=f"{t} (secondary input combination)")
        for t in targets
    ]
    ax.legend(handles=handles, fontsize=LEGEND_SIZE, loc="lower right",
              framealpha=0.92, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(outdir, "fig_progressive_sensor.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"✅ Saved: {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# Figure 3 — Clinical utility MAP (vertical layout, progressive shading)
# ════════════════════════════════════════════════════════════════════════
def plot_clinical_utility(outdir):

    conditions = [
        "Full\nsupervision\n(true MAP)",
        "MAP\nremoved",
        "G1 recon\n(r=0.225)\nNI only",
        "G2+ recon\n(r=0.862)\nNI+DBP",
        "G2++ recon\n(r=0.936)\nNI+DBP+SBP",
    ]
    aurocs   = [0.710, 0.668, 0.676, 0.708, 0.715]
    recovery = [None,  None,  20.2,  95.1,  110.8]

    # Colors: full=blue, removed=red, recons=progressively darker greens
    colors = [
        "#1565C0",   # Full — blue
        "#C62828",   # Removed — red
        "#A5D6A7",   # G1 — light green
        "#43A047",   # G2+ — medium green
        "#1B5E20",   # G2++ — dark green
    ]

    apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12),
                                    gridspec_kw={"hspace": 0.55})
    fig.suptitle(
        "Clinical Utility of MAP Reconstruction\n"
        "for Downstream Sepsis Classification",
        fontsize=14, fontweight="bold"
    )

    # ── Top panel: AUROC ─────────────────────────────────────────────
    x     = np.arange(len(conditions))
    bars  = ax1.bar(x, aurocs, color=colors, edgecolor="white",
                    linewidth=0.8, width=0.6)

    ax1.axhline(aurocs[0], color="#1565C0", linewidth=1.5,
                linestyle="--", alpha=0.7, label="Full supervision (ceiling)")
    ax1.axhline(aurocs[1], color="#C62828", linewidth=1.5,
                linestyle=":",  alpha=0.7, label="MAP removed (floor)")

    for bar, val in zip(bars, aurocs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=10, ha="center")
    ax1.set_ylim(0.62, 0.74)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title("Sepsis Classification AUROC Under Different MAP Conditions",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, loc="lower right", framealpha=0.9)

    # Bracket annotation showing G1→G2+→G2++ progression
    for i, (xi, val, rec) in enumerate(zip(x[2:], aurocs[2:], recovery[2:])):
        shade = ["Light", "Medium", "Dark"][i]
        ax1.annotate(f"G{i+1} recon",
                     xy=(xi, val), xytext=(xi, 0.628),
                     fontsize=8, ha="center", color=colors[i+2],
                     fontweight="bold",
                     arrowprops=dict(arrowstyle="-", color=colors[i+2],
                                     lw=1.2, linestyle="dashed"))

    # ── Bottom panel: Recovery % ──────────────────────────────────────
    rec_conditions = [
        "G1 recon\n(r=0.225)\nNI only",
        "G2+ recon\n(r=0.862)\nNI+DBP",
        "G2++ recon\n(r=0.936)\nNI+DBP+SBP",
    ]
    rec_vals   = [v for v in recovery if v is not None]
    rec_labels = rec_conditions
    rec_colors = [col for col, v in zip(colors, recovery) if v is not None]
    x2 = np.arange(len(rec_vals))

    bars2 = ax2.bar(x2, rec_vals, color=rec_colors,
                    edgecolor="white", linewidth=0.8, width=0.5)
    ax2.axhline(100, color="#1B5E20", linewidth=1.5,
                linestyle="--", alpha=0.8, label="Full recovery (100%)")

    for bar, val in zip(bars2, rec_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 2,
                 f"{val:.1f}%", ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(rec_labels, fontsize=10)
    ax2.set_ylim(0, 130)
    ax2.set_ylabel("AUROC Recovery (%)", fontsize=12)
    ax2.set_title(
        "Percentage of Lost AUROC Recovered by MAP Reconstruction\n"
        "(Progression: G1 light green → G2+ medium → G2++ dark green)",
        fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    out_path = os.path.join(outdir, "fig_clinical_utility_map.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"✅ Saved: {out_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str,
                        default="results/ALL_metrics.csv")
    parser.add_argument("--outdir",  type=str,
                        default="paper_figures/")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.metrics)
    print(f"Loaded {len(df)} rows from {args.metrics}")

    # Figure 1 removed
    plot_progressive(df, args.outdir)
    plot_clinical_utility(args.outdir)

    print("\nAll figures saved to:", args.outdir)