"""
classification_models.py

Purpose-built classifiers for sepsis risk prediction from ICU vital sign windows.

All models share the same interface:
    forward(x, demo=None) -> logits (B,)
    x    : (B, n_vitals, T)  — vital sign window
    demo : (B, n_demo)       — static demographics (optional)

Models implemented:
    1. TCNClassifier      — dilated TCN + last timestep + MLP head
    2. LSTMClassifier     — bidirectional LSTM + last timestep + MLP head
    3. TransformerClassifier — Transformer encoder + CLS token + MLP head
    4. ResNetClassifier   — 1D ResNet + global average pool + MLP head
    5. InceptionClassifier — 1D Inception + global average pool + MLP head
    6. ROCKETClassifier   — ROCKET random convolutions + logistic regression head
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════
# Shared MLP head (used by all models)
# ════════════════════════════════════════════════════════════════════════

class MLPHead(nn.Module):
    """
    2-layer MLP classification head.
    Accepts concatenated [backbone_output, demographics] as input.
    """
    def __init__(self, in_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════
# 1. TCN Classifier
# ════════════════════════════════════════════════════════════════════════

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        pad            = (kernel_size - 1) * dilation // 2
        self.conv1     = nn.Conv1d(in_ch,  out_ch, kernel_size,
                                   padding=pad, dilation=dilation)
        self.bn1       = nn.BatchNorm1d(out_ch)
        self.conv2     = nn.Conv1d(out_ch, out_ch, kernel_size,
                                   padding=pad, dilation=dilation)
        self.bn2       = nn.BatchNorm1d(out_ch)
        self.act       = nn.GELU()
        self.dropout   = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.act(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        if out.shape[-1] != res.shape[-1]:
            m = min(out.shape[-1], res.shape[-1])
            out, res = out[..., :m], res[..., :m]
        return self.act(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, n_vitals=7, n_demo=0,
                 channels=(64, 64, 128, 128), kernel_size=5, dropout=0.3):
        super().__init__()
        layers, ch = [], n_vitals
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(ch, out_ch, kernel_size, 2**i, dropout))
            ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.head     = MLPHead(ch + n_demo, hidden_dim=64, dropout=dropout)
        self.n_demo   = n_demo

    def forward(self, x, demo=None):
        feat = self.backbone(x)[:, :, -1]   # last timestep (B, ch)
        if self.n_demo > 0 and demo is not None:
            feat = torch.cat([feat, demo], dim=1)
        return self.head(feat)


# ════════════════════════════════════════════════════════════════════════
# 2. LSTM Classifier
# ════════════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    def __init__(self, n_vitals=7, n_demo=0, hidden_size=128,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_vitals,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            bidirectional = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        lstm_out  = hidden_size * 2   # bidirectional
        self.head = MLPHead(lstm_out + n_demo, hidden_dim=64, dropout=dropout)
        self.n_demo = n_demo

    def forward(self, x, demo=None):
        x    = x.permute(0, 2, 1)          # (B, T, C)
        out, (h, _) = self.lstm(x)
        # Concatenate last forward + last backward hidden states
        feat = torch.cat([h[-2], h[-1]], dim=1)   # (B, hidden*2)
        if self.n_demo > 0 and demo is not None:
            feat = torch.cat([feat, demo], dim=1)
        return self.head(feat)


# ════════════════════════════════════════════════════════════════════════
# 3. Transformer Classifier
# ════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe           = torch.zeros(max_len, d_model)
        pos          = torch.arange(0, max_len).unsqueeze(1).float()
        div          = torch.exp(torch.arange(0, d_model, 2).float() *
                                  (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerClassifier(nn.Module):
    def __init__(self, n_vitals=7, n_demo=0, d_model=64, nhead=4,
                 num_layers=3, dim_ff=256, dropout=0.3):
        super().__init__()
        self.proj    = nn.Linear(n_vitals, d_model)
        self.cls     = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head    = MLPHead(d_model + n_demo, hidden_dim=64, dropout=dropout)
        self.n_demo  = n_demo

    def forward(self, x, demo=None):
        x    = x.permute(0, 2, 1)                     # (B, T, C)
        x    = self.proj(x)                             # (B, T, d_model)
        cls  = self.cls.expand(x.size(0), -1, -1)      # (B, 1, d_model)
        x    = torch.cat([cls, x], dim=1)               # (B, T+1, d_model)
        x    = self.pos_enc(x)
        x    = self.encoder(x)
        feat = x[:, 0]                                  # CLS token
        if self.n_demo > 0 and demo is not None:
            feat = torch.cat([feat, demo], dim=1)
        return self.head(feat)


# ════════════════════════════════════════════════════════════════════════
# 4. ResNet Classifier (1D)
# ════════════════════════════════════════════════════════════════════════

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)
        self.skip  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + self.skip(x))


class ResNetClassifier(nn.Module):
    def __init__(self, n_vitals=7, n_demo=0, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_vitals, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.layer1 = nn.Sequential(ResBlock1D(64,  64,  dropout=dropout),
                                     ResBlock1D(64,  64,  dropout=dropout))
        self.layer2 = nn.Sequential(ResBlock1D(64,  128, dropout=dropout),
                                     ResBlock1D(128, 128, dropout=dropout))
        self.layer3 = nn.Sequential(ResBlock1D(128, 256, dropout=dropout),
                                     ResBlock1D(256, 256, dropout=dropout))
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.head   = MLPHead(256 + n_demo, hidden_dim=128, dropout=dropout)
        self.n_demo = n_demo

    def forward(self, x, demo=None):
        x    = self.stem(x)
        x    = self.layer1(x)
        x    = self.layer2(x)
        x    = self.layer3(x)
        feat = self.pool(x).squeeze(-1)       # (B, 256)
        if self.n_demo > 0 and demo is not None:
            feat = torch.cat([feat, demo], dim=1)
        return self.head(feat)


# ════════════════════════════════════════════════════════════════════════
# 5. Inception Classifier (1D)
# ════════════════════════════════════════════════════════════════════════

class InceptionBlock1D(nn.Module):
    """Multi-scale convolutions at kernel sizes 1, 3, 5, 7 + MaxPool branch."""
    def __init__(self, in_ch, out_ch_per_branch=32, dropout=0.2):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch_per_branch, 1, bias=False),
            nn.BatchNorm1d(out_ch_per_branch), nn.GELU())
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch_per_branch, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch_per_branch), nn.GELU())
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch_per_branch, 5, padding=2, bias=False),
            nn.BatchNorm1d(out_ch_per_branch), nn.GELU())
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch_per_branch, 7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch_per_branch), nn.GELU())
        self.branch_pool = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch_per_branch, 1, bias=False),
            nn.BatchNorm1d(out_ch_per_branch), nn.GELU())
        self.out_ch  = out_ch_per_branch * 5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([self.branch1(x), self.branch3(x), self.branch5(x),
                         self.branch7(x), self.branch_pool(x)], dim=1)
        return self.dropout(out)


class InceptionClassifier(nn.Module):
    def __init__(self, n_vitals=7, n_demo=0, dropout=0.3):
        super().__init__()
        self.block1 = InceptionBlock1D(n_vitals, out_ch_per_branch=32,
                                        dropout=dropout)
        self.block2 = InceptionBlock1D(160,      out_ch_per_branch=64,
                                        dropout=dropout)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        feat_dim    = 320
        self.head   = MLPHead(feat_dim + n_demo, hidden_dim=128, dropout=dropout)
        self.n_demo = n_demo

    def forward(self, x, demo=None):
        x    = self.block1(x)
        x    = self.block2(x)
        feat = self.pool(x).squeeze(-1)
        if self.n_demo > 0 and demo is not None:
            feat = torch.cat([feat, demo], dim=1)
        return self.head(feat)


# ════════════════════════════════════════════════════════════════════════
# 6. ROCKET Classifier
#    Random Convolutional Kernel Transform — Dempster et al. 2020
#    Uses random kernels to extract features, then a linear classifier
# ════════════════════════════════════════════════════════════════════════

class ROCKETClassifier(nn.Module):
    """
    ROCKET: Random Convolutional Kernel Transform.
    Generates n_kernels random 1D convolutions and extracts:
      - max value (PPV: proportion of positive values)
      - proportion of positive values
    These 2*n_kernels features are fed to a linear classifier.

    Unlike other models, ROCKET requires a two-step training process:
    Step 1: transform() — extract features (no gradients needed)
    Step 2: train linear head on features

    For end-to-end compatibility, the random kernels are fixed (no grad)
    and only the linear head is trained.
    """
    def __init__(self, n_vitals=7, n_demo=0, n_kernels=10000,
                 max_kernel_length=9, dropout=0.3):
        super().__init__()
        self.n_vitals      = n_vitals
        self.n_kernels     = n_kernels
        self.n_demo        = n_demo

        # Generate random kernels (fixed — not trained)
        torch.manual_seed(42)
        kernel_lengths = torch.randint(7, max_kernel_length + 1,
                                        (n_kernels,))
        # Store as list of fixed conv layers
        self.kernels = nn.ModuleList()
        for i in range(n_kernels):
            klen = int(kernel_lengths[i].item())
            if klen % 2 == 0:
                klen += 1   # ensure odd
            conv = nn.Conv1d(n_vitals, 1, klen,
                             padding=klen // 2, bias=True)
            # Random init with specific distribution from ROCKET paper
            with torch.no_grad():
                conv.weight.normal_(0, 1)
                conv.bias.uniform_(-1, 1)
            # Freeze kernel weights
            for p in conv.parameters():
                p.requires_grad = False
            self.kernels.append(conv)

        feat_dim  = n_kernels * 2 + n_demo   # max + PPV per kernel
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(self, x, demo=None):
        features = []
        for conv in self.kernels:
            out     = conv(x).squeeze(1)        # (B, T)
            max_val = out.max(dim=1).values     # (B,)
            ppv     = (out > 0).float().mean(dim=1)  # (B,)
            features.extend([max_val.unsqueeze(1), ppv.unsqueeze(1)])

        feat = torch.cat(features, dim=1)       # (B, n_kernels*2)
        if self.n_demo > 0 and demo is not None:
            feat = torch.cat([feat, demo], dim=1)
        return self.head(feat).squeeze(-1)      # (B,)


# ════════════════════════════════════════════════════════════════════════
# Registry
# ════════════════════════════════════════════════════════════════════════

CLF_REGISTRY = {
    "tcn":         TCNClassifier,
    "lstm":        LSTMClassifier,
    "transformer": TransformerClassifier,
    "resnet":      ResNetClassifier,
    "inception":   InceptionClassifier,
    "rocket":      ROCKETClassifier,
}


def build_classifier(name, n_vitals, n_demo=0, **kwargs):
    key = name.lower()
    if key not in CLF_REGISTRY:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Available: {list(CLF_REGISTRY.keys())}")
    return CLF_REGISTRY[key](n_vitals=n_vitals, n_demo=n_demo, **kwargs)


# ════════════════════════════════════════════════════════════════════════
# Quick sanity check
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, C, T   = 8, 7, 12   # batch, vitals, timesteps
    n_demo    = 4
    x         = torch.randn(B, C, T)
    demo      = torch.randn(B, n_demo)

    print(f"Input shape: {tuple(x.shape)}  Demo: {tuple(demo.shape)}\n")
    print(f"{'Model':<15} {'Output':>12}  {'Params':>10}")
    print("-" * 42)

    for name in CLF_REGISTRY:
        model  = build_classifier(name, n_vitals=C, n_demo=n_demo)
        out    = model(x, demo)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"{name:<15} {str(tuple(out.shape)):>12}  {nparams:>10,}")