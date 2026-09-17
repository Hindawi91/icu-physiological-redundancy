"""
models.py

Collection of seq2seq deep learning architectures for vital-sign
reconstruction (predict one missing modality from the others).

All models share the same interface:
    Input:  (B, C_in, T)   batch, input channels (vitals), time
    Output: (B, C_out, T)  batch, output channels (usually 1), time
"""

import math
import torch
import torch.nn as nn


# ========================================================================
# 1. 1D-CNN U-Net
# ========================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet1D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 base_channels: int = 32, depth: int = 3, kernel_size: int = 5):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        out_ch = base_channels
        for _ in range(depth):
            self.encoders.append(ConvBlock(ch, out_ch, kernel_size))
            self.pools.append(nn.MaxPool1d(2))
            ch = out_ch
            out_ch *= 2
        self.bottleneck = ConvBlock(ch, out_ch, kernel_size)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = out_ch
        for _ in range(depth):
            dec_out = dec_in // 2
            self.upconvs.append(nn.ConvTranspose1d(dec_in, dec_out, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(dec_out * 2, dec_out, kernel_size))
            dec_in = dec_out
        self.head = nn.Conv1d(dec_in, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        orig_len = x.shape[-1]
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = nn.functional.pad(x, (0, diff))
                else:
                    x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = dec(x)
        out = self.head(x)
        if out.shape[-1] != orig_len:
            diff = orig_len - out.shape[-1]
            if diff > 0:
                out = nn.functional.pad(out, (0, diff))
            else:
                out = out[..., :orig_len]
        return out


# ========================================================================
# 2. LSTM/GRU Seq2Seq (non-autoregressive, full-sequence visible)
# ========================================================================

class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 hidden_size: int = 64, num_layers: int = 2,
                 bidirectional: bool = True, dropout: float = 0.1,
                 rnn_type: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 2, out_channels),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = self.head(out)
        return out.permute(0, 2, 1)


# ========================================================================
# 3. Bidirectional LSTM (simple, strong RNN baseline)
# ========================================================================

class BiLSTM(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.rnn(x)
        out = self.head(out)
        return out.permute(0, 2, 1)


# ========================================================================
# 4. Temporal Convolutional Network (dilated, non-causal)
# ========================================================================

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1     = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1       = nn.BatchNorm1d(out_ch)
        self.conv2     = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2       = nn.BatchNorm1d(out_ch)
        self.relu      = nn.ReLU(inplace=True)
        self.dropout   = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        if out.shape[-1] != residual.shape[-1]:
            min_len = min(out.shape[-1], residual.shape[-1])
            out, residual = out[..., :min_len], residual[..., :min_len]
        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 channels=(32, 32, 64, 64), kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        layers = []
        ch = in_channels
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TCNBlock(ch, out_ch, kernel_size, dilation, dropout))
            ch = out_ch
        self.network = nn.Sequential(*layers)
        self.head    = nn.Conv1d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.network(x)
        return self.head(out)


# ========================================================================
# 5. Transformer Encoder (full sequence visible, no causal mask)
# ========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerSeq2Seq(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 256, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.input_proj  = nn.Linear(in_channels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head    = nn.Linear(d_model, out_channels)

    def forward(self, x):
        x   = x.permute(0, 2, 1)
        x   = self.input_proj(x)
        x   = self.pos_encoding(x)
        x   = self.encoder(x)
        out = self.head(x)
        return out.permute(0, 2, 1)


# ========================================================================
# 6. Conv-LSTM hybrid (CNN feature extractor + BiLSTM)
# ========================================================================

class ConvLSTM(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1,
                 conv_channels: int = 32, kernel_size: int = 5,
                 hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels,   conv_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=pad),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(inplace=True),
        )
        self.rnn = nn.LSTM(
            input_size=conv_channels, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size * 2, out_channels)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.permute(0, 2, 1)
        out, _ = self.rnn(feat)
        out    = self.head(out)
        return out.permute(0, 2, 1)


# ========================================================================
# 7. TCN Classifier for sepsis prediction
#    Reuses TCNBlock backbone - demographics injected after last timestep
# ========================================================================

class TCNClassifier(nn.Module):
    """
    Sepsis classifier that reuses the TCN backbone from reconstruction.

    Architecture:
        Vitals (n_vitals × T)
            ↓
        TCN blocks (same structure as TCN reconstruction model)
            ↓
        Last timestep hidden state  (channels[-1],)
            ↓
        Concatenate demographics    (channels[-1] + n_demo,)
            ↓
        MLP (2 layers + dropout)
            ↓
        Logit → sigmoid → P(sepsis)

    Parameters
    ----------
    n_vitals    : number of dynamic vital sign input channels
    n_demo      : number of static demographic features (0 = no demographics)
    channels    : TCN channel widths per block (same default as TCN)
    kernel_size : TCN kernel size
    dropout     : dropout rate in TCN blocks and MLP
    mlp_hidden  : hidden size of the MLP classification head
    """
    def __init__(self, n_vitals: int = 7, n_demo: int = 4,
                 channels=(32, 32, 64, 64), kernel_size: int = 5,
                 dropout: float = 0.1, mlp_hidden: int = 64):
        super().__init__()

        # -- TCN backbone (identical structure to TCN reconstruction model) --
        layers = []
        ch = n_vitals
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(TCNBlock(ch, out_ch, kernel_size, dilation, dropout))
            ch = out_ch
        self.backbone    = nn.Sequential(*layers)
        self.tcn_out_ch  = ch   # = channels[-1]
        self.n_demo      = n_demo

        # -- MLP classification head --------------------------------------
        mlp_in = self.tcn_out_ch + n_demo
        self.classifier = nn.Sequential(
            nn.Linear(mlp_in,          mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden,      mlp_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )

    def forward(self, vitals, demo=None):
        """
        Parameters
        ----------
        vitals : (B, n_vitals, T)
        demo   : (B, n_demo)  - or None if no demographics

        Returns
        -------
        logits : (B,)  - pass through sigmoid for probabilities
        """
        feat = self.backbone(vitals)          # (B, channels[-1], T)
        last = feat[:, :, -1]                # (B, channels[-1])  ← last timestep

        if self.n_demo > 0 and demo is not None:
            last = torch.cat([last, demo], dim=1)   # (B, channels[-1] + n_demo)

        logits = self.classifier(last).squeeze(-1)  # (B,)
        return logits


# ========================================================================
# Registry - swap models by name from train.py
# ========================================================================

MODEL_REGISTRY = {
    "unet1d":       UNet1D,
    "lstm_seq2seq": LSTMSeq2Seq,
    "bilstm":       BiLSTM,
    "tcn":          TCN,
    "transformer":  TransformerSeq2Seq,
    "conv_lstm":    ConvLSTM,
}


def build_model(name: str, **kwargs):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key](**kwargs)


if __name__ == "__main__":
    dummy = torch.randn(8, 4, 24)
    for name, cls in MODEL_REGISTRY.items():
        model = cls()
        out   = model(dummy)
        print(f"{name:15s} output shape: {tuple(out.shape)}")

    # Test TCNClassifier
    print("\nTCNClassifier test:")
    clf    = TCNClassifier(n_vitals=7, n_demo=4)
    vitals = torch.randn(8, 7, 12)
    demo   = torch.randn(8, 4)
    logits = clf(vitals, demo)
    print(f"  logits shape: {tuple(logits.shape)}")   # (8,)