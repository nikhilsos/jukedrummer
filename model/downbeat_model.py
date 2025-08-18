"""
File: beat_tracking_tcn/models/beat_net_offline.py
Description: Offline (non‑causal) variant of BeatNet.  
             The Temporal Convolutional Network is replaced with a non‑causal
             (bidirectional) dilated stack so every output frame has access to
             both past *and* future context.  Designed for full‑sequence /
             batch predictions where real‑time latency is not a concern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Tuple, Optional
from beat_tracking_tcn.models.tcn import NonCausalTemporalConvolutionalNetwork  
################################################################################
# Non‑causal TCN building blocks ################################################
################################################################################

class _NonCausalTemporalBlock(nn.Module):
    """A single non‑causal dilated residual block (no look‑ahead cropping)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        # Symmetric padding: pad equally left & right so length is preserved
        pad = ((kernel_size - 1) * dilation) // 2

        self.conv1 = weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        )
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.relu(self.conv1(x))
        y = self.dropout(y)
        y = self.relu(self.conv2(y))
        y = self.dropout(y)
        return self.relu(y + self.downsample(x))


class NonCausalTemporalConvNet(nn.Module):
    """Stack of non‑causal dilated residual blocks (TCN‑style)."""

    def __init__(
        self,
        in_channels: int,
        channels: Tuple[int, ...],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2 ** i  # Exponential growth, same as standard TCN
            in_ch = in_channels if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(
                _NonCausalTemporalBlock(
                    in_ch, out_ch, kernel_size, dilation, dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.network(x)

################################################################################
# BeatNet Offline ##############################################################
################################################################################

class BeatNetOffline(nn.Module):
    """
    Offline (non‑causal) BeatNet.  Identical front‑end CNN as the causal model
    but uses a NonCausalTemporalConvNet so predictions at time *t* incorporate
    future frames.  Suitable for batch/offline post‑processing pipelines.
    """

    def __init__(
        self,
        input: Tuple[int, int] = (3000, 81),
        output: int = 3000,
        channels: int = 16,
        tcn_kernel_size: int = 5,
        dropout: float = 0.1,
        downbeats: bool = False,
        num_rhythm_classes: int = 4,
        tcn_levels: int = 11,
    ):
        super().__init__()
        # ---------------- Front‑end CNN (unchanged) --------------------------
        self.conv1 = nn.Conv2d(1, channels, (3, 3), padding=(1, 0))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 3))

        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding=(1, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 3))

        self.conv3 = nn.Conv2d(channels, channels, (1, 8))
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)

        # ---------------- Non‑causal TCN ------------------------------------
        self.tcn = NonCausalTemporalConvolutionalNetwork(
            channels,
            [channels] * 11, # [channels, 2*channels, 4* channels, 8* channels] * 11
            tcn_kernel_size,
            dropout)


        # ---------------- Output heads --------------------------------------
        self.out_beat = nn.Conv1d(channels, 1 if not downbeats else 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.rhythm_pool = nn.AdaptiveAvgPool1d(1)
        self.rhythm_fc = nn.Linear(channels, num_rhythm_classes)

    # -------------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass.  Input x: (B, 1, T, F)."""
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        y = self.elu3(y)
        y = self.dropout3(y)

        # Reshape for 1D TCN: (B, C, T)
        y = y.view(y.size(0), y.size(1), y.size(2))
        y = self.tcn(y)
        embeddings = y  # (B, C, T)

        # Beat activations
        beat_out = self.sigmoid(self.out_beat(y))  # (B, 1/2, T)
        beat_out = beat_out.permute(0, 2, 1)  # (B, T, 1/2)

        # Rhythm classification
        rhythm_feat = self.rhythm_pool(y).squeeze(-1)
        rhythm_out = self.rhythm_fc(rhythm_feat)

        if return_embeddings:
            return beat_out, rhythm_out, embeddings
        return beat_out, rhythm_out

################################################################################
# Quick sanity check ###########################################################
################################################################################

if __name__ == "__main__":
    x = torch.randn(1, 1, 3000, 81)  # (batch, channel, time, freq)
    model = BeatNetOffline(num_rhythm_classes=4)
    beat, rhythm, emb = model(x, return_embeddings=True)
    print("Beat      :", beat.shape)
    print("Rhythm    :", rhythm.shape)
    print("Embeddings:", emb.shape)
