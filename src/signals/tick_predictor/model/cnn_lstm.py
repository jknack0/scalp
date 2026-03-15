"""CNN-LSTM model for raw L1 tick sequence direction prediction.

Architecture:
  Input: (batch, seq_len, 8) — raw L1 tick features (normalized)
  → Conv1D blocks: extract local microstructure patterns
    (sweeps, imbalance shifts, aggression bursts)
  → LSTM: capture temporal dependencies across the sequence
  → FC head: binary classification (DOWN vs UP)

Input channels per tick (8):
  0: price_rel      — price relative to initial mid-price (in ticks)
  1: size_log       — log(trade size)
  2: side           — trade side: -1 (sell), 0 (neutral), +1 (buy)
  3: bid_price_rel  — bid price relative to initial mid-price (in ticks)
  4: ask_price_rel  — ask price relative to initial mid-price (in ticks)
  5: bid_size_log   — log(bid size at best)
  6: ask_size_log   — log(ask size at best)
  7: imbalance      — (bid_size - ask_size) / (bid_size + ask_size)

Designed for ~500-tick windows of MES L1 data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CnnLstm(nn.Module):
    """CNN-LSTM for raw L1 tick sequence classification.

    Args:
        n_channels: number of input features per tick (default 8)
        cnn_filters: list of filter counts for Conv1D layers
        kernel_sizes: list of kernel sizes (one per CNN layer)
        lstm_hidden: LSTM hidden dimension
        lstm_layers: number of LSTM layers
        dropout: dropout rate for regularization
        pool_size: max-pool size between CNN layers (reduces seq length)
    """

    def __init__(
        self,
        n_channels: int = 8,
        cnn_filters: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        pool_size: int = 2,
    ) -> None:
        super().__init__()

        if cnn_filters is None:
            cnn_filters = [32, 64, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3]

        assert len(cnn_filters) == len(kernel_sizes)

        # CNN: 1D convolutions over the tick sequence
        # Input: (batch, n_channels, seq_len)
        cnn_blocks: list[nn.Module] = []
        in_ch = n_channels
        for i, (filters, ks) in enumerate(zip(cnn_filters, kernel_sizes)):
            cnn_blocks.extend([
                nn.Conv1d(in_ch, filters, ks, padding=ks // 2),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout),
            ])
            in_ch = filters

        self.cnn = nn.Sequential(*cnn_blocks)

        # LSTM: temporal patterns across compressed sequence
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,  # causal: only look backward
        )

        # Attention pooling over LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden, 1),
        )

        # FC head
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, n_channels) — raw L1 tick features

        Returns:
            logits: (batch, 1)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # Back to (batch, seq_len_compressed, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_compressed, hidden)

        # Attention pooling
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden)

        return self.head(context)
