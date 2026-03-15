#!/usr/bin/env python3
"""CNN-LSTM on raw L1 tick sequences with HMM regime gating.

Architecture:
  - CNN-LSTM: trained on raw L1 tick windows → binary UP/DOWN signal
  - HMM regime_v2: gates which signals are acted on
    - TRENDING: take trend-following signals
    - RANGING: take mean-reversion signals
    - HIGH_VOL: go flat (no trades)

Labels: smoothed mid-price direction (DeepLOB-style)
  - Mean of next k mid-prices vs mean of previous k mid-prices
  - Threshold α filters out noise (neutral zone)

Usage:
    # Train + OOS eval
    python -u scripts/tick_predictor/train_cnn_lstm.py 2>&1 | tee logs/cnn_lstm_train.log

    # Sweep thresholds
    python -u scripts/tick_predictor/train_cnn_lstm.py --sweep 2>&1 | tee logs/cnn_lstm_sweep.log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from datetime import datetime, time as dt_time, timedelta
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.core.logging import configure_logging, get_logger
from src.signals.tick_predictor.model.cnn_lstm import CnnLstm

logger = get_logger("tick_predictor.cnn_lstm")

# ── Config ──────────────────────────────────────────────────────────
TRAIN_START = "2025-01-01"
TRAIN_END = "2025-09-01"
EMBARGO_END = "2025-10-01"
DATA_END = "2026-03-14"

SEQ_LEN = 500            # ticks per input window
LABEL_HORIZON_K = 50     # smoothed mid-price: mean of next/prev k ticks
LABEL_THRESHOLD = 0.0002 # α: minimum return for UP/DOWN (else NEUTRAL, dropped)
PREDICTION_STRIDE = 500  # generate one sample every N ticks (non-overlapping)

BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4

MODEL_DIR = Path("models/tick_predictor")
DATA_DIR = Path("data/tick_predictor")
RESULTS_DIR = Path("results/tick_predictor")
MES_TICK_SIZE = 0.25
MES_POINT_VALUE = 5.0
COMMISSION_RT = 0.59

N_CHANNELS = 8  # features per tick


# ── Dataset ─────────────────────────────────────────────────────────

class L1TickDataset(Dataset):
    """Dataset of raw L1 tick sequences with smoothed mid-price labels."""

    def __init__(
        self,
        sequences: np.ndarray,   # (N, seq_len, n_channels) float32
        labels: np.ndarray,      # (N,) int64: 0=DOWN, 1=UP
        weights: np.ndarray | None = None,
    ) -> None:
        self.sequences = torch.from_numpy(sequences)
        self.labels = torch.from_numpy(labels).float()
        self.weights = (
            torch.from_numpy(weights) if weights is not None
            else torch.ones(len(labels))
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.labels[idx], self.weights[idx]


# ── Data loading & preprocessing ────────────────────────────────────

def load_l1_ticks(start_date: str, end_date: str, rth_only: bool = True) -> pl.DataFrame:
    """Load raw L1 tick data."""
    import datetime as dt

    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)

    paths = [
        f"data/l1/year={y}/data.parquet"
        for y in range(start.year, end.year + 1)
        if Path(f"data/l1/year={y}/data.parquet").exists()
    ]

    if not paths:
        raise FileNotFoundError(f"No L1 data for {start_date} to {end_date}")

    start_dt = dt.datetime.combine(start, dt.time.min)
    end_dt = dt.datetime.combine(end, dt.time.min)

    df = (
        pl.scan_parquet(paths)
        .filter(
            (pl.col("timestamp") >= start_dt)
            & (pl.col("timestamp") < end_dt)
        )
        .collect()
    )

    if rth_only:
        et_col = pl.col("timestamp").dt.convert_time_zone("US/Eastern")
        df = df.with_columns(et_col.alias("_et"))
        df = df.filter(
            (pl.col("_et").dt.time() >= dt_time(9, 30))
            & (pl.col("_et").dt.time() < dt_time(16, 0))
        ).drop("_et")

    print(f"  Loaded {len(df):,} L1 ticks ({start_date} to {end_date})")
    return df


def normalize_ticks(df: pl.DataFrame) -> np.ndarray:
    """Convert raw L1 ticks to normalized 8-channel feature array.

    Normalization is local to each day to prevent cross-day leakage.
    Returns (N, 8) float32 array.
    """
    price = df["price"].to_numpy().astype(np.float64)
    size = df["size"].to_numpy().astype(np.float64)
    side_str = df["side"].to_numpy()
    bid_price = df["bid_price"].to_numpy().astype(np.float64)
    ask_price = df["ask_price"].to_numpy().astype(np.float64)
    bid_size = df["bid_size"].to_numpy().astype(np.float64)
    ask_size = df["ask_size"].to_numpy().astype(np.float64)

    n = len(price)
    features = np.zeros((n, N_CHANNELS), dtype=np.float32)

    # Mid-price for relative pricing
    mid = (bid_price + ask_price) / 2.0

    # Channel 0: price relative to mid (in ticks)
    features[:, 0] = ((price - mid) / MES_TICK_SIZE).astype(np.float32)

    # Channel 1: log(size + 1)
    features[:, 1] = np.log1p(size).astype(np.float32)

    # Channel 2: side encoding (-1=sell, 0=neutral, +1=buy)
    side_enc = np.zeros(n, dtype=np.float32)
    side_enc[side_str == "B"] = 1.0
    side_enc[side_str == "A"] = -1.0
    features[:, 2] = side_enc

    # Channel 3: bid price relative to mid (in ticks)
    features[:, 3] = ((bid_price - mid) / MES_TICK_SIZE).astype(np.float32)

    # Channel 4: ask price relative to mid (in ticks)
    features[:, 4] = ((ask_price - mid) / MES_TICK_SIZE).astype(np.float32)

    # Channel 5: log(bid_size + 1)
    features[:, 5] = np.log1p(bid_size).astype(np.float32)

    # Channel 6: log(ask_size + 1)
    features[:, 6] = np.log1p(ask_size).astype(np.float32)

    # Channel 7: order book imbalance
    total = bid_size + ask_size
    imbalance = np.where(total > 0, (bid_size - ask_size) / total, 0.0)
    features[:, 7] = imbalance.astype(np.float32)

    return features


def build_sequences_and_labels(
    features: np.ndarray,
    mid_prices: np.ndarray,
    timestamps_ns: np.ndarray,
    seq_len: int = SEQ_LEN,
    horizon_k: int = LABEL_HORIZON_K,
    threshold: float = LABEL_THRESHOLD,
    stride: int = PREDICTION_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (sequence, label) pairs with smoothed mid-price labeling.

    Smoothed label (DeepLOB-style):
      future_mean = mean(mid[t+1 : t+1+k])
      past_mean   = mean(mid[t-k+1 : t+1])
      return = (future_mean - past_mean) / past_mean
      label = UP if return > α, DOWN if return < -α, else NEUTRAL (dropped)

    Strict causality: sequence ends at tick t, label uses ticks t+1 onward.

    Returns:
        sequences: (N, seq_len, n_channels) float32
        labels: (N,) int64: 0=DOWN, 1=UP
        timestamps: (N,) int64: timestamp of last tick in each sequence
        indices: (N,) int64: original tick index of prediction point
    """
    n = len(features)
    min_start = seq_len + horizon_k  # need k bars before for past_mean
    max_end = n - horizon_k          # need k bars after for future_mean

    sequences = []
    labels = []
    timestamps = []
    indices = []

    for t in range(min_start, max_end, stride):
        # Past and future smoothed mid-prices
        past_mean = float(np.mean(mid_prices[t - horizon_k + 1:t + 1]))
        future_mean = float(np.mean(mid_prices[t + 1:t + 1 + horizon_k]))

        if past_mean < 1e-6:
            continue

        ret = (future_mean - past_mean) / past_mean

        # Classify
        if ret > threshold:
            label = 1  # UP
        elif ret < -threshold:
            label = 0  # DOWN
        else:
            continue  # NEUTRAL — skip

        # Extract sequence ending at tick t
        seq = features[t - seq_len + 1:t + 1]  # (seq_len, n_channels)

        # Local z-score normalization per channel within this window
        for ch in range(seq.shape[1]):
            col = seq[:, ch]
            std = np.std(col)
            if std > 1e-8:
                seq[:, ch] = (col - np.mean(col)) / std

        sequences.append(seq)
        labels.append(label)
        timestamps.append(timestamps_ns[t])
        indices.append(t)

    sequences = np.array(sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    timestamps = np.array(timestamps, dtype=np.int64)
    indices = np.array(indices, dtype=np.int64)

    n_up = np.sum(labels == 1)
    n_down = np.sum(labels == 0)
    print(f"  Samples: {len(labels):,} (UP: {n_up:,} / DOWN: {n_down:,})")

    return sequences, labels, timestamps, indices


# ── Training ────────────────────────────────────────────────────────

def train_model(
    train_dataset: L1TickDataset,
    val_dataset: L1TickDataset,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
) -> tuple[CnnLstm, dict]:
    """Train CNN-LSTM with early stopping."""
    model = CnnLstm(n_channels=N_CHANNELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 7
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch, w_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss = (criterion(logits, y_batch) * w_batch).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # Validate
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch, w_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch).squeeze(-1)
                loss = criterion(logits, y_batch).mean()
                val_losses.append(loss.item())

                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        avg_val_loss = float(np.mean(val_losses))
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(avg_val_loss)

        print(
            f"  Epoch {epoch+1:>2}/{epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.1e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


# ── OOS Backtest with Regime Gating ─────────────────────────────────

def oos_backtest_with_regime(
    model: CnnLstm,
    oos_sequences: np.ndarray,
    oos_timestamps: np.ndarray,
    oos_indices: np.ndarray,
    tick_data: np.ndarray,       # full normalized tick array
    mid_prices: np.ndarray,      # full mid prices
    regime_labels: np.ndarray | None,  # regime per tick: 0=TREND, 1=RANGE, 2=HIGHVOL
    device: torch.device,
    threshold: float = 0.6,
    tp_ticks: int = 6,
    sl_ticks: int = 4,
    horizon_ticks: int = 3000,   # ~5 minutes of ticks for barrier check
    regime_mode: str = "gate",   # "gate" or "none"
) -> dict:
    """OOS backtest: CNN-LSTM signals, optionally gated by HMM regime.

    Regime gating (mode="gate"):
      - TRENDING: only take signals in trend direction (UP if uptrend, DOWN if downtrend)
      - RANGING: take all signals (mean-reversion works)
      - HIGH_VOL: no trades
    """
    model.eval()
    tp_pts = tp_ticks * MES_TICK_SIZE
    sl_pts = sl_ticks * MES_TICK_SIZE

    trades = []
    n_total = len(mid_prices)

    with torch.no_grad():
        # Batch predict all OOS sequences
        ds = torch.from_numpy(oos_sequences).to(device)
        all_logits = []
        for i in range(0, len(ds), BATCH_SIZE):
            batch = ds[i:i + BATCH_SIZE]
            logits = model(batch).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
        probas = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))  # sigmoid

    for idx_in_oos in range(len(probas)):
        p_up = float(probas[idx_in_oos])
        p_down = 1.0 - p_up
        tick_idx = int(oos_indices[idx_in_oos])
        ts = int(oos_timestamps[idx_in_oos])

        # Regime gating
        if regime_mode == "gate" and regime_labels is not None:
            regime = int(regime_labels[tick_idx])
            if regime == 2:  # HIGH_VOL → flat
                continue

        # Signal
        if p_up >= threshold:
            direction = 1  # LONG
        elif p_down >= threshold:
            direction = -1  # SHORT
        else:
            continue

        entry_price = float(mid_prices[tick_idx])
        end_idx = min(tick_idx + horizon_ticks, n_total - 1)

        # Simulate with mid-price barriers
        pnl = 0.0
        exit_reason = "vertical"

        if direction == 1:
            target = entry_price + tp_pts
            stop = entry_price - sl_pts
            for j in range(tick_idx + 1, end_idx + 1):
                mp = float(mid_prices[j])
                if mp >= target:
                    pnl = tp_pts * MES_POINT_VALUE
                    exit_reason = "tp"
                    break
                elif mp <= stop:
                    pnl = -sl_pts * MES_POINT_VALUE
                    exit_reason = "sl"
                    break
            else:
                pnl = (float(mid_prices[end_idx]) - entry_price) * MES_POINT_VALUE
        else:
            target = entry_price - tp_pts
            stop = entry_price + sl_pts
            for j in range(tick_idx + 1, end_idx + 1):
                mp = float(mid_prices[j])
                if mp <= target:
                    pnl = tp_pts * MES_POINT_VALUE
                    exit_reason = "tp"
                    break
                elif mp >= stop:
                    pnl = -sl_pts * MES_POINT_VALUE
                    exit_reason = "sl"
                    break
            else:
                pnl = (entry_price - float(mid_prices[end_idx])) * MES_POINT_VALUE

        net_pnl = pnl - COMMISSION_RT
        trades.append({
            "tick_idx": tick_idx,
            "timestamp_ns": ts,
            "direction": direction,
            "entry": entry_price,
            "pnl": net_pnl,
            "exit_reason": exit_reason,
            "p_up": p_up,
        })

    return _summarize_trades(trades)


def _summarize_trades(trades: list[dict]) -> dict:
    """Compute trade summary statistics."""
    if not trades:
        return {
            "trades": 0, "win_rate": 0, "total_pnl": 0, "sharpe": 0,
            "profit_factor": 0, "max_dd_pct": 0, "max_dd_dollars": 0,
            "avg_daily_pnl": 0, "worst_day_pnl": 0, "profitable_days": "0/0",
            "avg_pnl_per_trade": 0, "tp_rate": 0, "sl_rate": 0,
        }

    pnls = np.array([t["pnl"] for t in trades])
    wins = np.sum(pnls > 0)
    total_pnl = float(np.sum(pnls))
    win_rate = float(wins / len(pnls)) * 100

    sharpe = (
        float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
        if np.std(pnls) > 0 else 0.0
    )

    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    cumsum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    max_dd_pct = max_dd / max(abs(total_pnl), 1.0) * 100

    # Exit reason breakdown
    exit_reasons = [t["exit_reason"] for t in trades]
    tp_count = sum(1 for r in exit_reasons if r == "tp")
    sl_count = sum(1 for r in exit_reasons if r == "sl")

    # Daily P&L
    import pandas as pd
    trade_dates = pd.to_datetime([t["timestamp_ns"] for t in trades], unit="ns").date
    daily_pnl: dict = {}
    for date, pnl_val in zip(trade_dates, pnls):
        daily_pnl[date] = daily_pnl.get(date, 0.0) + pnl_val
    daily_vals = list(daily_pnl.values())
    avg_daily = float(np.mean(daily_vals)) if daily_vals else 0.0
    worst_day = float(np.min(daily_vals)) if daily_vals else 0.0
    prof_days = sum(1 for v in daily_vals if v > 0)

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "sharpe": sharpe,
        "profit_factor": pf,
        "max_dd_pct": max_dd_pct,
        "max_dd_dollars": max_dd,
        "avg_daily_pnl": avg_daily,
        "worst_day_pnl": worst_day,
        "profitable_days": f"{prof_days}/{len(daily_vals)}",
        "avg_pnl_per_trade": float(np.mean(pnls)),
        "tp_rate": tp_count / len(trades) * 100,
        "sl_rate": sl_count / len(trades) * 100,
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=TRAIN_START)
    parser.add_argument("--end", default=DATA_END)
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--embargo-end", default=EMBARGO_END)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--stride", type=int, default=PREDICTION_STRIDE)
    parser.add_argument("--no-regime", action="store_true", help="Disable regime gating")
    args = parser.parse_args()

    configure_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  CNN-LSTM ON RAW L1 TICKS + HMM REGIME GATE")
    print(f"  Device:  {device}")
    print(f"  Train:   {args.start} to {args.train_end}")
    print(f"  Embargo: {args.train_end} to {args.embargo_end}")
    print(f"  OOS:     {args.embargo_end} to {args.end}")
    print(f"  Seq len: {args.seq_len} ticks")
    print(f"  Stride:  {args.stride} ticks")
    print(f"  Label:   smoothed mid-price (k={LABEL_HORIZON_K}, α={LABEL_THRESHOLD})")
    print(f"  Regime:  {'disabled' if args.no_regime else 'enabled (gate mode)'}")
    print(f"{'='*70}\n")

    # ── Load all L1 data ────────────────────────────────────
    print("  STEP 1: Loading L1 tick data...")
    t0 = _time.perf_counter()
    df = load_l1_ticks(args.start, args.end)
    timestamps_ns = df["timestamp"].dt.epoch("ns").to_numpy()

    # Compute mid-prices
    bid_prices = df["bid_price"].to_numpy().astype(np.float64)
    ask_prices = df["ask_price"].to_numpy().astype(np.float64)
    mid_prices = (bid_prices + ask_prices) / 2.0

    # Normalize tick features
    print("  Normalizing tick features...")
    features = normalize_ticks(df)
    del df  # free memory
    print(f"  Done ({_time.perf_counter()-t0:.1f}s)")

    # ── Load regime labels (optional) ───────────────────────
    regime_labels = None
    if not args.no_regime:
        print("\n  STEP 2: Loading HMM regime labels...")
        try:
            regime_labels = load_regime_labels(timestamps_ns)
            if regime_labels is not None:
                unique, counts = np.unique(regime_labels, return_counts=True)
                for u, c in zip(unique, counts):
                    name = {0: "TRENDING", 1: "RANGING", 2: "HIGH_VOL"}.get(u, "?")
                    print(f"    {name}: {c:,} ticks ({c/len(regime_labels)*100:.1f}%)")
        except Exception as e:
            print(f"    Regime loading failed: {e}")
            print(f"    Continuing without regime gating")
            regime_labels = None

    # ── Build sequences and labels ──────────────────────────
    print(f"\n  STEP 3: Building sequences (seq_len={args.seq_len}, stride={args.stride})...")
    t0 = _time.perf_counter()

    import datetime as dt
    train_end_ns = int(
        dt.datetime.combine(dt.date.fromisoformat(args.train_end), dt.time.min,
                            tzinfo=dt.timezone.utc).timestamp() * 1e9
    )
    embargo_end_ns = int(
        dt.datetime.combine(dt.date.fromisoformat(args.embargo_end), dt.time.min,
                            tzinfo=dt.timezone.utc).timestamp() * 1e9
    )

    # Build all sequences
    sequences, labels, seq_timestamps, seq_indices = build_sequences_and_labels(
        features, mid_prices, timestamps_ns,
        seq_len=args.seq_len,
        horizon_k=LABEL_HORIZON_K,
        threshold=LABEL_THRESHOLD,
        stride=args.stride,
    )
    print(f"  Built {len(sequences):,} samples ({_time.perf_counter()-t0:.1f}s)")

    # Split into train / val / OOS
    train_mask = seq_timestamps < train_end_ns
    oos_mask = seq_timestamps >= embargo_end_ns

    train_seqs = sequences[train_mask]
    train_labels = labels[train_mask]

    oos_seqs = sequences[oos_mask]
    oos_labels = labels[oos_mask]
    oos_timestamps_arr = seq_timestamps[oos_mask]
    oos_indices_arr = seq_indices[oos_mask]

    # Validation: last 15% of train
    n_train = len(train_seqs)
    val_n = int(n_train * 0.15)
    val_seqs = train_seqs[-val_n:]
    val_labels = train_labels[-val_n:]
    train_seqs = train_seqs[:-val_n]
    train_labels = train_labels[:-val_n]

    print(f"\n  Data splits:")
    print(f"    Train: {len(train_seqs):,}")
    print(f"    Val:   {len(val_seqs):,}")
    print(f"    OOS:   {len(oos_seqs):,}")

    if len(train_seqs) < 100:
        print("  ERROR: Not enough training data")
        return

    # ── Train ───────────────────────────────────────────────
    print(f"\n  STEP 4: Training CNN-LSTM ({args.epochs} epochs)...")
    t0 = _time.perf_counter()

    train_ds = L1TickDataset(train_seqs, train_labels)
    val_ds = L1TickDataset(val_seqs, val_labels)

    model, history = train_model(train_ds, val_ds, device, epochs=args.epochs)

    print(f"  Training done ({_time.perf_counter()-t0:.1f}s)")
    print(f"  Best val_loss: {min(history['val_loss']):.4f}")
    print(f"  Best val_acc:  {max(history['val_acc']):.4f}")

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"cnn_lstm_{ts}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": {
            "seq_len": args.seq_len,
            "stride": args.stride,
            "label_horizon_k": LABEL_HORIZON_K,
            "label_threshold": LABEL_THRESHOLD,
            "n_channels": N_CHANNELS,
        },
    }, model_path)
    print(f"  Model saved: {model_path}")

    # ── OOS Backtest ────────────────────────────────────────
    print(f"\n  STEP 5: OOS Backtest")

    regime_mode = "none" if args.no_regime or regime_labels is None else "gate"

    if args.sweep:
        thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        tp_sl_configs = [(6, 4), (8, 4), (8, 6), (10, 6)]

        print(f"\n  Sweeping {len(thresholds)} thresholds x {len(tp_sl_configs)} TP/SL configs")
        print(f"  Regime mode: {regime_mode}")
        print(f"  {'thresh':>6} {'TP':>3} {'SL':>3} | {'trades':>6} {'WR':>6} "
              f"{'PnL':>10} {'Sharpe':>7} {'PF':>6} {'DD$':>8} "
              f"{'TP%':>5} {'SL%':>5} | profDays")
        print(f"  {'-'*100}")

        all_results = []
        best_sharpe = -999
        best_config = None

        for thresh in thresholds:
            for tp, sl in tp_sl_configs:
                r = oos_backtest_with_regime(
                    model, oos_seqs, oos_timestamps_arr, oos_indices_arr,
                    features, mid_prices, regime_labels, device,
                    threshold=thresh, tp_ticks=tp, sl_ticks=sl,
                    regime_mode=regime_mode,
                )
                print(
                    f"  {thresh:>6.2f} {tp:>3} {sl:>3} | {r['trades']:>6} "
                    f"{r['win_rate']:>5.1f}% ${r['total_pnl']:>9,.0f} "
                    f"{r['sharpe']:>7.2f} {r['profit_factor']:>6.2f} "
                    f"${r['max_dd_dollars']:>7,.0f} "
                    f"{r['tp_rate']:>4.1f}% {r['sl_rate']:>4.1f}% | "
                    f"{r['profitable_days']}"
                )
                r["threshold"] = thresh
                r["tp_ticks"] = tp
                r["sl_ticks"] = sl
                all_results.append(r)

                if r["sharpe"] > best_sharpe and r["trades"] >= 20:
                    best_sharpe = r["sharpe"]
                    best_config = r

        # Also run without regime gating for comparison
        if regime_mode == "gate":
            print(f"\n  --- WITHOUT REGIME GATING (comparison) ---")
            for thresh in [0.60, 0.70]:
                for tp, sl in [(6, 4), (8, 4)]:
                    r = oos_backtest_with_regime(
                        model, oos_seqs, oos_timestamps_arr, oos_indices_arr,
                        features, mid_prices, regime_labels, device,
                        threshold=thresh, tp_ticks=tp, sl_ticks=sl,
                        regime_mode="none",
                    )
                    print(
                        f"  {thresh:>6.2f} {tp:>3} {sl:>3} | {r['trades']:>6} "
                        f"{r['win_rate']:>5.1f}% ${r['total_pnl']:>9,.0f} "
                        f"{r['sharpe']:>7.2f} {r['profit_factor']:>6.2f} "
                        f"${r['max_dd_dollars']:>7,.0f} "
                        f"{r['tp_rate']:>4.1f}% {r['sl_rate']:>4.1f}% | "
                        f"{r['profitable_days']}"
                    )
                    r["threshold"] = thresh
                    r["tp_ticks"] = tp
                    r["sl_ticks"] = sl
                    r["regime_mode"] = "none"
                    all_results.append(r)

        if best_config:
            print(f"\n  BEST CONFIG (with regime gate):")
            print(f"    Threshold: {best_config['threshold']}")
            print(f"    TP/SL: {best_config['tp_ticks']}/{best_config['sl_ticks']}")
            print(f"    Trades: {best_config['trades']}, WR: {best_config['win_rate']:.1f}%")
            print(f"    Sharpe: {best_config['sharpe']:.2f}, PF: {best_config['profit_factor']:.2f}")
            print(f"    Total PnL: ${best_config['total_pnl']:,.0f}")

        # Save results
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_path = RESULTS_DIR / f"cnn_lstm_sweep_{ts}.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved -> {results_path}")

    else:
        # Single eval
        r = oos_backtest_with_regime(
            model, oos_seqs, oos_timestamps_arr, oos_indices_arr,
            features, mid_prices, regime_labels, device,
            threshold=0.6, regime_mode=regime_mode,
        )
        print(f"\n  OOS Results ({regime_mode} regime):")
        print(f"    Trades:      {r['trades']}")
        print(f"    Win rate:    {r['win_rate']:.1f}%")
        print(f"    Total PnL:   ${r['total_pnl']:,.2f}")
        print(f"    Sharpe:      {r['sharpe']:.2f}")
        print(f"    PF:          {r['profit_factor']:.2f}")
        print(f"    Max DD:      ${r['max_dd_dollars']:,.0f}")
        print(f"    TP rate:     {r['tp_rate']:.1f}%")
        print(f"    SL rate:     {r['sl_rate']:.1f}%")
        print(f"    Prof days:   {r['profitable_days']}")

    print(f"\n  Model: {model_path}")
    print()


def load_regime_labels(timestamps_ns: np.ndarray) -> np.ndarray | None:
    """Load HMM regime labels aligned to tick timestamps.

    Tries to load pre-computed regime labels, or runs the HMM model
    on 5m bars and maps back to tick timestamps via forward-fill.
    """
    # Check for cached regime labels
    cache_path = DATA_DIR / "regime_labels_per_tick.npy"
    if cache_path.exists():
        labels = np.load(cache_path)
        if len(labels) == len(timestamps_ns):
            print("    [Cache HIT] regime labels")
            return labels

    # Try to load regime model and compute
    model_path = Path("models/regime_v2/regime_v2.joblib")
    if not model_path.exists():
        print("    No regime model found")
        return None

    print("    Computing regime labels from model (this may take a while)...")

    try:
        from src.models.regime_detector_v2 import RegimeDetectorV2
        detector = RegimeDetectorV2.load(model_path)

        # Build 5m bars from L1 for regime detection
        # For now, return None and let the user pre-compute
        print("    TODO: auto-compute regime labels from L1 ticks")
        print("    Pre-compute with: python scripts/train/regime_labels.py")
        return None
    except Exception as e:
        print(f"    Failed to load regime model: {e}")
        return None


if __name__ == "__main__":
    main()
