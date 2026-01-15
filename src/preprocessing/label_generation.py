# src/preprocessing/label_generation.py
"""
label_generation.py (version "prod")
------------------------------------
Crée les labels BUY / SELL / HOLD à partir des features (ou du clean),
puis construit le dataset ML final:

- Input : data/features/<symbol>_<interval>_features.parquet
- Output: data/processed/<symbol>_<interval>_ml.parquet
  (ex: data/processed/btcusdt_1h_ml.parquet)

Paramètres:
- horizon_hours (par défaut 24) : on regarde le futur à T + H
- buy_threshold : ex 0.005 (= +0.5%)
- sell_threshold: ex 0.005 (= -0.5% en magnitude)
- fee_rate : ex 0.001 (= 0.1%)
- fee_mode : one_way | round_trip
    - one_way    : coût 1 fois (entrée OU sortie)
    - round_trip : coût 2 fois (entrée + sortie) => plus conservateur

⚠️ IMPORTANT:
- Le dataset final contient label + (optionnel) target_future_return pour debug.
- En entraînement, tu dois EXCLURE toutes colonnes target_*.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.config import (
    ensure_directories,
    FEATURES_DIR,
    PROCESSED_DIR,
    SYMBOLS as CONFIG_SYMBOLS,
    DEFAULT_KLINE_INTERVAL,
)

# -------------------------
# Logging
# -------------------------
ensure_directories()
logger = logging.getLogger("label_generation_prod")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)


def compute_labels(
    close: pd.Series,
    horizon: int,
    buy_threshold: float,
    sell_threshold: float,
    fee_rate: float,
    fee_mode: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Retourne:
    - future_return: (close[t+h]/close[t] - 1)
    - net_profit_long: future_return - fees
    - net_profit_short: (-future_return) - fees  (profit si le prix baisse)
    """
    future_close = close.shift(-horizon)
    future_return = (future_close / close) - 1.0

    if fee_mode == "round_trip":
        fee_cost = 2.0 * fee_rate
    elif fee_mode == "one_way":
        fee_cost = 1.0 * fee_rate
    else:
        raise ValueError("fee_mode must be one_way or round_trip")

    net_profit_long = future_return - fee_cost
    net_profit_short = (-future_return) - fee_cost
    return future_return, net_profit_long, net_profit_short


def decide_label(net_long: float, net_short: float, buy_threshold: float, sell_threshold: float) -> str:
    """
    Règle:
    - BUY  si net_long  >= buy_threshold
    - SELL si net_short >= sell_threshold
    - sinon HOLD

    Si BUY et SELL sont vrais (rare), on prend celui avec la meilleure marge.
    """
    is_buy = net_long >= buy_threshold
    is_sell = net_short >= sell_threshold

    if is_buy and is_sell:
        return "BUY" if net_long >= net_short else "SELL"
    if is_buy:
        return "BUY"
    if is_sell:
        return "SELL"
    return "HOLD"


def process_symbol(
    symbol: str,
    interval: str,
    horizon_hours: int,
    buy_threshold: float,
    sell_threshold: float,
    fee_rate: float,
    fee_mode: str,
    dropna_features: bool = True,
) -> Path:
    symbol_slug = symbol.lower()
    interval_slug = interval.lower()

    in_path = FEATURES_DIR / f"{symbol_slug}_{interval_slug}_features.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    logger.info(
        f"➡️  Label gen: {symbol} {interval} | horizon={horizon_hours}h "
        f"buy={buy_threshold:.4f} sell={sell_threshold:.4f} fee={fee_rate:.4f} mode={fee_mode}"
    )

    df = pd.read_parquet(in_path)
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "close" not in df.columns:
        raise ValueError(f"{in_path.name}: missing 'close' column")

    # Compute target-related columns
    future_return, net_long, net_short = compute_labels(
        close=df["close"],
        horizon=horizon_hours,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        fee_rate=fee_rate,
        fee_mode=fee_mode,
    )

    df["target_future_return"] = future_return
    df["target_net_long"] = net_long
    df["target_net_short"] = net_short

    # Label string
    df["label"] = [
        decide_label(l, s, buy_threshold, sell_threshold)
        for l, s in zip(df["target_net_long"].to_numpy(), df["target_net_short"].to_numpy())
    ]

    # Label encoding (utile pour ML)
    mapping = {"SELL": -1, "HOLD": 0, "BUY": 1}
    df["label_int"] = df["label"].map(mapping).astype("int8")

    # On supprime les dernières lignes où le futur n'existe pas (NaN)
    df = df.dropna(subset=["target_future_return"]).reset_index(drop=True)

    # Optionnel: retirer toutes les lignes avec NaN dans les features (rolling windows)
    if dropna_features:
        # On évite de dropper les colonnes target_* et label qui sont déjà OK.
        # On drop si NaN dans une colonne feature quelconque (hors quelques colonnes de base).
        exclude = {"timestamp", "open_time", "symbol", "kline_interval", "label", "label_int",
                   "target_future_return", "target_net_long", "target_net_short"}
        feature_cols = [c for c in df.columns if c not in exclude]
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

    out_path = PROCESSED_DIR / f"{symbol_slug}_{interval_slug}_ml.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"✅ Saved ML dataset: {out_path} | rows={len(df)} cols={len(df.columns)}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate BUY/SELL/HOLD labels + build ML parquet.")
    p.add_argument("--symbols", type=str, default=",".join(CONFIG_SYMBOLS), help="BTCUSDT,ETHUSDT,...")
    p.add_argument("--interval", type=str, default=DEFAULT_KLINE_INTERVAL, help="ex: 1h")

    p.add_argument("--horizon-hours", type=int, default=24, help="Label horizon in hours (default: 24)")
    p.add_argument("--buy-threshold", type=float, default=0.005, help="Net threshold for BUY (default: 0.005 = 0.5%)")
    p.add_argument("--sell-threshold", type=float, default=0.005, help="Net threshold for SELL (default: 0.005 = 0.5%)")
    p.add_argument("--fee-rate", type=float, default=0.001, help="Fee rate (default: 0.001 = 0.1%)")
    p.add_argument("--fee-mode", type=str, default="round_trip", choices=["one_way", "round_trip"])
    p.add_argument("--keep-nans", action="store_true", help="Do not drop NaNs in feature cols (not recommended)")

    return p.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    interval = args.interval.strip().lower()

    dropna_features = not args.keep_nans

    logger.info(f"=== label_generation PROD start === symbols={symbols} interval={interval}")
    for sym in symbols:
        try:
            process_symbol(
                symbol=sym,
                interval=interval,
                horizon_hours=int(args.horizon_hours),
                buy_threshold=float(args.buy_threshold),
                sell_threshold=float(args.sell_threshold),
                fee_rate=float(args.fee_rate),
                fee_mode=str(args.fee_mode),
                dropna_features=dropna_features,
            )
        except Exception as e:
            logger.exception(f"❌ Failed for {sym} {interval}: {e}")

    logger.info("=== label_generation PROD done ===")


if __name__ == "__main__":
    main()
