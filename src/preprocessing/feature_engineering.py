# src/preprocessing/feature_engineering.py
"""
feature_engineering.py (version "prod", SANS LABEL)
---------------------------------------------------
Génère des features techniques à partir du fichier clean:
- Input : data/processed/<symbol>_<interval>_clean.parquet
- Output: data/features/<symbol>_<interval>_features.parquet

✅ Points clés:
- Zéro fuite temporelle (features uniquement basées sur <= T)
- Multi-symboles via config.py / CLI
- Logs + contrôles basiques
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
    PROCESSED_DIR,
    FEATURES_DIR,
    SYMBOLS as CONFIG_SYMBOLS,
    DEFAULT_KLINE_INTERVAL,
)

# -------------------------
# Logging
# -------------------------
ensure_directories()
logger = logging.getLogger("feature_engineering_prod")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)


# -------------------------
# Helpers indicators
# -------------------------
def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI (Wilder-style simple version using rolling means)
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(close, fast)
    ema_slow = compute_ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    ATR (Average True Range) - version simple
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(period, min_periods=period).mean()
    return atr


def compute_bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return upper, lower, width


# -------------------------
# Feature generation
# -------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    df attendu (clean):
      timestamp (UTC), open_time (ms), open, high, low, close, volume, number_of_trades
    """

    df = df.copy()

    # Sécurité: tri chronologique
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Base returns (log return 1h)
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["return_1"] = df["close"].pct_change(1)

    # Returns multi-horizon (sur 3,6,12,24h)
    for h in [3, 6, 12, 24]:
        df[f"log_return_{h}"] = np.log(df["close"] / df["close"].shift(h))
        df[f"return_{h}"] = df["close"].pct_change(h)

    # Spreads (intrabar)
    df["hl_spread"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["oc_spread"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)

    # Volume features
    df["vol_chg_1"] = df["volume"].pct_change(1)
    df["vol_ma_24"] = df["volume"].rolling(24, min_periods=24).mean()
    df["vol_z_24"] = (df["volume"] - df["vol_ma_24"]) / df["volume"].rolling(24, min_periods=24).std()

    # Moving averages / EMA
    for p in [10, 20, 50, 100]:
        df[f"sma_{p}"] = df["close"].rolling(p, min_periods=p).mean()
        df[f"ema_{p}"] = compute_ema(df["close"], p)
        df[f"close_over_sma_{p}"] = df["close"] / df[f"sma_{p}"].replace(0, np.nan)
        df[f"close_over_ema_{p}"] = df["close"] / df[f"ema_{p}"].replace(0, np.nan)

    # Volatilité rolling sur log_return_1
    for p in [6, 12, 24, 48, 72]:
        df[f"vol_{p}"] = df["log_return_1"].rolling(p, min_periods=p).std()

    # RSI
    df["rsi_14"] = compute_rsi(df["close"], 14)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"], 12, 26, 9)

    # ATR
    df["atr_14"] = compute_atr(df["high"], df["low"], df["close"], 14)
    df["atr_pct_14"] = df["atr_14"] / df["close"].replace(0, np.nan)

    # Bollinger
    df["bb_upper_20"], df["bb_lower_20"], df["bb_width_20"] = compute_bollinger(df["close"], 20, 2.0)
    df["bb_pos_20"] = (df["close"] - df["bb_lower_20"]) / (df["bb_upper_20"] - df["bb_lower_20"]).replace(0, np.nan)

    # Time features (optionnel mais utile)
    df["hour_utc"] = df["timestamp"].dt.hour
    df["dayofweek_utc"] = df["timestamp"].dt.dayofweek

    return df


def process_symbol(symbol: str, interval: str) -> Path:
    symbol_slug = symbol.lower()
    interval_slug = interval.lower()

    in_path = PROCESSED_DIR / f"{symbol_slug}_{interval_slug}_clean.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    logger.info(f"➡️  Feature engineering: {symbol} {interval} | input={in_path.name}")

    df = pd.read_parquet(in_path)

    # Basic schema check
    required = {"timestamp", "open_time", "open", "high", "low", "close", "volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"{in_path.name}: missing columns {missing_cols}")

    # Add metadata columns
    df["symbol"] = symbol
    df["kline_interval"] = interval

    df_feat = build_features(df)

    # Ne pas dropna agressif ici; on laisse le label script décider du drop final.
    # Mais on enlève les lignes totalement invalides sur le prix.
    df_feat = df_feat.dropna(subset=["close"]).reset_index(drop=True)

    out_path = FEATURES_DIR / f"{symbol_slug}_{interval_slug}_features.parquet"
    df_feat.to_parquet(out_path, index=False)

    logger.info(f"✅ Saved features: {out_path} | rows={len(df_feat)} cols={len(df_feat.columns)}")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature engineering (prod) for Binance klines (no labels).")
    p.add_argument("--symbols", type=str, default=",".join(CONFIG_SYMBOLS), help="BTCUSDT,ETHUSDT,...")
    p.add_argument("--interval", type=str, default=DEFAULT_KLINE_INTERVAL, help="ex: 1h")
    return p.parse_args()


def main() -> None:
    ensure_directories()
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    interval = args.interval.strip().lower()

    logger.info(f"=== feature_engineering PROD start === symbols={symbols} interval={interval}")
    for sym in symbols:
        try:
            process_symbol(sym, interval)
        except Exception as e:
            logger.exception(f"❌ Failed for {sym} {interval}: {e}")

    logger.info("=== feature_engineering PROD done ===")


if __name__ == "__main__":
    main()
