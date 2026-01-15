# src/preprocessing/clean_data.py
"""
clean_data.py (version "prod")
------------------------------
Nettoyage + fusion des fichiers RAW klines (JSON) en un seul Parquet "clean"
par (symbol, interval).

✅ Ce script :
- Lit tous les JSON klines dans data/raw/klines/
- Groupe par (symbol, interval) en déduisant ces infos depuis le nom de fichier
- Concatène toutes les bougies (pagination / multi-fichiers)
- Déduplique (clé = open_time / timestamp)
- Trie chronologiquement
- Vérifie la continuité temporelle (détecte trous)
- Génère des stats (lignes, duplicats supprimés, trous, période couverte, etc.)
- Sauvegarde :
  - data/processed/<symbol>_<interval>_clean.parquet  (ex: btcusdt_1h_clean.parquet)
  - logs/clean_data_<symbol>_<interval>_stats.json
  - logs/clean_data_prod.log

📌 Hypothèse de format JSON :
- Fichiers klines au format Binance "list of lists", ex:
  [
    [open_time, open, high, low, close, volume, close_time, quote_asset_volume,
     number_of_trades, taker_buy_base_volume, taker_buy_quote_volume, ignore],
    ...
  ]

📌 Nommage supporté (au moins) :
- klines_<interval>_<symbol>_lookback_<Nd>d_<timestamp>.json
- klines_<interval>_<symbol>_<timestamp>.json

Exécution :
    python src/preprocessing/clean_data.py
Optionnel :
    python src/preprocessing/clean_data.py --symbols BTCUSDT,ETHUSDT --interval 1h
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.config import (
    ensure_directories,
    RAW_KLINES_DIR,
    PROCESSED_DIR,
    LOGS_DIR,
    SYMBOLS as CONFIG_SYMBOLS,
    DEFAULT_KLINE_INTERVAL,
)

# =========================
# Logging
# =========================
ensure_directories()
LOG_FILE = LOGS_DIR / "clean_data_prod.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("clean_data_prod")

# Log console aussi (pratique en dev)
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    logger.addHandler(_console)


# =========================
# Utils
# =========================
KLINES_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


def interval_to_pandas_freq(interval: str) -> str:
    """
    Convertit un interval Binance (ex: 1h, 15m) en fréquence pandas (ex: 1H, 15T).
    """
    interval = interval.strip().lower()
    m = re.match(r"^(\d+)([smhdw])$", interval)
    if not m:
        raise ValueError(f"Intervalle non supporté: {interval} (attendu ex: 1h, 15m, 1d)")

    n = int(m.group(1))
    unit = m.group(2)

    mapping = {"s": "S", "m": "T", "h": "H", "d": "D", "w": "W"}
    return f"{n}{mapping[unit]}"


def slug_symbol(symbol: str) -> str:
    return symbol.strip().lower()


def slug_interval(interval: str) -> str:
    return interval.strip().lower()


# =========================
# Fichier -> (symbol, interval)
# =========================
@dataclass(frozen=True)
class KlineFileInfo:
    path: Path
    symbol: str
    interval: str


def parse_symbol_interval_from_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Extrait (symbol, interval) depuis le nom de fichier.
    Exemples acceptés :
      - klines_1h_BTCUSDT_lookback_365d_20260107_120000.json
      - klines_1h_BTCUSDT_20251223_125952.json
    """
    m = re.match(r"^klines_([^_]+)_([^_]+)_lookback_\d+d_\d{8}_\d{6}\.json$", filename)
    if m:
        return m.group(2).upper(), m.group(1).lower()

    m = re.match(r"^klines_([^_]+)_([^_]+)_\d{8}_\d{6}\.json$", filename)
    if m:
        return m.group(2).upper(), m.group(1).lower()

    m = re.match(r"^klines_([^_]+)_([^_]+)_.+\.json$", filename)
    if m:
        return m.group(2).upper(), m.group(1).lower()

    return None


def discover_kline_files(symbols: List[str], interval: str) -> List[KlineFileInfo]:
    """
    Liste tous les fichiers JSON klines dans RAW_KLINES_DIR pour les symboles/interval demandés.
    """
    files: List[KlineFileInfo] = []
    for p in RAW_KLINES_DIR.glob("klines_*.json"):
        parsed = parse_symbol_interval_from_filename(p.name)
        if not parsed:
            continue
        sym, itv = parsed
        if sym in symbols and itv == interval:
            files.append(KlineFileInfo(path=p, symbol=sym, interval=itv))
    return sorted(files, key=lambda x: x.path.name)


# =========================
# Lecture + fusion
# =========================
def read_klines_json(path: Path) -> pd.DataFrame:
    """
    Lit un fichier JSON klines (list of lists) -> DataFrame structuré.
    """
    with open(path, "r") as f:
        raw = json.load(f)

    return pd.DataFrame(raw, columns=KLINES_COLUMNS)


def normalize_klines_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast types, ajoute timestamp UTC, garde colonnes utiles + normalisation.
    """
    keep = ["open_time", "open", "high", "low", "close", "volume", "number_of_trades"]
    df = df[keep].copy()

    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["number_of_trades"] = pd.to_numeric(df["number_of_trades"], errors="coerce").astype("Int64")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["open_time", "timestamp", "open", "high", "low", "close", "volume"])

    df = df[["timestamp", "open_time", "open", "high", "low", "close", "volume", "number_of_trades"]]
    return df


def detect_gaps(df: pd.DataFrame, interval: str) -> Tuple[int, List[pd.Timestamp]]:
    if df.empty:
        return 0, []

    freq = interval_to_pandas_freq(interval)
    start = df["timestamp"].min()
    end = df["timestamp"].max()

    expected = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
    actual = pd.DatetimeIndex(df["timestamp"])
    missing = expected.difference(actual)

    missing_list = list(missing[:500])
    return len(missing), missing_list


def build_stats(
    symbol: str,
    interval: str,
    files_used: List[Path],
    rows_raw: int,
    rows_after_concat: int,
    duplicates_removed: int,
    rows_final: int,
    missing_count: int,
    missing_examples: List[pd.Timestamp],
    df: pd.DataFrame,
) -> Dict:
    if df.empty:
        start_ts = end_ts = None
        start_ms = end_ms = None
    else:
        start_ts = df["timestamp"].min().isoformat()
        end_ts = df["timestamp"].max().isoformat()
        start_ms = int(df["open_time"].min())
        end_ms = int(df["open_time"].max())

    return {
        "symbol": symbol,
        "interval": interval,
        "files_count": len(files_used),
        "files_used": [p.name for p in files_used],
        "rows_raw_total": rows_raw,
        "rows_after_concat": rows_after_concat,
        "duplicates_removed": duplicates_removed,
        "rows_final": rows_final,
        "start_timestamp_utc": start_ts,
        "end_timestamp_utc": end_ts,
        "start_open_time_ms": start_ms,
        "end_open_time_ms": end_ms,
        "missing_timestamps_count": missing_count,
        "missing_timestamps_examples_utc": [t.isoformat() for t in missing_examples],
        "missing_ratio": (missing_count / max(rows_final + missing_count, 1)),
    }


def save_stats(stats: Dict, symbol: str, interval: str) -> Path:
    out = LOGS_DIR / f"clean_data_{slug_symbol(symbol)}_{slug_interval(interval)}_stats.json"
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    return out


def save_clean_parquet(df: pd.DataFrame, symbol: str, interval: str) -> Path:
    out = PROCESSED_DIR / f"{slug_symbol(symbol)}_{slug_interval(interval)}_clean.parquet"
    df.to_parquet(out, index=False)
    return out


def clean_merge_symbol_interval(symbol: str, interval: str) -> Tuple[Optional[Path], Optional[Path]]:
    symbol = symbol.upper()
    interval = interval.lower()

    files_info = discover_kline_files([symbol], interval)
    files = [fi.path for fi in files_info]

    if not files:
        logger.warning(f"Aucun fichier klines trouvé pour {symbol} {interval} dans {RAW_KLINES_DIR}")
        return None, None

    logger.info(f"➡️  {symbol} {interval}: {len(files)} fichier(s) à fusionner")

    dfs: List[pd.DataFrame] = []
    rows_raw_total = 0

    for p in files:
        try:
            df_raw = read_klines_json(p)
            rows_raw_total += len(df_raw)
            df_norm = normalize_klines_df(df_raw)
            dfs.append(df_norm)
        except Exception as e:
            logger.exception(f"Erreur lecture/normalisation {p.name}: {e}")

    if not dfs:
        logger.error(f"{symbol} {interval}: aucun dataframe valide après lecture.")
        return None, None

    df = pd.concat(dfs, ignore_index=True)
    rows_after_concat = len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["open_time"])
    after = len(df)
    duplicates_removed = before - after

    df = df.sort_values("timestamp").reset_index(drop=True)

    # ✅ AJOUT METADATA (indispensable pour Postgres)
    df["symbol"] = symbol
    df["kline_interval"] = interval

    # (Optionnel) ordre des colonnes final
    df = df[
        [
            "symbol",
            "kline_interval",
            "timestamp",
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "number_of_trades",
        ]
    ]

    missing_count, missing_examples = detect_gaps(df, interval)
    if missing_count > 0:
        logger.warning(f"{symbol} {interval}: {missing_count} timestamp(s) manquant(s) détecté(s).")

    stats = build_stats(
        symbol=symbol,
        interval=interval,
        files_used=files,
        rows_raw=rows_raw_total,
        rows_after_concat=rows_after_concat,
        duplicates_removed=duplicates_removed,
        rows_final=len(df),
        missing_count=missing_count,
        missing_examples=missing_examples,
        df=df,
    )

    parquet_path = save_clean_parquet(df, symbol, interval)
    stats_path = save_stats(stats, symbol, interval)

    logger.info(f"✅ Saved clean parquet: {parquet_path} (rows={len(df)})")
    logger.info(f"✅ Saved stats: {stats_path}")

    return parquet_path, stats_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean & merge Binance klines JSON into clean Parquet.")
    parser.add_argument(
        "--symbols",
        type=str,
        default=",".join(CONFIG_SYMBOLS),
        help="Liste de symboles séparés par des virgules (ex: BTCUSDT,ETHUSDT).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_KLINE_INTERVAL,
        help="Intervalle Binance (ex: 1h, 15m).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    interval = args.interval.strip().lower()

    logger.info(f"=== clean_data PROD start === symbols={symbols}, interval={interval}")
    ensure_directories()

    outputs = []
    for sym in symbols:
        parquet_path, stats_path = clean_merge_symbol_interval(sym, interval)
        outputs.append((sym, parquet_path, stats_path))

    ok = [(s, p) for (s, p, _) in outputs if p is not None]
    logger.info(f"=== clean_data PROD done === produced={len(ok)} parquet(s)")

    print("\nRésultats :")
    for sym, p, st in outputs:
        print(f"- {sym} {interval}: parquet={p} | stats={st}")


if __name__ == "__main__":
    main()
