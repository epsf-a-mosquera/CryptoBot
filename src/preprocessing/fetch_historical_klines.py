# src/preprocessing/fetch_historical_klines.py
"""
Récupération historique Klines (candles) sur 1 an (par défaut),
avec pagination propre et gestion rate limit (via BinanceClient).

Sortie:
- data/raw/klines/klines_<interval>_<symbol>_lookback_<Nd>_<timestamp>.json

Pourquoi JSON ici?
- Conforme à ta Phase 1 (RAW)
- Tu pourras ensuite nettoyer/normaliser vers Parquet via clean_data.py

Important:
- Pour 1h sur 1 an: ~8760 bougies / symbole => raisonnable en 1 fichier JSON
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Any, Dict, Optional

from src.api.market_data import MarketData
from src.utils.config import (
    SYMBOLS,
    DEFAULT_KLINE_INTERVAL,
    HISTORICAL_LOOKBACK_DAYS,
    RAW_KLINES_DIR,
    MAX_LIMIT,
    ensure_directories,
)

logger = logging.getLogger("fetch_historical_klines")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def paginate_klines(
    market: MarketData,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> List[List[Any]]:
    """
    Paginate sur /klines en avançant startTime à (dernier open_time + 1ms)
    jusqu'à end_ms.
    """
    all_rows: List[List[Any]] = []
    cursor = start_ms

    while True:
        if cursor >= end_ms:
            break

        rows = market.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=cursor,
            end_time=end_ms,
            limit=limit,
        )

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = rows[-1][0]
        next_cursor = int(last_open_time) + 1

        # Sécurité anti boucle infinie
        if next_cursor <= cursor:
            logger.warning(f"{symbol}: pagination cursor did not advance (cursor={cursor}, last={last_open_time}). Stop.")
            break

        cursor = next_cursor

        # Si Binance renvoie moins que limit, c'est probablement la fin de fenêtre
        if len(rows) < limit:
            break

    return all_rows


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"✅ Saved: {path}")


def main() -> None:
    ensure_directories()

    interval = DEFAULT_KLINE_INTERVAL
    lookback_days = HISTORICAL_LOOKBACK_DAYS

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)

    start_ms = dt_to_ms(start_dt)
    end_ms = dt_to_ms(end_dt)

    logger.info(f"Fetching klines for symbols={SYMBOLS}, interval={interval}, lookback_days={lookback_days}")
    logger.info(f"Time window UTC: {start_dt.isoformat()} -> {end_dt.isoformat()}")

    market = MarketData()

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    for symbol in SYMBOLS:
        logger.info(f"➡️  Fetching {symbol} {interval} ...")

        rows = paginate_klines(
            market=market,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=end_ms,
            limit=min(1000, MAX_LIMIT),
        )

        # Format RAW : on garde le tableau Binance tel quel (list of lists),
        # pour rester compatible avec ton clean_data.py actuel.
        out_file = RAW_KLINES_DIR / f"klines_{interval}_{symbol}_lookback_{lookback_days}d_{ts}.json"
        save_json(out_file, rows)

        logger.info(f"✅ {symbol}: fetched {len(rows)} klines")

    logger.info("🎯 Historical klines fetch complete.")


if __name__ == "__main__":
    main()
