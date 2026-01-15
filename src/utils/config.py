# src/utils/config.py
"""
Configuration centrale du projet CryptoBot.

Objectifs:
- Avoir une seule source de vérité pour:
  - symboles, timeframe
  - chemins de données
  - paramètres réseau / retries
- Permettre l'exécution multi-symboles en gardant le code simple.

Note:
- Tous les endpoints utilisés ici sont publics (pas de clé API requise).
"""

from __future__ import annotations

import os
from pathlib import Path

# =========================
# Root paths (robuste)
# =========================
# .../CryptoBot/src/utils/config.py -> parents[2] = .../CryptoBot
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_KLINES_DIR = RAW_DIR / "klines"
RAW_LIVE_DIR = RAW_DIR / "live"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
EXAMPLES_DIR = DATA_DIR / "examples"
LOGS_DIR = ROOT_DIR / "logs"

def ensure_directories() -> None:
    """Crée les répertoires nécessaires si absents."""
    for d in [
        DATA_DIR, RAW_DIR, RAW_KLINES_DIR, RAW_LIVE_DIR,
        PROCESSED_DIR, FEATURES_DIR, EXAMPLES_DIR, LOGS_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)

# =========================
# Binance API
# =========================
# Base officielle (Spot REST): /api/v3/...
BINANCE_API_URL = os.getenv("BINANCE_API_URL", "https://api.binance.com/api/v3")

# =========================
# Paramètres projet (Phase 1 / 2)
# =========================
# Multi-symboles: tu peux ajouter/enlever ici, tous les scripts doivent s'appuyer dessus
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
SYMBOLS = [s.strip().upper() for s in SYMBOLS if s.strip()]

# Timeframe principal
DEFAULT_KLINE_INTERVAL = os.getenv("DEFAULT_KLINE_INTERVAL", "1h")

# Historique: 1 an par défaut (365 jours)
HISTORICAL_LOOKBACK_DAYS = int(os.getenv("HISTORICAL_LOOKBACK_DAYS", "365"))

# Limites Binance (klines: max 1000)
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "1000"))

# =========================
# Paramètres HTTP / robustesse
# =========================
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "15"))

# Retry/backoff sur erreurs réseau + 429/418 + 5xx
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "8"))
BACKOFF_BASE_SECONDS = float(os.getenv("BACKOFF_BASE_SECONDS", "1.0"))
BACKOFF_MAX_SECONDS = float(os.getenv("BACKOFF_MAX_SECONDS", "60"))

# Debug logs
DEBUG = os.getenv("DEBUG", "true").lower() in {"1", "true", "yes", "y"}
