# src/api/fetch_raw_data.py
"""
fetch_raw_data.py
-----------------
Récupère les données brutes depuis l'API Binance
et les stocke dans data/raw/.
"""

import json
from pathlib import Path
from datetime import datetime

from src.api.binance_client import BinanceClient
from src.api.market_data import MarketData

# Dossiers de sortie
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"


def save_json(data, filename):
    """Sauvegarde un fichier JSON avec timestamp"""
    filepath = RAW_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Fichier enregistré : {filepath}")


def main():
    client = BinanceClient()
    market = MarketData()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # 1️⃣ Server time
    server_time = client.get_server_time()
    save_json(server_time, f"server_time_{timestamp}.json")

    # 2️⃣ Ticker price
    ticker = client.get_ticker_price(SYMBOL)
    save_json(ticker, f"ticker_price_{SYMBOL}_{timestamp}.json")

    # 3️⃣ Order book
    order_book = client.get_order_book(SYMBOL, limit=100)
    save_json(order_book, f"order_book_{SYMBOL}_{timestamp}.json")

    # 4️⃣ Ticker 24h
    ticker_24h = market.get_ticker_24h(SYMBOL)
    save_json(ticker_24h, f"ticker_24h_{SYMBOL}_{timestamp}.json")

    # 5️⃣ Trades récents
    trades = market.get_recent_trades(SYMBOL, limit=100)
    save_json(trades, f"recent_trades_{SYMBOL}_{timestamp}.json")

    # 6️⃣ Klines (1h)
    klines = market.get_klines(SYMBOL, interval="1h", limit=100)
    save_json(klines, f"klines_1h_{SYMBOL}_{timestamp}.json")


if __name__ == "__main__":
    main()
