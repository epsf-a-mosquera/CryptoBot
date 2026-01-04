# src/api/create_example_files.py
"""
create_example_files.py
-----------------------
Crée des fichiers d'exemples légers pour les tests
(stockés dans data/examples/).
"""

import json
from pathlib import Path

from src.api.binance_client import BinanceClient
from src.api.market_data import MarketData

EXAMPLES_DIR = Path("data/examples")
EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"


def save_json(data, filename):
    filepath = EXAMPLES_DIR / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Exemple créé : {filepath}")


def main():
    client = BinanceClient()
    market = MarketData()

    # 🔹 Exemple minimal pour tests
    save_json(
        client.get_server_time(),
        "example_server_time.json"
    )

    save_json(
        client.get_ticker_price(SYMBOL),
        "example_ticker_price.json"
    )

    save_json(
        client.get_order_book(SYMBOL, limit=5),
        "example_order_book.json"
    )

    save_json(
        market.get_recent_trades(SYMBOL, limit=5),
        "example_recent_trades.json"
    )

    save_json(
        market.get_klines(SYMBOL, interval="1h", limit=5),
        "example_klines.json"
    )


if __name__ == "__main__":
    main()
