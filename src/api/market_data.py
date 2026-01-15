# src/api/market_data.py
"""
Wrapper "Market Data" autour de BinanceClient.
Aucun requests.get() ici: tout passe par BinanceClient.request().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.api.binance_client import BinanceClient
from src.utils.config import MAX_LIMIT


class MarketData:
    def __init__(self, client: Optional[BinanceClient] = None):
        self.client = client or BinanceClient()

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> List[List[Any]]:
        """
        GET /api/v3/klines
        - start_time / end_time en millisecondes (UTC)
        - limit max 1000
        """
        limit = min(int(limit), MAX_LIMIT)

        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        return self.client.request("/klines", params=params)

    def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        return self.client.request("/ticker/24hr", params={"symbol": symbol})

    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        limit = min(int(limit), 1000)
        return self.client.request("/trades", params={"symbol": symbol, "limit": limit})

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        return self.client.get_order_book(symbol, limit)

    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        return self.client.get_ticker_price(symbol)

    def ping(self) -> Dict[str, Any]:
        return self.client.ping()

    def get_server_time(self) -> Dict[str, Any]:
        return self.client.get_server_time()
