# src/api/market_data.py
"""
market_data.py
--------------
Client pour récupérer les données de marché via Binance API.
Utilise BinanceClient pour effectuer les requêtes.
"""

from src.api.binance_client import BinanceClient

class MarketData:
    """
    Classe pour interagir avec les endpoints de marché Binance.
    """

    def __init__(self):
        self.client = BinanceClient()  # Réutilise le client générique

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """
        Récupère les candlesticks (klines) pour un symbole et un intervalle donné.
        """
        endpoint = f"{self.client.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        import requests
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def get_recent_trades(self, symbol, limit=500):
        """
        Récupère les trades récents pour un symbole.
        """
        endpoint = f"{self.client.base_url}/trades"
        params = {"symbol": symbol, "limit": limit}

        import requests
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def get_historical_trades(self, symbol, limit=500, from_id=None):
        """
        Récupère les anciens trades pour un symbole.
        """
        endpoint = f"{self.client.base_url}/historicalTrades"
        params = {"symbol": symbol, "limit": limit}
        if from_id:
            params["fromId"] = from_id

        import requests
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def get_ticker_24h(self, symbol):
        """
        Récupère le ticker 24h pour un symbole.
        """
        endpoint = f"{self.client.base_url}/ticker/24hr"
        params = {"symbol": symbol}

        import requests
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def get_order_book(self, symbol, limit=100):
        """
        Récupère le carnet d'ordres (order book) pour un symbole via market_data.
        """
        return self.client.get_order_book(symbol, limit)
