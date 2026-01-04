# src/api/binance_client.py
"""
binance_client.py
-----------------
Client générique pour interagir avec l'API Binance.
Toutes les requêtes vers l'API passent par ce module.
"""

import requests
from src.utils.config import BINANCE_API_URL

class BinanceClient:
    """
    Client pour interagir avec l'API Binance.
    """

    def __init__(self, base_url=None):
        self.base_url = base_url or BINANCE_API_URL

    def get_server_time(self):
        """
        Vérifie que l'API est accessible et renvoie l'heure serveur.
        Returns:
            dict: {"serverTime": <timestamp>}
        """
        endpoint = f"{self.base_url}/time"
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()

    def get_ticker_price(self, symbol):
        """
        Récupère le dernier prix pour un symbole donné.
        Args:
            symbol (str): Symbole de trading ex: "BTCUSDT"
        Returns:
            dict: {"symbol": "BTCUSDT", "price": "50000.00"}
        """
        endpoint = f"{self.base_url}/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def get_order_book(self, symbol, limit=100):
        """
        Récupère le carnet d'ordres (order book) pour un symbole.
        Args:
            symbol (str): Symbole de trading ex: "BTCUSDT"
            limit (int): Nombre de niveaux à récupérer (max 5000)
        Returns:
            dict: {"lastUpdateId": int, "bids": [...], "asks": [...]}
        """
        endpoint = f"{self.base_url}/depth"
        params = {"symbol": symbol, "limit": limit}
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    def get_exchange_info(self, symbol=None):
        """
        Récupère les informations de l'exchange pour un ou plusieurs symboles.
        Args:
            symbol (str, optional): Symbole de trading ex: "BTCUSDT"
        Returns:
            dict: Informations sur le marché et les symboles
        """
        endpoint = f"{self.base_url}/exchangeInfo"
        params = {"symbol": symbol} if symbol else {}
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
