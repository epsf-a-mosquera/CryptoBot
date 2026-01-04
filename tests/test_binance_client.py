# tests/test_binance_client.py
import pytest
from src.api.binance_client import BinanceClient

client = BinanceClient()

def test_get_server_time():
    result = client.get_server_time()
    assert "serverTime" in result

def test_get_ticker_price():
    result = client.get_ticker_price("BTCUSDT")
    assert result["symbol"] == "BTCUSDT"
    assert float(result["price"]) > 0

def test_get_order_book():
    result = client.get_order_book("BTCUSDT")
    assert "bids" in result
    assert "asks" in result

def test_get_exchange_info():
    result = client.get_exchange_info("BTCUSDT")
    assert "symbols" in result
