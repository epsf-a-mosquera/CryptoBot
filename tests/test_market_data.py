# tests/test_market_data.py
import pytest
from src.api.market_data import MarketData

SYMBOL = "BTCUSDT"

@pytest.fixture
def market_data_client():
    return MarketData()

def test_get_klines(market_data_client):
    result = market_data_client.get_klines(SYMBOL, interval="1m", limit=5)
    assert isinstance(result, list)
    assert len(result) > 0
    assert len(result[0]) >= 6  # Chaque kline a au moins 6 éléments

def test_get_recent_trades(market_data_client):
    result = market_data_client.get_recent_trades(SYMBOL, limit=5)
    assert isinstance(result, list)
    if result:
        assert "price" in result[0]
        assert "qty" in result[0]

def test_get_historical_trades(market_data_client):
    result = market_data_client.get_historical_trades(SYMBOL, limit=5)
    assert isinstance(result, list)
    if result:
        assert "price" in result[0]
        assert "qty" in result[0]

def test_get_ticker_24h(market_data_client):
    result = market_data_client.get_ticker_24h(SYMBOL)
    assert isinstance(result, dict)
    assert result["symbol"] == SYMBOL
    assert "lastPrice" in result

def test_get_order_book(market_data_client):
    result = market_data_client.get_order_book(SYMBOL, limit=5)
    assert isinstance(result, dict)
    assert "bids" in result
    assert "asks" in result
    assert len(result["bids"]) > 0
    assert len(result["asks"]) > 0
