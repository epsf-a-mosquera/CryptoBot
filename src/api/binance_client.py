# src/api/binance_client.py
"""
Client générique Binance Spot REST (public endpoints).
- Centralise TOUTES les requêtes HTTP
- Gère timeout, retries, backoff, 429/418 (Retry-After), 5xx, erreurs réseau
- Logue les headers X-MBX-USED-WEIGHT-* pour suivi du rate limit
"""

from __future__ import annotations

import time
import random
import logging
from typing import Any, Dict, Optional

import requests
from requests import Response

from src.utils.config import (
    BINANCE_API_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    BACKOFF_BASE_SECONDS,
    BACKOFF_MAX_SECONDS,
    DEBUG,
)


class BinanceHTTPError(RuntimeError):
    """Erreur HTTP Binance non récupérable (4xx hors 429/418, etc.)."""


class BinanceClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = REQUEST_TIMEOUT,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = (base_url or BINANCE_API_URL).rstrip("/")
        self.timeout = timeout

        self.session = session or requests.Session()
        # Header minimal; User-Agent utile pour logs côté proxy / debug
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "CryptoBot/1.0 (DataEngineering Project)",
            }
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        if DEBUG and not self.logger.handlers:
            # évite double handlers si import multiple
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s"))
            self.logger.addHandler(h)
        self.logger.setLevel(logging.INFO if DEBUG else logging.WARNING)

    def _sleep_backoff(self, attempt: int, retry_after: Optional[str] = None) -> None:
        """
        Backoff exponentiel + jitter.
        Si Retry-After est fourni (cas 429/418), on le respecte.
        """
        if retry_after:
            try:
                wait_s = float(retry_after)
            except ValueError:
                wait_s = None
            if wait_s is not None and wait_s > 0:
                self.logger.warning(f"Rate limited. Sleeping Retry-After={wait_s}s")
                time.sleep(min(wait_s, BACKOFF_MAX_SECONDS))
                return

        # backoff exponentiel borné + jitter
        base = BACKOFF_BASE_SECONDS * (2 ** attempt)
        jitter = random.uniform(0.0, 0.25 * base)
        wait_s = min(base + jitter, BACKOFF_MAX_SECONDS)
        self.logger.warning(f"Retrying with backoff: sleep {wait_s:.2f}s (attempt={attempt})")
        time.sleep(wait_s)

    def _log_rate_limit_headers(self, resp: Response) -> None:
        """
        Binance renvoie des headers de consommation (poids) du rate limit.
        Exemple: X-MBX-USED-WEIGHT-1m, X-MBX-USED-WEIGHT
        """
        used = {k: v for k, v in resp.headers.items() if k.lower().startswith("x-mbx-used-weight")}
        if used:
            self.logger.info(f"RateLimit headers: {used}")

    def request(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = "GET",
    ) -> Any:
        """
        Requête générique vers Binance.

        Gère :
        - timeouts
        - erreurs réseau
        - 429 (too many requests) => backoff + Retry-After si présent
        - 418 (IP banned) => respecter Retry-After
        - 5xx => retry
        - autres 4xx => exception
        """
        method = method.upper().strip()
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.base_url}{path}"

        last_err: Optional[Exception] = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                t0 = time.time()
                resp = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    timeout=self.timeout,
                )
                dt_ms = (time.time() - t0) * 1000

                self.logger.info(f"{method} {url} params={params} -> {resp.status_code} ({dt_ms:.0f}ms)")
                self._log_rate_limit_headers(resp)

                # OK
                if 200 <= resp.status_code < 300:
                    # Certaines réponses peuvent être vides; ici Binance renvoie du JSON sur endpoints market data
                    return resp.json()

                # Rate limit
                if resp.status_code in (429, 418):
                    retry_after = resp.headers.get("Retry-After")
                    self.logger.warning(f"HTTP {resp.status_code} (rate limit / ban). Retry-After={retry_after}")
                    self._sleep_backoff(attempt, retry_after=retry_after)
                    continue

                # 5xx (problème serveur)
                if 500 <= resp.status_code < 600:
                    self.logger.warning(f"HTTP {resp.status_code} (server error). Will retry.")
                    self._sleep_backoff(attempt)
                    continue

                # autres 4xx => non récupérable ici
                try:
                    payload = resp.json()
                except Exception:
                    payload = resp.text
                raise BinanceHTTPError(f"HTTP {resp.status_code} error for {url}: {payload}")

            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = e
                self.logger.warning(f"Network error: {repr(e)}")
                self._sleep_backoff(attempt)
                continue
            except Exception as e:
                last_err = e
                break

        raise RuntimeError(f"Binance request failed after retries. Last error: {repr(last_err)}")

    # ====== Endpoints généraux ======
    def ping(self) -> Dict[str, Any]:
        return self.request("/ping")

    def get_server_time(self) -> Dict[str, Any]:
        return self.request("/time")

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        params = {"symbol": symbol} if symbol else None
        return self.request("/exchangeInfo", params=params)

    # ====== Endpoints market data utiles ======
    def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        return self.request("/ticker/price", params={"symbol": symbol})

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        return self.request("/depth", params={"symbol": symbol, "limit": limit})
