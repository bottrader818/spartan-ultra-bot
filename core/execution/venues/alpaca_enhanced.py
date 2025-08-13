# core/execution/venues/alpaca_enhanced.py
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

@dataclass
class ExecutionResult:
    order_id: str
    status: str = "new"
    filled_qty: float = 0.0
    avg_price: Optional[float] = None
    fees: Optional[float] = None
    error: Optional[str] = None
    raw: Dict[str, Any] | None = None


class AlpacaEnhancedClient:
    """Test‑friendly Alpaca client shim with async WS + order mocks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        cfg = {**(config or {}), **kwargs}

        # Public, test-accessible attributes
        self.api_key: str = cfg.get("api_key", "test_key")
        self.api_secret: str = cfg.get("api_secret", "test_secret")
        self.profile: str = cfg.get("profile", "paper-stocks")
        self.symbols: list[str] = list(cfg.get("symbols", ["SPY"]))
        self.quote_hz: int = int(cfg.get("quote_hz", 10))
        self.chaos: bool = bool(cfg.get("chaos", False))

        # Internal state
        self.orders: Dict[str, Dict[str, Any]] = {}
        self._callbacks: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._ws_task: Optional[asyncio.Task] = None

    @property
    def config(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "profile": self.profile,
            "symbols": self.symbols,
            "quote_hz": self.quote_hz,
            "chaos": self.chaos,
        }

    def register_callback(self, event: str, callback: Callable[[Dict[str, Any]], Any]) -> None:
        self._callbacks[event] = callback

    def place_order(self, symbol: str, side: str, qty: float, **kwargs) -> ExecutionResult:
        order_id = f"ord-{uuid.uuid4().hex[:8]}"
        order = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "status": "new",
            **kwargs,
        }
        self.orders[order_id] = order
        return ExecutionResult(order_id=order_id, status="new", raw=order)

    async def place_order_async(self, symbol: str, side: str, qty: float, **kwargs) -> ExecutionResult:
        await asyncio.sleep(0.01)
        return self.place_order(symbol, side, qty, **kwargs)

    async def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        cb = self._callbacks.get(event)
        if not cb:
            return
        res = cb(payload)
        if asyncio.iscoroutine(res):
            await res

    async def start_ws(self) -> None:
        if self._ws_task and not self._ws_task.done():
            return
        self._ws_task = asyncio.create_task(self._ws_loop())
        # give the loop a tick to deliver the initial message
        await asyncio.sleep(0.05)

    async def _ws_loop(self) -> None:
        # Guaranteed first tick
        if "market_data" in self._callbacks and self.symbols:
            await self._emit(
                "market_data",
                {"symbol": self.symbols[0], "price": 100.0, "ts": time.time()},
            )

        # Periodic ticks
        period = 1.0 / max(1, int(self.quote_hz))
        try:
            while True:
                if "market_data" in self._callbacks and self.symbols:
                    # small variation to avoid identical ticks
                    price = 100.0 + 0.1 * (time.time() % 10)
                    await self._emit(
                        "market_data",
                        {"symbol": self.symbols[0], "price": float(price), "ts": time.time()},
                    )
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            pass  # graceful shutdown

    async def stop_ws(self) -> None:
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
