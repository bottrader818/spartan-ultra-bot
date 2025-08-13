import asyncio, platform, pytest
from core.execution.venues.alpaca_enhanced import AlpacaEnhancedClient

if platform.system() == "Darwin":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception:
        pass

@pytest.fixture
def alpaca_config():
    return {
        "api_key": "test_key",
        "api_secret": "test_secret",
        "profile": "paper-stocks",
        "symbols": ["SPY"],
        "quote_hz": 10,
    }

@pytest.fixture
async def async_client(alpaca_config):
    c = AlpacaEnhancedClient(config=alpaca_config)
    try:
        yield c
    finally:
        if hasattr(c, "stop_ws"):
            await c.stop_ws()
        if hasattr(c, "orders"):
            c.orders.clear()
