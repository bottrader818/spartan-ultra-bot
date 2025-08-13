# tests/execution/test_alpaca_client.py
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_full_compatibility(async_client):
    """Validate that AlpacaEnhancedClient meets all core test expectations"""
    
    # --- Credential Handling ---
    assert async_client.api_key == "test_key"
    assert async_client.api_secret == "test_secret"
    
    # --- Order Placement ---
    result = await async_client.place_order_async("SPY", "buy", 10)
    assert result.status == "new"  # Critical for other test dependencies
    assert "SPY" in async_client.orders[result.order_id]["symbol"]
    
    # --- WebSocket Event Dispatch ---
    mock_cb = AsyncMock()
    async_client.register_callback("market_data", mock_cb)
    
    await async_client.start_ws()
    await async_client.stop_ws()
    
    mock_cb.assert_called()  # Guaranteed at least one call
