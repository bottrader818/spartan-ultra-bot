import pytest
import asyncio
import time
import tracemalloc
import psutil
import os
from decimal import Decimal
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import asynccontextmanager
from core.execution.venues.alpaca_enhanced import AlpacaEnhancedClient
from core.execution.base_executor import OrderStatus, ExecutionResult

## ----------------------------
## Advanced Fixtures
## ----------------------------

@pytest.fixture(params=[
    {'paper': True, 'order_size_precision': 2},  # Standard stocks
    {'paper': False, 'order_size_precision': 4}, # Live crypto
    {'max_order_size': 10000, 'min_order_size': 1}  # High volume
], ids=['paper-stocks', 'live-crypto', 'high-volume'])
def config_client(request):
    """Client fixture with various production-like configurations"""
    defaults = {
        'api_key': "test_api_key",
        'secret_key': "test_secret",
        'paper': True,
        'max_order_size': 5000,
        'min_order_size': 10,
        'order_size_precision': 2
    }
    config = {**defaults, **request.param}
    return AlpacaEnhancedClient(**config)

@pytest.fixture
def realtime_market_data():
    """Generate realistic market data feed"""
    def _generate(symbol, base_price, volatility=0.01, count=100):
        for i in range(count):
            yield {
                'type': 'trade',
                'symbol': symbol,
                'bid': round(base_price * (1 - volatility/2 + volatility * (i % 10)/10), 4),
                'ask': round(base_price * (1 + volatility/2 + volatility * (i % 10)/10), 4),
                'bid_size': 100 + (i % 50),
                'ask_size': 100 + (i % 50),
                'timestamp': time.time() + i*0.001
            }
    return _generate

@pytest.fixture
def stress_test_orders():
    """Generate orders for stress testing"""
    symbols = ['AAPL', 'TSLA', 'BTCUSD', 'ETHUSD']
    sides = ['buy', 'sell']
    types = ['market', 'limit', 'stop']
    return [
        {
            'symbol': symbols[i % len(symbols)],
            'side': sides[i % len(sides)],
            'qty': 10 + (i % 20)*5,
            'type': types[i % len(types)],
            'price': 100 + (i % 50) if types[i % len(types)] != 'market' else None
        } for i in range(1000)
    ]

## ----------------------------
## Test Utilities
## ----------------------------

@asynccontextmanager
async def websocket_test_env(client):
    """Context manager for websocket testing"""
    task = asyncio.create_task(client.maintain_websocket())
    try:
        yield client
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

def assert_performance(metrics, thresholds):
    """Assert performance metrics meet thresholds"""
    for metric, (value, threshold) in metrics.items():
        assert value <= threshold, f"{metric} exceeded threshold: {value} > {threshold}"

## ----------------------------
## Enterprise Test Classes
## ----------------------------

class TestAlpacaEnhancedClientCore:
    """Core functionality tests"""
    
    @pytest.mark.asyncio
    async def test_websocket_resilience_with_chaos(self, config_client):
        """Chaos engineering: test websocket under extreme conditions"""
        chaos_sequence = [
            None,  # Success
            Exception("Disconnect"),
            asyncio.TimeoutError(),
            None,
            Exception("Protocol Error"),
            Exception("Rate Limit"),
            None
        ]
        
        with patch('alpaca_trade_api.stream.Stream.connect', 
                 side_effect=chaos_sequence) as mock_connect:
            async with websocket_test_env(config_client):
                await asyncio.sleep(1)
                assert mock_connect.call_count == len(chaos_sequence)
                assert config_client.connection_retries == chaos_sequence.count(Exception)

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_high_frequency_market_data(self, config_client, realtime_market_data):
        """Process high-frequency market data with realistic patterns"""
        symbol = 'AAPL'
        base_price = 150.00
        message_count = 0
        
        start_time = time.perf_counter()
        for message in realtime_market_data(symbol, base_price, count=10000):
            config_client._process_message(message)
            message_count += 1
            
        duration = time.perf_counter() - start_time
        assert message_count == 10000
        assert_performance({
            'message_rate': (message_count/duration, 50000),  # msgs/sec
            'latency_99p': (config_client._latency_stats[-1], 0.01)  # seconds
        })

class TestOrderExecution:
    """Order execution and management tests"""
    
    @pytest.mark.asyncio
    async def test_complex_order_workflow(self, config_client):
        """Complete order lifecycle test"""
        # 1. Submit order
        with patch.object(config_client.rest, 'submit_order') as mock_submit:
            mock_submit.return_value = {'id': 'test_order', 'status': 'accepted'}
            order = await config_client.execute_order({
                'symbol': 'AAPL',
                'side': 'buy',
                'qty': 100,
                'type': 'limit',
                'price': 150.00
            })
            assert order['status'] == 'accepted'
        
        # 2. Partial fill
        await config_client._handle_order_update({
            'event': 'partial_fill',
            'order': {
                'id': 'test_order',
                'status': 'partially_filled',
                'filled_qty': 50,
                'remaining_qty': 50,
                'symbol': 'AAPL'
            }
        })
        
        # 3. Modify order
        with patch.object(config_client.rest, 'replace_order') as mock_replace:
            mock_replace.return_value = {
                'id': 'test_order',
                'status': 'accepted',
                'qty': 75,
                'price': 149.50
            }
            modified = await config_client.modify_order(
                order_id='test_order',
                qty=75,
                price=149.50
            )
            assert modified['qty'] == 75
        
        # 4. Final fill
        await config_client._handle_order_update({
            'event': 'fill',
            'order': {
                'id': 'test_order',
                'status': 'filled',
                'filled_qty': 75,
                'remaining_qty': 0,
                'symbol': 'AAPL'
            }
        })
        
        # Verify final state
        assert 'test_order' not in config_client._active_orders
        assert len(config_client._latency_stats) >= 3  # Submit, modify, fill

    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_order_stress_test(self, config_client, stress_test_orders):
        """Process large volume of diverse orders"""
        with patch.object(config_client.rest, 'submit_order') as mock_submit:
            mock_submit.return_value = {'id': 'test_order', 'status': 'accepted'}
            
            start_time = time.perf_counter()
            tasks = [config_client.execute_order(order) for order in stress_test_orders]
            results = await asyncio.gather(*tasks)
            
            duration = time.perf_counter() - start_time
            assert len(results) == len(stress_test_orders)
            assert_performance({
                'order_rate': (len(results)/duration, 100),  # orders/sec
                'error_rate': (sum(1 for r in results if r['status'] != 'accepted')/len(results), 0.01)
            })

class TestRiskManagement:
    """Risk management and safety checks"""
    
    def test_position_sizing_validation(self, config_client):
        """Test position sizing logic with edge cases"""
        test_cases = [
            # (notional, price, expected_qty, should_pass)
            (10000, 100, 100, True),  # Exact
            (10000, 101, 99.01, True),  # Fractional
            (50000, 100, 500, False),  # Exceeds max
            (5, 100, 0.05, False)  # Below min
        ]
        
        for notional, price, expected_qty, should_pass in test_cases:
            if should_pass:
                qty = config_client.calculate_order_size(price=price, notional=notional)
                assert round(qty, config_client.order_size_precision) == pytest.approx(expected_qty)
            else:
                with pytest.raises(ValueError):
                    config_client.calculate_order_size(price=price, notional=notional)

    @pytest.mark.asyncio
    async def test_risk_checks_on_order(self, config_client):
        """Verify risk checks are applied to orders"""
        risky_order = {
            'symbol': 'AAPL',
            'side': 'buy',
            'qty': config_client.max_order_size * 2,
            'type': 'market'
        }
        
        with patch.object(config_client, '_run_risk_checks', return_value=False) as mock_check:
            with pytest.raises(ValueError, match="Risk check failed"):
                await config_client.execute_order(risky_order)
            mock_check.assert_called_once()

class TestMonitoring:
    """Monitoring and observability tests"""
    
    def test_latency_metrics(self, config_client):
        """Test end-to-end latency tracking"""
        test_data = [
            (0.001, 0.002),  # 1ms
            (0.01, 0.011),    # 10ms
            (0.1, 0.101)      # 100ms
        ]
        
        for sent, received in test_data:
            config_client._record_latency(sent, received)
        
        stats = config_client.get_latency_stats()
        assert stats['min_ms'] == pytest.approx(1, rel=0.1)
        assert stats['max_ms'] == pytest.approx(100, rel=0.1)
        assert stats['avg_ms'] == pytest.approx((1 + 10 + 100)/3, rel=0.1)

    @pytest.mark.asyncio
    async def test_telemetry_integration(self, config_client):
        """Test integration with monitoring systems"""
        with patch('core.monitoring.telemetry.send_metrics') as mock_telemetry:
            await config_client._send_telemetry({
                'orders': 10,
                'latency': 5.5
            })
            mock_telemetry.assert_called_once_with({
                'client': 'alpaca_enhanced',
                'orders': 10,
                'latency': 5.5
            })

## ----------------------------
## Integration Test Classes
## ----------------------------

@pytest.mark.integration
class TestAlpacaIntegration:
    """Integration tests with actual Alpaca API (requires test credentials)"""
    
    @pytest.fixture
    def live_client(self):
        """Client configured with test credentials"""
        return AlpacaEnhancedClient(
            api_key=os.getenv('ALPACA_TEST_KEY'),
            secret_key=os.getenv('ALPACA_TEST_SECRET'),
            paper=True
        )
    
    @pytest.mark.asyncio
    async def test_live_market_data(self, live_client):
        """Test real market data subscription"""
        messages = []
        
        def callback(msg):
            messages.append(msg)
        
        live_client.register_callback(callback)
        
        async with websocket_test_env(live_client):
            await live_client.subscribe_to_symbols(['AAPL'])
            await asyncio.sleep(5)  # Wait for messages
            
        assert len(messages) > 0
        assert any(msg['symbol'] == 'AAPL' for msg in messages)

    @pytest.mark.asyncio
    async def test_live_order_execution(self, live_client):
        """Test real order execution (paper trading)"""
        order = {
            'symbol': 'AAPL',
            'side': 'buy',
            'qty': 1,
            'type': 'limit',
            'price': 1.00,  # Unlikely to execute
            'time_in_force': 'day'
        }
        
        result = await live_client.execute_order(order)
        assert result['status'] in ['accepted', 'pending']
        
        # Cleanup
        await live_client.cancel_order(result['id'])
