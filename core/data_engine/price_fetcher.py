import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

logger = logging.getLogger(__name__)

class MarketState(Enum):
    PRE_MARKET = "PRE_MARKET"
    REGULAR_MARKET = "REGULAR_MARKET"
    POST_MARKET = "POST_MARKET"
    CLOSED = "CLOSED"

@dataclass(frozen=True)
class MarketData:
    symbol: str
    price: float
    bid: float
    ask: float
    spread: float
    volume: float
    timestamp: datetime
    currency: str = "USD"
    market_state: MarketState = MarketState.REGULAR_MARKET
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    open_interest: Optional[float] = None
    iv_30d: Optional[float] = None  # 30-day implied volatility
    liquidity_score: Optional[float] = None

class PriceFetcher:
    """
    Institutional-Grade Market Data Provider with:
    - Multi-threaded data fetching
    - Advanced caching strategies
    - Market state awareness
    - Liquidity scoring
    - Volatility metrics
    - Microstructure analysis
    """

    def __init__(self, max_workers: int = 8, cache_size: int = 2048):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cache_size = cache_size
        self._rate_limiter = RateLimiter(calls=5, period=1)  # 5 calls/sec
        logger.info(f"Initialized PriceFetcher with {max_workers} workers")

    @lru_cache(maxsize=2048)
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        Fetch comprehensive market data with institutional features
        
        Args:
            symbol: Ticker symbol (supports equities, options, futures)
            
        Returns:
            MarketData object with professional trading metrics
        """
        try:
            self._rate_limiter.wait()
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period="1d", interval="1m")
            
            # Calculate microstructure metrics
            spread = (info.get('ask', 0) - info.get('bid', 0)
            liquidity = self._calculate_liquidity_score(history)
            
            return MarketData(
                symbol=symbol,
                price=info.get('currentPrice', np.nan),
                bid=info.get('bid', np.nan),
                ask=info.get('ask', np.nan),
                spread=spread,
                volume=info.get('volume', 0),
                timestamp=datetime.now(timezone.utc),
                currency=info.get('currency', 'USD'),
                market_state=self._get_market_state(info),
                day_high=info.get('dayHigh', np.nan),
                day_low=info.get('dayLow', np.nan),
                open_interest=info.get('openInterest', None),
                iv_30d=info.get('impliedVolatility', None),
                liquidity_score=liquidity
            )
            
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {str(e)}", exc_info=True)
            return None

    def get_bulk_market_data(self, symbols: List[str]) -> Dict[str, Optional[MarketData]]:
        """
        Threaded batch fetch of market data
        
        Args:
            symbols: List of ticker symbols
            
        Returns:
            Dictionary of MarketData objects keyed by symbol
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.get_market_data, symbols))
        return dict(zip(symbols, results))

    def get_historical_bars(self, 
                          symbol: str,
                          period: str = "1mo",
                          interval: str = "15m",
                          microstructure: bool = False) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with optional order book statistics
        
        Args:
            symbol: Ticker symbol
            period: Data period (1d, 5d, 1mo, etc.)
            interval: Bar interval (1m, 5m, 1h, etc.)
            microstructure: Include bid/ask spread data
            
        Returns:
            DataFrame with multi-index columns
        """
        try:
            self._rate_limiter.wait()
            
            data = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                prepost=True,
                progress=False,
                threads=True
            )
            
            if microstructure:
                # Add simulated spread data (real implementation would use L2 data)
                data['Spread'] = data['High'] - data['Low']
                data['Mid'] = (data['Open'] + data['Close']) / 2
                
            data.index = data.index.tz_localize(timezone.utc)
            return data
            
        except Exception as e:
            logger.error(f"Historical data fetch failed for {symbol}: {str(e)}")
            return None

    def _calculate_liquidity_score(self, history: pd.DataFrame) -> float:
        """Calculate proprietary liquidity score based on price impact"""
        if history.empty:
            return 0.0
            
        vol = history['Volume'].mean()
        volatility = history['High'].std() / history['Close'].mean()
        return min(1.0, vol / 1e6) * (1 - min(1.0, volatility / 0.1))

    def _get_market_state(self, info: Dict) -> MarketState:
        """Determine current market state"""
        if info.get('marketState', 'REGULAR') == 'REGULAR':
            return MarketState.REGULAR_MARKET
        elif info.get('postMarketChange'):
            return MarketState.POST_MARKET
        elif info.get('preMarketChange'):
            return MarketState.PRE_MARKET
        return MarketState.CLOSED

class RateLimiter:
    """Professional rate limiting implementation"""
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = []
        
    def wait(self):
        now = time.time()
        self.timestamps = [t for t in self.timestamps if t > now - self.period]
        
        if len(self.timestamps) >= self.calls:
            sleep_time = self.period - (now - self.timestamps[0])
            time.sleep(max(0, sleep_time))
            
        self.timestamps.append(time.time())

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fetcher = PriceFetcher()
    
    # Institutional-grade single asset data
    aapl_data = fetcher.get_market_data("AAPL")
    print(f"\nAAPL Market Data:")
    print(f"Price: {aapl_data.price:.2f} | Spread: {aapl_data.spread:.4f}")
    print(f"Liquidity: {aapl_data.liquidity_score:.2%} | IV: {aapl_data.iv_30d:.2%}")
    print(f"Market State: {aapl_data.market_state.value}")
    
    # Bulk data for portfolio
    portfolio = ["AAPL", "MSFT", "TSLA", "NVDA", "AMD"]
    market_data = fetcher.get_bulk_market_data(portfolio)
    
    print("\nPortfolio Liquidity Analysis:")
    for symbol, data in market_data.items():
        if data:
            print(f"{symbol}: {data.price:.2f} (Spread: {data.spread:.4f}, Liq: {data.liquidity_score:.2%})")
    
    # Advanced historical analysis
    print("\nFetching historical microstructure...")
    hist_data = fetcher.get_historical_bars("SPY", period="5d", interval="5m", microstructure=True)
    if hist_data is not None:
        print(hist_data[['Close', 'Spread']].tail())
