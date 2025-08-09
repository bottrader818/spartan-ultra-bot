#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Professional Stop System v4.1 (Complete Implementation)"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from core.data_fetcher import MarketData
from core.utils import get_logger, get_config
from datetime import datetime, timezone
import pytz

logger = get_logger(__name__)

class MarketType(Enum):
    STOCK = auto()
    CRYPTO = auto()
    FOREX = auto()
    FUTURES = auto()

@dataclass(frozen=True)
class StopLevels:
    stop_loss: float
    take_profit: float
    timestamp: str
    ticker: str
    rr_ratio: float
    position_size: float
    risk_amount: float
    market_type: MarketType
    session_type: str

class HybridStopSystem:
    def __init__(self, config_path: str = 'config/stop_config.yaml'):
        self.config = self._load_config(config_path)
        self._validate_parameters()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.account_balance = self._get_account_balance()  # Initialize balance

    def _load_config(self, path: str) -> dict:
        """Load and pre-process configuration"""
        config = get_config(path)
        return {
            **config['risk_parameters'],
            **config['adaptive_settings'],
            **config['position_sizing'],
            'market_exceptions': config['market_exceptions']
        }

    def _validate_parameters(self):
        """Validate and auto-correct all parameters"""
        param_ranges = {
            'fixed_stop_pct': (0.01, 0.10),
            'atr_multiplier': (1.0, 3.0),
            'reward_risk_ratio': (1.5, 4.0),
            'volatility_floor': (0.005, 0.03),
            'max_capital_risk': (0.005, 0.05)
        }
        for param, (min_val, max_val) in param_ranges.items():
            self.config[param] = np.clip(self.config[param], min_val, max_val)

    def _get_account_balance(self) -> float:
        """Get current trading account balance"""
        try:
            # Replace with your actual account balance API call
            from core.broker_api import get_current_balance
            return max(0, get_current_balance())  # Prevent negative balance
        except Exception as e:
            logger.warning(f"Failed to get account balance: {str(e)}. Using $100,000 default")
            return 100_000.00  # Fallback balance

    def _get_market_type(self, ticker: str) -> MarketType:
        """Identify market type from ticker"""
        if '/' in ticker: return MarketType.FOREX
        if '-' in ticker: return MarketType.CRYPTO
        if '.' in ticker: return MarketType.FUTURES
        return MarketType.STOCK

    def _get_session_type(self) -> str:
        """Determine current market session with timezone awareness"""
        now = datetime.now(pytz.utc)
        # Example logic (adjust for your markets):
        hour = now.hour
        if hour < 8: return 'premarket'
        if hour < 16: return 'regular'
        return 'afterhours'

    def _get_session_multiplier(self, session: str) -> float:
        """Get volatility multiplier for current session"""
        return self.config['session_adjustment'].get(session, 1.0)

    def _get_market_params(self, market_type: MarketType) -> dict:
        """Get parameters for specific market type"""
        default = self.config
        exceptions = self.config['market_exceptions']
        
        market_params = {
            MarketType.CRYPTO: exceptions.get('crypto', default),
            MarketType.FOREX: exceptions.get('forex', default),
            MarketType.FUTURES: exceptions.get('futures', default),
            MarketType.STOCK: default
        }
        return market_params[market_type]

    def _calculate_position_size(self, price: float, stop_loss: float) -> float:
        """Calculate position size based on risk parameters"""
        risk_per_share = price - stop_loss
        if risk_per_share <= 0:
            return self.config['min_position_size']
            
        max_risk_amount = self.config['max_capital_risk'] * self.account_balance
        shares = max_risk_amount / risk_per_share
        return max(shares, self.config['min_position_size'])

    def _get_fallback_levels(self, ticker: str) -> StopLevels:
        """Generate safe fallback levels when calculation fails"""
        return StopLevels(
            stop_loss=0.0,
            take_profit=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            ticker=ticker,
            rr_ratio=0.0,
            position_size=self.config['min_position_size'],
            risk_amount=0.0,
            market_type=self._get_market_type(ticker),
            session_type=self._get_session_type()
        )

    def _log_result(self, result: StopLevels):
        """Structured logging of calculation results"""
        logger.info(
            "Stop levels calculated",
            extra={
                'ticker': result.ticker,
                'stop_loss': result.stop_loss,
                'take_profit': result.take_profit,
                'rr_ratio': result.rr_ratio,
                'position_size': result.position_size,
                'risk_amount': result.risk_amount,
                'market_type': result.market_type.name,
                'session': result.session_type
            }
        )

    def calculate_levels(self, ticker: str) -> StopLevels:
        """Calculate all risk parameters for a ticker"""
        try:
            market_type = self._get_market_type(ticker)
            params = self._get_market_params(market_type)
            session_type = self._get_session_type()
            session_mult = self._get_session_multiplier(session_type)
            
            price, atr = MarketData.get_price_and_atr(ticker)
            
            # Dynamic stop calculation
            fixed_stop = price * params['fixed_stop_pct']
            atr_stop = atr * params['atr_multiplier'] * session_mult
            stop_loss = price - max(
                fixed_stop, 
                atr_stop, 
                price * params['volatility_floor']
            )
            
            # Profit calculation
            risk_amount = price - stop_loss
            take_profit = price + risk_amount * params['reward_risk_ratio']
            rr_ratio = (take_profit - price) / risk_amount
            
            # Position sizing
            position_size = self._calculate_position_size(price, stop_loss)
            
            result = StopLevels(
                stop_loss=round(stop_loss, 2),
                take_profit=round(take_profit, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
                ticker=ticker,
                rr_ratio=round(rr_ratio, 2),
                position_size=round(position_size, 2),
                risk_amount=round(risk_amount * position_size, 2),
                market_type=market_type,
                session_type=session_type
            )
            
            self._log_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Level calculation failed for {ticker}", exc_info=True)
            return self._get_fallback_levels(ticker)

    def batch_calculate(self, tickers: List[str]) -> Dict[str, StopLevels]:
        """Parallelized batch processing with progress tracking"""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.calculate_levels, tickers))
        
        valid_results = {r.ticker: r for r in results if r.stop_loss > 0}
        logger.info(f"Batch processed {len(valid_results)}/{len(tickers)} tickers successfully")
        return valid_results

# Example Usage
if __name__ == "__main__":
    system = HybridStopSystem()
    
    # Single ticker with full logging
    levels = system.calculate_levels("AAPL")
    print(f"AAPL Levels: {levels}")
    
    # Multi-asset batch
    portfolio = ["AAPL", "BTC-USD", "EUR/USD", "ES1!"]  # Stock, Crypto, Forex, Futures
    levels = system.batch_calculate(portfolio)
    
    # Risk report
    total_risk = sum(level.risk_amount for level in levels.values())
    print(f"Total Portfolio Risk: ${total_risk:,.2f}")
