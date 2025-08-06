from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum, auto
import logging
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

class CompoundingStrategy(Enum):
    """Compounding strategy types"""
    FULL_REINVESTMENT = auto()
    PARTIAL_REINVESTMENT = auto()
    FIXED_AMOUNT = auto()
    TARGETED_GROWTH = auto()
    DECAYING_REINVESTMENT = auto()

class WithdrawalStrategy(Enum):
    """Withdrawal strategy types"""
    NONE = auto()
    FIXED_AMOUNT = auto()
    PERCENTAGE = auto()
    TARGET_BALANCE = auto()

@dataclass
class CompoundingResult:
    """Detailed compounding calculation result"""
    new_balance: float
    reinvested_amount: float
    withdrawn_amount: float
    strategy_used: str
    timestamp: datetime
    growth_rate: Optional[float] = None
    projected_balance: Optional[float] = None
    risk_metrics: Optional[Dict[str, float]] = None

class CompoundingEngine:
    """
    Professional-Grade Compounding Engine with:
    - Multiple compounding strategies
    - Withdrawal options
    - Risk-adjusted reinvestment
    - Performance tracking
    - Projection capabilities
    - Tax-aware calculations
    """
    
    def __init__(self,
                 reinvest_rate: float = 0.5,
                 compounding_strategy: CompoundingStrategy = CompoundingStrategy.PARTIAL_REINVESTMENT,
                 withdrawal_strategy: WithdrawalStrategy = WithdrawalStrategy.NONE,
                 withdrawal_amount: float = 0.0,
                 volatility_factor: float = 1.0,
                 tax_rate: float = 0.0):
        """
        Initialize compounding engine
        
        Args:
            reinvest_rate: Percentage of profits to reinvest (0-1)
            compounding_strategy: Strategy for reinvestment
            withdrawal_strategy: Strategy for withdrawals
            withdrawal_amount: Fixed amount or percentage for withdrawals
            volatility_factor: Risk adjustment multiplier (0.5-2.0)
            tax_rate: Percentage tax on profits (0-1)
        """
        self.reinvest_rate = min(max(0.0, reinvest_rate), 1.0)
        self.compounding_strategy = compounding_strategy
        self.withdrawal_strategy = withdrawal_strategy
        self.withdrawal_amount = max(0.0, withdrawal_amount)
        self.volatility_factor = min(max(0.5, volatility_factor), 2.0)
        self.tax_rate = min(max(0.0, tax_rate), 0.5)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._performance_history = []
        self._risk_metrics = {}
        
        # Initialize strategy parameters
        self._strategy_params = {
            CompoundingStrategy.FULL_REINVESTMENT: {
                'description': 'Reinvest 100% of profits',
                'min_reinvest': 1.0
            },
            CompoundingStrategy.PARTIAL_REINVESTMENT: {
                'description': 'Reinvest fixed percentage of profits',
                'base_rate': self.reinvest_rate
            },
            CompoundingStrategy.FIXED_AMOUNT: {
                'description': 'Reinvest fixed amount regardless of profit',
                'fixed_amount': 0.0  # Must be set separately
            },
            CompoundingStrategy.TARGETED_GROWTH: {
                'description': 'Adjust reinvestment to target growth rate',
                'target_growth': 0.05,  # 5% per period
                'max_reinvest': 1.0
            },
            CompoundingStrategy.DECAYING_REINVESTMENT: {
                'description': 'Decrease reinvestment as balance grows',
                'base_rate': self.reinvest_rate,
                'decay_factor': 0.01
            }
        }
        
        # Initialize withdrawal parameters
        self._withdrawal_params = {
            WithdrawalStrategy.NONE: {
                'description': 'No withdrawals'
            },
            WithdrawalStrategy.FIXED_AMOUNT: {
                'description': 'Withdraw fixed amount each period',
                'min_balance': 0.0  # Minimum balance to allow withdrawals
            },
            WithdrawalStrategy.PERCENTAGE: {
                'description': 'Withdraw percentage of profits',
                'cap': 1.0  # Maximum percentage that can be withdrawn
            },
            WithdrawalStrategy.TARGET_BALANCE: {
                'description': 'Withdraw to maintain target balance',
                'target': 0.0  # Must be set separately
            }
        }
        
    def compute(self,
               current_balance: float,
               period_profit: float,
               period_duration: timedelta = timedelta(days=1),
               risk_adjusted: bool = False) -> CompoundingResult:
        """
        Compute new balance after compounding
        
        Args:
            current_balance: Current account balance
            period_profit: Profit for this period
            period_duration: Length of the period
            risk_adjusted: Whether to adjust for volatility
            
        Returns:
            CompoundingResult with detailed calculation
        """
        # Calculate after-tax profit
        taxable_profit = max(0.0, period_profit)
        after_tax_profit = period_profit - (taxable_profit * self.tax_rate)
        
        # Calculate reinvestment amount based on strategy
        reinvest_amount = self._calculate_reinvestment(
            current_balance,
            after_tax_profit,
            period_duration,
            risk_adjusted
        )
        
        # Calculate withdrawal amount based on strategy
        withdrawal_amount = self._calculate_withdrawal(
            current_balance + after_tax_profit,
            reinvest_amount
        )
        
        # Calculate new balance
        new_balance = current_balance + reinvest_amount - withdrawal_amount
        
        # Calculate growth metrics
        growth_rate = (new_balance - current_balance) / max(1.0, current_balance)
        projected_balance = self._project_balance(new_balance, growth_rate)
        
        # Prepare result
        result = CompoundingResult(
            new_balance=new_balance,
            reinvested_amount=reinvest_amount,
            withdrawn_amount=withdrawal_amount,
            strategy_used=self.compounding_strategy.name,
            timestamp=datetime.now(),
            growth_rate=growth_rate,
            projected_balance=projected_balance,
            risk_metrics=self._risk_metrics.copy()
        )
        
        # Update performance history
        self._performance_history.append({
            'timestamp': datetime.now(),
            'balance': new_balance,
            'profit': period_profit,
            'reinvested': reinvest_amount,
            'withdrawn': withdrawal_amount,
            'growth_rate': growth_rate
        })
        
        return result
    
    def _calculate_reinvestment(self,
                              balance: float,
                              profit: float,
                              duration: timedelta,
                              risk_adjusted: bool) -> float:
        """Calculate amount to reinvest based on strategy"""
        params = self._strategy_params[self.compounding_strategy]
        
        if self.compounding_strategy == CompoundingStrategy.FULL_REINVESTMENT:
            return profit * params['min_reinvest']
        
        elif self.compounding_strategy == CompoundingStrategy.PARTIAL_REINVESTMENT:
            base_amount = profit * params['base_rate']
            return self._adjust_for_risk(base_amount, risk_adjusted)
        
        elif self.compounding_strategy == CompoundingStrategy.FIXED_AMOUNT:
            return min(profit, params['fixed_amount'])
        
        elif self.compounding_strategy == CompoundingStrategy.TARGETED_GROWTH:
            required_profit = balance * params['target_growth']
            reinvest_rate = min(1.0, required_profit / max(1.0, profit))
            return profit * reinvest_rate
        
        elif self.compounding_strategy == CompoundingStrategy.DECAYING_REINVESTMENT:
            decay_factor = 1.0 / (1.0 + params['decay_factor'] * balance)
            return profit * params['base_rate'] * decay_factor
        
        return 0.0
    
    def _calculate_withdrawal(self,
                            potential_balance: float,
                            reinvest_amount: float) -> float:
        """Calculate amount to withdraw based on strategy"""
        params = self._withdrawal_params[self.withdrawal_strategy]
        
        if self.withdrawal_strategy == WithdrawalStrategy.NONE:
            return 0.0
        
        elif self.withdrawal_strategy == WithdrawalStrategy.FIXED_AMOUNT:
            return min(potential_balance - reinvest_amount, self.withdrawal_amount)
        
        elif self.withdrawal_strategy == WithdrawalStrategy.PERCENTAGE:
            max_withdrawal = (potential_balance - reinvest_amount) * self.withdrawal_amount
            return min(max_withdrawal, potential_balance * params['cap'])
        
        elif self.withdrawal_strategy == WithdrawalStrategy.TARGET_BALANCE:
            target = params['target']
            if potential_balance > target:
                return min(potential_balance - target, potential_balance - reinvest_amount)
        
        return 0.0
    
    def _adjust_for_risk(self, amount: float, risk_adjusted: bool) -> float:
        """Adjust reinvestment amount based on volatility"""
        if not risk_adjusted:
            return amount
            
        adjusted_amount = amount * self.volatility_factor
        self._risk_metrics = {
            'original_amount': amount,
            'adjusted_amount': adjusted_amount,
            'volatility_factor': self.volatility_factor
        }
        return adjusted_amount
    
    def _project_balance(self, current_balance: float, growth_rate: float) -> float:
        """Project future balance based on current growth rate"""
        periods = 12  # Project 12 periods ahead
        return current_balance * (1.0 + growth_rate) ** periods
    
    def get_performance_history(self,
                              lookback_period: Optional[timedelta] = None) -> pd.DataFrame:
        """Get performance history as DataFrame"""
        df = pd.DataFrame(self._performance_history)
        
        if lookback_period and not df.empty:
            cutoff = datetime.now() - lookback_period
            df = df[df['timestamp'] >= cutoff]
        
        return df
    
    def analyze_performance(self) -> Dict[str, float]:
        """Calculate key performance metrics"""
        if not self._performance_history:
            return {}
            
        df = pd.DataFrame(self._performance_history)
        metrics = {
            'total_growth': (df['balance'].iloc[-1] - df['balance'].iloc[0]) / max(1.0, df['balance'].iloc[0]),
            'average_growth': df['growth_rate'].mean(),
            'max_drawdown': (df['balance'].max() - df['balance'].min()) / max(1.0, df['balance'].max()),
            'profit_factor': df[df['profit'] > 0]['profit'].sum() / max(1.0, abs(df[df['profit'] < 0]['profit'].sum())),
            'withdrawal_ratio': df['withdrawn'].sum() / max(1.0, df['reinvested'].sum())
        }
        
        return metrics
    
    def set_strategy_parameter(self,
                             strategy: CompoundingStrategy,
                             param_name: str,
                             value: float) -> None:
        """Update parameters for a specific strategy"""
        if strategy in self._strategy_params and param_name in self._strategy_params[strategy]:
            self._strategy_params[strategy][param_name] = value
        else:
            raise ValueError(f"Invalid parameter {param_name} for strategy {strategy}")
    
    def set_withdrawal_parameter(self,
                               strategy: WithdrawalStrategy,
                               param_name: str,
                               value: float) -> None:
        """Update parameters for a specific withdrawal strategy"""
        if strategy in self._withdrawal_params and param_name in self._withdrawal_params[strategy]:
            self._withdrawal_params[strategy][param_name] = value
        else:
            raise ValueError(f"Invalid parameter {param_name} for withdrawal strategy {strategy}")
    
    def reset_history(self) -> None:
        """Clear performance history"""
        self._performance_history = []
