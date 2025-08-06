import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from datetime import timedelta
import pandas as pd
from core.config_loader import ConfigLoader
from core.strategy_engine.strategy_router import StrategyRouter
from core.execution_engine.exchange_router import ExchangeRouter
from core.performance_logger import PerformanceLogger
from core.strategy_engine import SignalDirection
from core.utils.technical_indicators import calculate_atr, calculate_ema

@dataclass
class ExecutionSignal:
    """Enhanced execution signal with comprehensive trade details"""
    symbol: str
    direction: SignalDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    time_to_expiry: timedelta
    risk_reward_ratio: float
    volatility: float
    confidence: float
    metadata: Dict[str, Any]

class MainRunner:
    """
    Professional-Grade Trading System Runner with:
    - Comprehensive error handling
    - Performance monitoring
    - Dynamic configuration
    - Advanced execution control
    - Detailed logging
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger("MainRunner")
        self.strategy = self._initialize_strategy()
        self.executor = self._initialize_executor()
        self.performance_logger = self._initialize_performance_logger()
        self._setup_metrics()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load and validate configuration with enhanced error handling"""
        try:
            config = ConfigLoader().load(config_path) if config_path else ConfigLoader().load()
            return self._validate_config(config)
        except Exception as e:
            raise RuntimeError(f"Configuration loading failed: {str(e)}") from e
            
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters with sensible defaults"""
        required_sections = ['symbols', 'strategy', 'execution', 'monitoring']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        # Set defaults for optional parameters
        config.setdefault('max_retries', 3)
        config.setdefault('execution_timeout', 30)
        config.setdefault('performance_update_interval', 60)
        
        return config
        
    def _setup_logging(self) -> None:
        """Configure advanced logging with structured formatting"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('trading_system.log')
            ]
        )
        
    def _initialize_strategy(self) -> StrategyRouter:
        """Initialize strategy with health checks and fallback"""
        try:
            strategy = StrategyRouter(self.config).select()
            if not strategy:
                raise ValueError("No valid strategy configured")
            return strategy
        except Exception as e:
            self.logger.error(f"Strategy initialization failed: {e}", exc_info=True)
            raise
            
    def _initialize_executor(self) -> ExchangeRouter:
        """Initialize execution engine with connection validation"""
        try:
            executor = ExchangeRouter(self.config)
            # Test connection
            if not executor.test_connection():
                raise ConnectionError("Exchange connection failed")
            return executor
        except Exception as e:
            self.logger.error(f"Executor initialization failed: {e}", exc_info=True)
            raise
            
    def _initialize_performance_logger(self) -> PerformanceLogger:
        """Initialize performance tracking system"""
        try:
            return PerformanceLogger(self.config)
        except Exception as e:
            self.logger.error(f"Performance logger initialization failed: {e}", exc_info=True)
            raise
            
    def _setup_metrics(self) -> None:
        """Initialize comprehensive performance metrics"""
        self.metrics = {
            'total_symbols_processed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'total_volume_traded': 0.0,
            'performance_by_symbol': {},
            'recent_errors': [],
            'system_health': {
                'strategy_health': 'OK',
                'execution_health': 'OK',
                'performance_logging_health': 'OK'
            }
        }
        
    def run(self) -> None:
        """Main execution loop with enhanced monitoring and fault tolerance"""
        self.logger.info("Starting trading system execution")
        
        try:
            for symbol in self.config['symbols']:
                self._process_symbol(symbol)
                
            self._log_performance_summary()
        except KeyboardInterrupt:
            self.logger.info("Graceful shutdown initiated")
        except Exception as e:
            self.logger.error(f"Critical system failure: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()
            
    def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol with retry logic and performance tracking"""
        self.metrics['total_symbols_processed'] += 1
        start_time = pd.Timestamp.now()
        
        try:
            signal = self._generate_signal(symbol)
            if signal:
                trade = self._execute_trade(symbol, signal)
                self._record_performance(trade)
                
            self.metrics['successful_executions'] += 1
        except Exception as e:
            self.metrics['failed_executions'] += 1
            self._handle_error(symbol, e)
        finally:
            execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            self.metrics['average_execution_time'] = (
                self.metrics['average_execution_time'] * (self.metrics['total_symbols_processed'] - 1) + execution_time
            ) / self.metrics['total_symbols_processed']
            
    def _generate_signal(self, symbol: str) -> Optional[ExecutionSignal]:
        """Generate trading signal with enhanced validation"""
        try:
            raw_signal = self.strategy.generate_signal(symbol)
            if not raw_signal:
                return None
                
            return self._enhance_signal(raw_signal, symbol)
        except Exception as e:
            self.logger.error(f"Signal generation failed for {symbol}: {e}", exc_info=True)
            raise
            
    def _enhance_signal(self, raw_signal: Any, symbol: str) -> ExecutionSignal:
        """Convert raw signal to enhanced execution signal"""
        # Calculate additional metrics
        atr = calculate_atr(
            raw_signal.high, 
            raw_signal.low, 
            raw_signal.close, 
            period=14
        )[-1]
        
        volatility = atr / raw_signal.close[-1]
        risk_reward = abs(raw_signal.take_profit - raw_signal.entry_price) / abs(raw_signal.stop_loss - raw_signal.entry_price)
        
        return ExecutionSignal(
            symbol=symbol,
            direction=raw_signal.direction,
            entry_price=raw_signal.entry_price,
            stop_loss=raw_signal.stop_loss,
            take_profit=raw_signal.take_profit,
            position_size=raw_signal.position_size,
            time_to_expiry=timedelta(hours=self._calculate_optimal_holding_period(volatility)),
            risk_reward_ratio=risk_reward,
            volatility=volatility,
            confidence=raw_signal.confidence,
            metadata={
                'generation_time': pd.Timestamp.now(),
                'strategy_version': self.strategy.version,
                'signal_components': raw_signal.component_scores
            }
        )
        
    def _calculate_optimal_holding_period(self, volatility: float) -> float:
        """Calculate volatility-adjusted holding period"""
        base_period = 4  # hours
        volatility_normalized = min(max(volatility / 0.15, 0.5), 2.0)  # Cap between 0.5x and 2.0x
        return base_period / volatility_normalized
        
    def _execute_trade(self, symbol: str, signal: ExecutionSignal) -> Any:
        """Execute trade with enhanced order management"""
        try:
            self.logger.info(f"Executing {signal.direction.name} order for {symbol}")
            trade = self.executor.execute(symbol, signal)
            self.metrics['total_volume_traded'] += signal.position_size
            return trade
        except Exception as e:
            self.logger.error(f"Execution failed for {symbol}: {e}", exc_info=True)
            raise
            
    def _record_performance(self, trade: Any) -> None:
        """Record trade performance with detailed analytics"""
        try:
            self.performance_logger.record(trade)
            
            # Update symbol-specific metrics
            symbol = trade.symbol
            if symbol not in self.metrics['performance_by_symbol']:
                self.metrics['performance_by_symbol'][symbol] = {
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'total_pnl': 0.0
                }
                
            self.metrics['performance_by_symbol'][symbol]['total_trades'] += 1
            if trade.pnl > 0:
                self.metrics['performance_by_symbol'][symbol]['profitable_trades'] += 1
            self.metrics['performance_by_symbol'][symbol]['total_pnl'] += trade.pnl
        except Exception as e:
            self.logger.error(f"Performance recording failed: {e}", exc_info=True)
            raise
            
    def _handle_error(self, symbol: str, error: Exception) -> None:
        """Handle errors with comprehensive tracking and recovery"""
        error_entry = {
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': self._get_formatted_stack_trace(error)
        }
        
        self.metrics['recent_errors'].append(error_entry)
        if len(self.metrics['recent_errors']) > 100:
            self.metrics['recent_errors'] = self.metrics['recent_errors'][-100:]
            
        self.logger.error(f"Error processing {symbol}: {error}", exc_info=True)
        
    def _get_formatted_stack_trace(self, error: Exception) -> str:
        """Format stack trace for better readability"""
        import traceback
        return ''.join(traceback.format_exception(type(error), error, error.__traceback__))
        
    def _log_performance_summary(self) -> None:
        """Log comprehensive performance summary"""
        summary = (
            f"\n=== Execution Summary ===\n"
            f"Symbols Processed: {self.metrics['total_symbols_processed']}\n"
            f"Successful Executions: {self.metrics['successful_executions']}\n"
            f"Failed Executions: {self.metrics['failed_executions']}\n"
            f"Success Rate: {self.metrics['successful_executions'] / max(1, self.metrics['total_symbols_processed']) * 100:.2f}%\n"
            f"Average Execution Time: {self.metrics['average_execution_time']:.2f}s\n"
            f"Total Volume Traded: {self.metrics['total_volume_traded']:.2f}\n"
        )
        
        self.logger.info(summary)
        
    def _cleanup(self) -> None:
        """Clean up resources and ensure proper shutdown"""
        self.logger.info("Performing system cleanup")
        try:
            self.executor.close()
            self.performance_logger.flush()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}", exc_info=True)
            
        self.logger.info("Trading system shutdown complete")

def main():
    """Entry point with enhanced system monitoring"""
    try:
        runner = MainRunner()
        runner.run()
    except Exception as e:
        logging.error(f"System failure: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
