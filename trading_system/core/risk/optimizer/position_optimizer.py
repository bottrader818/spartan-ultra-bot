class PositionOptimizer:
    def calculate_position_size(self, metrics: StrategyMetrics) -> float:
        """Hybrid risk model with regime detection"""
        if self._current_regime() == 'high_volatility':
            return self._conservative_size(metrics)
        return self._dynamic_size(metrics)
    
    def _dynamic_size(self, metrics: StrategyMetrics) -> float:
        """Adaptive sizing based on multiple factors"""
        base_size = self._base_model_size(metrics)
        return base_size * self._portfolio_risk * self._liquidity_factor(metrics.symbol)
