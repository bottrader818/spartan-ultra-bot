class NeuralEnsembleStrategy:
    def generate_signal(self, market_state: MarketState) -> TradingSignal:
        """Ensemble with uncertainty quantification"""
        predictions = {
            name: model.predict_with_uncertainty(market_state)
            for name, model in self.models.items()
        }
        
        combined = self._combine_predictions(predictions)
        if combined['uncertainty'] > self.uncertainty_threshold:
            return self._conservative_signal(combined)
        return self._normal_signal(combined)
    
    def _update_weights(self):
        """Bayesian optimization of model weights"""
        recent_perf = self._get_performance_metrics()
        self.ensemble_weights = self._bayesian_optimizer.optimize(recent_perf)
