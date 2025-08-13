from typing import Dict, Any
import math

class _DummyModel:
    def __init__(self, std: float = 0.3, mean: float = 0.5):
        self._std = std
        self._mean = mean
    def predict_with_uncertainty(self, _market_data: Dict[str, Any]) -> Dict[str, float]:
        return {"mean": self._mean, "std": self._std}

class NeuralEnsembleStrategy:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.ensemble_config = self.config.get("ensemble_config", {})
        self.models = {
            "temporal": _DummyModel(0.2, 0.5),
            "graph": _DummyModel(0.3, 0.5),
            "fundamental": _DummyModel(0.4, 0.5),
        }

    def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        preds = {n: m.predict_with_uncertainty(market_data) for n, m in self.models.items()}
        eps = 1e-6
        inv = {n: 1.0 / (float(p.get("std", 0.3))**2 + eps) for n, p in preds.items()}
        total = sum(inv.values()) or 1.0
        weights = {n: v/total for n, v in inv.items()}
        signal_val = sum(float(p.get("mean", 0.0)) * weights[n] for n, p in preds.items())
        avg_std = sum(float(p.get("std", 0.0)) for p in preds.values()) / max(len(preds), 1)
        confidence = max(0.0, min(1.0, 1.0 - avg_std))
        out = {
            "signal": float(signal_val),
            "confidence": float(confidence),
            "weights": {k: float(v) for k, v in weights.items()},
        }
        for n, w in weights.items():
            out[f"{n}_weight"] = float(w)
        out["direction"] = 1.0 if out["signal"] >= 0 else -1.0
        return out

    def _apply_risk_overlay(self, base_signal: Dict[str, Any], risk_inputs: Dict[str, Any]) -> Dict[str, Any]:
        vix = float(risk_inputs.get("vix", 15.0))
        if vix <= 15: scale = 1.0
        elif vix <= 25: scale = 0.8
        elif vix <= 35: scale = 0.5
        else: scale = 0.3
        direction = float(base_signal.get("direction", 0.0)) * scale
        max_vol = float(self.ensemble_config.get("max_volatility", 1.0))
        if abs(direction) > max_vol:
            direction = math.copysign(max_vol, direction)
        out = dict(base_signal)
        out["direction"] = direction
        return out
