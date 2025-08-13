from abc import ABC, abstractmethod

class StrategyBase(ABC):
    def __init__(self, config=None):
        self.config = config or {}
        self.position = 0  # +1 long, -1 short, 0 flat

    @abstractmethod
    def on_bar(self, bar):  # bar: dict with 'close', etc.
        ...

    def size(self, equity: float) -> int:
        # simple fixed fraction sizing
        risk_frac = float(self.config.get("risk_frac", 0.01))
        price = bar_price = self.last_price if hasattr(self, "last_price") else 100.0
        units = max(1, int((equity * risk_frac) / max(price, 1e-6)))
        return units
