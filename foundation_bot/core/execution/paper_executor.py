from dataclasses import dataclass

@dataclass
class FillReport:
    filled_qty: float
    avg_price: float
    status: str  # 'filled' | 'flat'

class PaperExecutor:
    def submit(self, side: str, qty: float, price: float) -> FillReport:
        if qty <= 0:
            return FillReport(0.0, price, "flat")
        # Naive: assume immediate fill at given price
        return FillReport(qty, price, "filled")
