from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ExecutionResult:
    order_id: str
    status: str                 # "filled" | "partial" | "rejected" | "submitted"
    filled_qty: float = 0.0
    avg_price: float = 0.0
    venue: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
