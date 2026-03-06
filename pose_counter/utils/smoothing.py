from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EMA:
    """Exponential Moving Average smoother."""

    alpha: float = 0.25
    value: Optional[float] = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return float(self.value)
