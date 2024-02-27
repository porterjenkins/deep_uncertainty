from dataclasses import dataclass


@dataclass
class RegressionMetrics:
    mse: float | None = None
    mae: float | None = None
    mape: float | None = None
    mean_calibration: float | None = None
