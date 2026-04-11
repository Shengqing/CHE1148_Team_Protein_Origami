from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float
    kendall: float


def kendall_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # pandas handles ties well; avoids scipy dependency
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="kendall"))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    sp = float(kendall_corr(y_true, y_pred))
    return Metrics(mae=mae, rmse=rmse, r2=r2, kendall=sp)


def metrics_to_dict(m: Metrics) -> Dict[str, float]:
    return {"MAE": m.mae, "RMSE": m.rmse, "R2": m.r2, "Kendall": m.kendall}
