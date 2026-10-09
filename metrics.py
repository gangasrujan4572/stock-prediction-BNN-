"""
Evaluation Metrics Module
Implements point forecast metrics and uncertainty quantification metrics as defined in
Section 2.6 and 2.7 of the paper:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- Prediction Interval Coverage Probability (PICP)
- Mean Prediction Interval Width (MPIW)
"""

import numpy as np
from typing import Dict, Tuple

def compute_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and MSE (Equations 17, 18, 19).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    diff = y_true - y_pred
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse
    }

def compute_uncertainty_metrics(
    y_true: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray
) -> Dict[str, float]:
    """
    Computes PICP and MPIW (Equations 20 and 21):
    - PICP: proportion of actual values falling inside [lower_ci, upper_ci]
    - MPIW: average width of confidence intervals (upper_ci - lower_ci)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    lower_ci = np.asarray(lower_ci, dtype=np.float64)
    upper_ci = np.asarray(upper_ci, dtype=np.float64)
    
    # Coverage indicator: 1 if lower <= y <= upper else 0
    covered = (y_true >= lower_ci) & (y_true <= upper_ci)
    picp = float(np.mean(covered.astype(np.float64)))
    
    # Width of intervals
    widths = upper_ci - lower_ci
    mpiw = float(np.mean(widths))
    
    return {
        "PICP": picp,
        "MPIW": mpiw
    }

def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower_ci: np.ndarray = None,
    upper_ci: np.ndarray = None
) -> Dict[str, float]:
    """
    Computes all point and uncertainty metrics simultaneously.
    """
    results = compute_point_metrics(y_true, y_pred)
    if lower_ci is not None and upper_ci is not None:
        u_metrics = compute_uncertainty_metrics(y_true, lower_ci, upper_ci)
        results.update(u_metrics)
    return results
