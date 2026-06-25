import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from typing import List, Tuple, Dict

class ImprovementPipeline:
    """
    Automated evaluation and calibration pipeline.
    Extracted from NCAA 2025 and ARC evaluation logic.
    """
    @staticmethod
    def calibrate_probabilities(preds: np.ndarray, labels: np.ndarray, k: int = 5) -> UnivariateSpline:
        """
        Uses Univariate Spline to map regression outputs to calibrated probabilities.
        """
        # Sort by predictions
        sorted_indices = np.argsort(preds)
        s_preds = preds[sorted_indices]
        s_labels = labels[sorted_indices]

        # Fit spline
        spline = UnivariateSpline(s_preds, s_labels, k=k)
        return spline

    @staticmethod
    def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculates key competition metrics."""
        from sklearn.metrics import mean_absolute_error, brier_score_loss, roc_auc_score

        metrics = {
            "mae": mean_absolute_error(y_true, y_pred),
            "brier": brier_score_loss(y_true > 0.5, y_pred),
            "auc": roc_auc_score(y_true > 0.5, y_pred)
        }
        return metrics

if __name__ == "__main__":
    pipeline = ImprovementPipeline()
    print("[*] Improvement Pipeline Loaded.")
