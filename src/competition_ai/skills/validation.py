import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from typing import List, Callable, Any

class ValidationSkill:
    """
    Skill: Robust Validation and Ensembling.
    Extracted from 'Approaching Almost Any NLP Problem' kernel.
    """
    @staticmethod
    def run_stratified_cv(X: np.ndarray, y: np.ndarray, model_factory: Callable, n_splits: int = 5):
        """Standard Stratified K-Fold cross-validation loop."""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(y))
        scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_factory()
            model.fit(X_train, y_train)

            preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = preds

            # Metric calculation (e.g. logloss)
            from sklearn.metrics import log_loss
            score = log_loss(y_val, preds)
            scores.append(score)
            print(f"Fold {fold} score: {score}")

        return oof_preds, np.mean(scores)
