import numpy as np


class Predictor:
    """A class that unifies the implementation of prediction across sklearn and skorch models to have uniform `predict`
     and `predict_proba` methods."""

    def __init__(self, model=None):
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        """Enforce returning only a single series."""
        y_pred_proba = self.model.predict_proba(x)
        if y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]

        y_pred_proba = np.clip(y_pred_proba, 1e-5, 1-1e-5)

        return y_pred_proba
