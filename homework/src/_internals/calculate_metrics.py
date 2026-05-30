"""Utility to calculate regression metrics.

Provides `calculate_metrics(x, y, estimator)` returning (mse, mae, r2).
"""
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(x, y, estimator):
    """Calculate MSE, MAE and R2 for given data and estimator.

    Args:
        x: Features (array-like or DataFrame).
        y: True targets (array-like).
        estimator: Fitted estimator with a `predict` method.

    Returns:
        Tuple of (mse, mae, r2).
    """
    y_pred = estimator.predict(x)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, mae, r2
