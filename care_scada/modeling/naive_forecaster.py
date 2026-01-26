""""Naive Forecaster Model Implementation."""
from typing import Optional
import pandas as pd
from care_scada.model_factory import TimeSeriesModel

class NaiveForecaster(TimeSeriesModel):
    """
    Naive forecaster.
    Predicts y_t as the last available observation y_{t-1}.
    """

    def __init__(self):
        self.last_train_value: Optional[
            pd.Series] = None

    def fit(self, train_df: pd.DataFrame, target_col: str):
        """
        Stores the last observed training value.
        """
        if train_df.empty:
            raise ValueError("Training data is empty.")
        if target_col is None:
            raise ValueError("target_col must be specified for NaiveForecaster.")
        if target_col not in train_df.columns:
            raise ValueError(f"target_col '{target_col}' not found in train_df columns.")

        self.last_train_value = train_df[target_col].iloc[-1]

    def predict(self, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Predicts each time step using the previous observation.
        """
        if self.last_train_value is None:
            raise RuntimeError("Model must be fitted before calling predict.")
        if target_col is None:
            raise ValueError("target_col must be specified for NaiveForecaster.")
        if target_col not in test_df.columns:
            raise ValueError(f"target_col '{target_col}' not found in test_df columns.")

        preds = []
        prev = self.last_train_value

        for _, row in test_df.iterrows():
            preds.append(prev)
            prev = row[target_col]  # walk forward using ground truth

        preds_df = pd.DataFrame(
            preds,
            index=test_df.index,
            columns=[target_col]
        )

        return preds_df

    def reset(self):
        self.last_train_value = None
