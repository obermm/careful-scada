"""Abstract Time Series Model and Model Factory."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Identifiers / bookkeeping columns that carry no physical information about the turbine
METADATA_COLUMNS = {"asset_id", "id", "status_type_id"}
_STAT_SUFFIXES = ("_avg", "_min", "_max", "_std")


def _sensor_base(name: str) -> str:
    for suffix in _STAT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """
    Default model inputs for a normal-behaviour model of ``target_col``: all numeric columns
    except the target, the other statistics of the target sensor (e.g. ``*_max``, ``*_min``,
    ``*_std`` for an ``*_avg`` target, which would leak the target) and metadata columns.

    :param df: Data containing the candidate features
    :param target_col: Name of the target column
    :return: List of feature column names
    """
    target_base = _sensor_base(target_col)
    return [col for col in df.select_dtypes(include=np.number).columns
            if col != target_col and _sensor_base(col) != target_base
            and col not in METADATA_COLUMNS]


class TimeSeriesModel(ABC):
    """
    Abstract base class for time series models.
    """

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_col: str):
        """
        Fit the model to the training data on the specified target column.
        
        :param self: The instance of the model
        :param train_df: Training data as a DataFrame
        :type train_df: pd.DataFrame
        :param target_col: Name of the target column in the DataFrame
        :type target_col: str
        """

    @abstractmethod
    def predict(self, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Predict using the model on the given DataFrame and target column.
        
        :param self: The instance of the model
        :param test_df: DataFrame containing the data for prediction
        :type test_df: pd.DataFrame
        :param target_col: Name of the target column in the DataFrame
        :type target_col: str
        :return: DataFrame with predictions for the target column
        :rtype: pd.DataFrame
        """

    @abstractmethod
    def reset(self):
        """
        Reset the model to its initial state.
        """


class ModelFactory:
    """
    Factory for creating time series models.
    """

    @staticmethod
    def create(model_name: str, **kwargs) -> TimeSeriesModel:
        """
        Creates a time series model instance based on the model_name.
        
        :param model_name: Name of the model to create
        :type model_name: str
        :param kwargs: Additional keyword arguments for the model constructor
        :return: An instance of a time series model
        :rtype: TimeSeriesModel
        """
        if model_name == "naive_forecaster":
            # Lazy import to avoid circular dependency with NaiveForecaster -> TimeSeriesModel
            from care_scada.modeling.naive_forecaster import NaiveForecaster
            return NaiveForecaster(**kwargs)
        
        if model_name == "constant_forecaster":
            from care_scada.modeling.naive_forecaster import ConstantForecaster
            return ConstantForecaster(**kwargs)
        
        if model_name == "random_forest":
            class RandomForestTimeSeriesModel(TimeSeriesModel):
                def __init__(self, **rf_kwargs):
                    self.rf_kwargs = {"n_estimators": 10, **rf_kwargs}
                    self.model = RandomForestRegressor(**self.rf_kwargs)
                    self.target_col = None
                    self.feature_cols = None

                def fit(self, train_df: pd.DataFrame, target_col: str):
                    self.target_col = target_col
                    self.feature_cols = feature_columns(train_df, target_col)
                    X = train_df[self.feature_cols].reset_index(drop=True)
                    y = train_df[target_col].reset_index(drop=True)
                    self.model.fit(X, y)

                def predict(self, test_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
                    X_test = test_df[self.feature_cols].reset_index(drop=True)
                    
                    preds = self.model.predict(X_test)
                    preds_df = pd.DataFrame(
                        preds,
                        index=test_df.index,
                        columns=[target_col]
                    )
                    return preds_df

                def reset(self):
                    self.model = RandomForestRegressor(**self.rf_kwargs)
    
            return RandomForestTimeSeriesModel(**kwargs)

        # Neural models live in their own module (JAX/Flax, optional dependencies)
        if model_name == "lstm":
            from care_scada.modeling.neural_forecasters import LSTMForecaster
            return LSTMForecaster(**kwargs)

        if model_name == "pcrnn":
            from care_scada.modeling.neural_forecasters import PCRNNForecaster
            return PCRNNForecaster(**kwargs)

        if model_name == "kalman":
            from care_scada.modeling.kalman_forecaster import KalmanThermalForecaster
            return KalmanThermalForecaster(**kwargs)

        raise ValueError(f"Unknown model_name '{model_name}'")
