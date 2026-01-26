"""Abstract Time Series Model and Model Factory."""

from abc import ABC, abstractmethod
import pandas as pd

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

        if model_name == "arima":
            raise NotImplementedError("ARIMAModel is not implemented yet.")
        if model_name == "lstm":
            raise NotImplementedError("LSTMModel is not implemented yet.")
        if model_name == "pcrnn":
            raise NotImplementedError("PCRNNModel is not implemented yet.")

        raise ValueError(f"Unknown model_name '{model_name}'")
