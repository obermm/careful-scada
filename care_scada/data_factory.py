""""Dataset Factory for Turbine Level Datasets."""

from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class TurbineDataset:
    """
    Represents a dataset for a single turbine within a wind farm.
    Args:
        farm_name: Name of the wind farm ('A', 'B', or 'C')
        dataset_id: Unique identifier for the turbine dataset
        train_df: Training DataFrame for the turbine
        test_df: Testing DataFrame for the turbine
    """
    farm_name: str
    dataset_id: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame


class DatasetFactory:
    """ Factory for creating TurbineDataset instances from preporcessed DataFrames. """

    @staticmethod
    def create(
        farm_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> list[TurbineDataset]:
        """ Create TurbineDataset instances for each unique dataset_id.
        :param farm_name: Name of the wind farm ('A', 'B', or 'C').
        :type farm_name: str
        :param train_df: Preprocessed training DataFrame with MultiIndex (dataset_id, time_stamp).
        :type train_df: pd.DataFrame
        :param test_df: Preprocessed testing DataFrame with MultiIndex (dataset_id, time_stamp).
        :type test_df: pd.DataFrame
        :return: List of TurbineDataset instances, one per unique dataset_id.
        :rtype: list[TurbineDataset]
        """

        datasets = []

        dataset_ids = train_df.index.get_level_values(0).unique()

        for dataset_id in dataset_ids:
            ds = TurbineDataset(
                farm_name=str(farm_name),
                dataset_id=str(dataset_id),
                train_df=train_df.loc[dataset_id],
                test_df=test_df.loc[dataset_id],
            )
            datasets.append(ds)

        return datasets
