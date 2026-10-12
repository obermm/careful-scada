"""
SCADA data preprocessing pipeline.

Main API for preprocessing wind turbine SCADA data including data quality checks,
missing value handling, and outlier detection/handling.
"""

from .preprocessor import Preprocessor

__all__ = ["Preprocessor"]
