from typing import Optional

import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop: Optional[list[str]] = None, remove_outliers: bool = False) -> None:
        self.features_to_drop = features_to_drop
        self.remove_outliers = remove_outliers
        self.outlier_detector = IsolationForest(random_state=0)
        super().__init__()

    def fit(self, X: ndarray, y: Optional[ndarray] = None):
        """Fit function to conform to sklearn API.

        Args:
            X (ndarray): Array to use for fitting
            y (ndarray, optional): Target variable. Defaults to None.

        Returns:
            Self: Returns fit instance
        """
        if self.remove_outliers:
            self.outlier_detector.fit(self.clean_data(X))
        return self

    def fit_transform(self, X: ndarray, y: Optional[ndarray] = None, **fit_params) -> ndarray:
        """Fit and transform data. This function is used to conform to the sklearn API.

        Args:
            X (ndarray): Input array
            y (Optional[ndarray], optional): Target variable. Defaults to None.

        Returns:
            ndarray: Transformed input data
        """
        return super().fit_transform(X, y, **fit_params)

    def transform(self, X: ndarray, y: Optional[ndarray] = None) -> ndarray:
        """Transforms the data. This function is used to conform to the sklearn API.
        The data transformation includes data cleaning, and if enabled when instantiating
        the class, outlier removal.

        Args:
            X (ndarray): Input data.
            y (Optional[ndarray], optional): Target variable. Defaults to None.

        Returns:
            ndarray: Transformed input data
        """
        X_clean = self.clean_data(X)
        self._feature_names_out = list(X_clean.columns)
        self._n_features_out = len(self._feature_names_out)
        if self.remove_outliers:
            predictions = self.outlier_detector.predict(X)
            X_clean = X_clean.loc[predictions > 0]
        return X_clean

    def clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cleans the data, removing unnecessary features.

        Args:
            X (pd.DataFrame): Data for which to remove unnecessary features.

        Returns:
            pd.DataFrame: Clean dataframe.
        """
        if self.features_to_drop is not None:
            X = X.drop(columns=self.features_to_drop)
        # Extra data cleaning goes here
        return X

    def get_feature_names_out(self, input_features=None):
        return self._feature_names_out
