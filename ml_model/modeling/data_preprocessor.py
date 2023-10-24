from typing import Optional

from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin


class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop: Optional[list[str]] = None) -> None:
        self.features_to_drop = features_to_drop
        super().__init__()

    def fit(self, X: ndarray, y: Optional[ndarray] = None):
        """Fit function to conform to sklearn API. Does nothing

        Args:
            X (ndarray): Array to use for fitting
            y (ndarray, optional): Target variable. Defaults to None.

        Returns:
            Self: Returns fit instance
        """
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

        Args:
            X (ndarray): Input data.
            y (Optional[ndarray], optional): Target variable. Defaults to None.

        Returns:
            ndarray: Transformed input data
        """
        X_clean = self.clean_data(X)
        return X_clean

    def clean_data(self, X: ndarray) -> ndarray:
        if self.features_to_drop is not None:
            X = X.drop(columns=self.features_to_drop)
        # Extra data cleaning goes here
        return X
