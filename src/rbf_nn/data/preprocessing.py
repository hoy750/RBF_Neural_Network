"""
Data Preprocessing Module
=========================

This module provides utilities for data preprocessing, including
normalization, feature extraction, and train/test splitting for
RBF neural network applications.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """
    Data preprocessing pipeline for RBF neural networks.

    This class provides a comprehensive set of data preprocessing
    methods tailored for RBF neural network training and evaluation,
    including normalization, feature engineering, and data splitting.

    Attributes
    ----------
    scaler : StandardScaler or MinMaxScaler
        Fitted scaler object (set after fit_transform)
    threshold : float
        Threshold value for binary classification conversion
    feature_columns : List[str]
        Names of input feature columns
    target_column : str
        Name of target variable column

    Examples
    --------
    >>> import pandas as pd
    >>> from src.rbf_nn.data.preprocessing import DataPreprocessor
    >>> df = pd.read_csv("data.csv")
    >>> preprocessor = DataPreprocessor(
    ...     feature_columns=["Co", "Cr", "Mg", "Pb"],
    ...     target_column="Ti"
    ... )
    >>> X_scaled, y_binary = preprocessor.fit_transform(df)
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        normalization_method: str = "standard",
        classification_threshold: str = "median"
    ):
        """
        Initialize DataPreprocessor with configuration.

        Parameters
        ----------
        feature_columns : List[str], optional
            List of column names to use as input features.
            If None, will use default columns ["Co", "Cr", "Mg", "Pb"]
        target_column : str, optional
            Column name of the target variable.
            If None, will use default "Ti"
        normalization_method : str, default="standard"
            Normalization method: "standard" (z-score) or "minmax" (0-1 range)
        classification_threshold : str, default="median"
            Method to determine binary classification threshold:
            "median" uses median, "mean" uses mean value
        """
        self.feature_columns = feature_columns or ["Co", "Cr", "Mg", "Pb"]
        self.target_column = target_column or "Ti"
        self.normalization_method = normalization_method
        self.classification_threshold = classification_threshold

        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.threshold: Optional[float] = None

    def _get_scaler(self) -> Union[StandardScaler, MinMaxScaler]:
        """
        Get appropriate scaler based on normalization method.

        Returns
        -------
        Union[StandardScaler, MinMaxScaler]
            Scaler instance configured with chosen method
        """
        if self.normalization_method.lower() == "standard":
            return StandardScaler()
        elif self.normalization_method.lower() == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalization_method}. "
                f"Use 'standard' or 'minmax'"
            )

    def fit_transform(
        self,
        df: pd.DataFrame,
        return_binary_target: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessor to data and transform it.

        Extracts features, normalizes them, and optionally converts
        regression targets to binary classification labels.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing features and target
        return_binary_target : bool, default=True
            If True, convert target to binary using threshold;
            If False, return raw target values for regression

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X_scaled : Normalized feature matrix (n_samples, n_features)
            y_processed : Processed target vector (n_samples,)

        Raises
        ------
        ValueError
            If required columns are missing from DataFrame

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "Co": [1.0, 2.0, 3.0],
        ...     "Cr": [4.0, 5.0, 6.0],
        ...     "Mg": [7.0, 8.0, 9.0],
        ...     "Pb": [10.0, 11.0, 12.0],
        ...     "Ti": [0.5, 1.5, 2.5]
        ... })
        >>> preprocessor = DataPreprocessor()
        >>> X, y = preprocessor.fit_transform(df)
        >>> X.shape
        (3, 4)
        """
        # Validate required columns exist
        missing_features = [
            col for col in self.feature_columns if col not in df.columns
        ]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")

        if self.target_column not in df.columns:
            raise ValueError(f"Missing target column: {self.target_column}")

        # Extract features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values

        # Normalize features
        self.scaler = self._get_scaler()
        X_scaled = self.scaler.fit_transform(X)

        # Process target variable
        if return_binary_target:
            # Calculate threshold based on specified method
            if self.classification_threshold == "median":
                self.threshold = float(np.median(y))
            elif self.classification_threshold == "mean":
                self.threshold = float(np.mean(y))
            else:
                self.threshold = float(np.median(y))

            y_processed = (y > self.threshold).astype(int)
        else:
            y_processed = y

        return X_scaled, y_processed

    def transform(
        self,
        df: pd.DataFrame,
        return_binary_target: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform new data using fitted preprocessor.

        Uses previously fitted scaler to normalize features and
        apply same threshold for target processing.

        Parameters
        ----------
        df : pd.DataFrame
            New data to transform
        return_binary_target : bool, default=True
            Whether to return binary target labels

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Transformed features and processed targets

        Raises
        ------
        RuntimeError
            If transform is called before fit_transform
        """
        if self.scaler is None:
            raise RuntimeError(
                "Must call fit_transform() before transform(). "
                "Scaler has not been fitted yet."
            )

        X = df[self.feature_columns].values
        y = df[self.target_column].values

        X_scaled = self.scaler.transform(X)

        if return_binary_target and self.threshold is not None:
            y_processed = (y > self.threshold).astype(int)
        else:
            y_processed = y

        return X_scaled, y_processed

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and test sets.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        test_size : float, default=0.2
            Proportion of data to use for testing (0-1)
        random_state : int, default=42
            Random seed for reproducibility
        stratify : bool, default=False
            Whether to use stratified sampling (for classification)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            X_train, X_test, y_train, y_test split arrays
        """
        stratify_arr = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr
        )

        return X_train, X_test, y_train, y_test

    @staticmethod
    def add_noise(
        y: np.ndarray,
        noise_level: float = 0.03,
        noise_type: str = "uniform"
    ) -> np.ndarray:
        """
        Add controlled noise to target variable.

        Useful for creating robustness tests or simulating
        measurement uncertainty.

        Parameters
        ----------
        y : np.ndarray
            Target array to add noise to
        noise_level : float, default=0.03
            Magnitude of noise to add
        noise_type : str, default="uniform"
            Type of noise distribution: "uniform" or "gaussian"

        Returns
        -------
        np.ndarray
            Noisy version of input array
        """
        if noise_type == "uniform":
            noise = noise_level * np.random.rand(*y.shape)
        elif noise_type == "gaussian":
            noise = noise_level * np.random.randn(*y.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        return y + noise

    def get_feature_stats(self) -> dict:
        """
        Get statistics about the fitted features.

        Returns
        -------
        dict
            Dictionary containing feature statistics from scaler
        """
        if self.scaler is None:
            raise RuntimeError("Must call fit_transform() first")

        stats = {
            "feature_columns": self.feature_columns,
            "normalization_method": self.normalization_method,
        }

        if hasattr(self.scaler, "mean_"):
            stats["mean"] = self.scaler.mean_.tolist()
            stats["std"] = self.scaler.scale_.tolist()
        elif hasattr(self.scaler, "data_min_"):
            stats["min"] = self.scaler.data_min_.tolist()
            stats["max"] = self.scaler.data_max_.tolist()

        if self.threshold is not None:
            stats["classification_threshold"] = self.threshold

        return stats

    def __repr__(self) -> str:
        """Return string representation of the preprocessor."""
        return (
            f"DataPreprocessor("
            f"features={self.feature_columns}, "
            f"target={self.target_column}, "
            f"method={self.normalization_method})"
        )
