"""
Metrics Calculation Module
==========================

This module provides comprehensive evaluation metrics for assessing
RBF neural network performance, supporting both regression and
classification tasks.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class MetricsCalculator:
    """
    Comprehensive metrics calculator for model evaluation.

    Provides methods for computing both classification and regression
    metrics, with support for multiple output formats and aggregation.

    Examples
    --------
    >>> import numpy as np
    >>> from src.rbf_nn.evaluation.metrics import MetricsCalculator
    >>> calculator = MetricsCalculator()
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0, 1, 0, 0, 1])
    >>> metrics = calculator.compute_classification_metrics(y_true, y_pred)
    >>> print(metrics['accuracy'])
    0.8
    """

    @staticmethod
    def compute_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Union[float, str]]:
        """
        Compute comprehensive classification metrics.

        Calculates precision, recall, F1 score, and accuracy for
        binary or multi-class classification problems.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels (1D array)
        y_pred : np.ndarray
            Predicted labels (1D array)
        model_name : str, default="Model"
            Name identifier for the model being evaluated

        Returns
        -------
        Dict[str, Union[float, str]]
            Dictionary containing:
            - 'model': Model name
            - 'accuracy': Classification accuracy
            - 'precision': Precision score
            - 'recall': Recall score
            - 'f1': F1 score

        Examples
        --------
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> metrics = MetricsCalculator.compute_classification_metrics(
        ...     y_true, y_pred, "TestModel"
        ... )
        >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        Accuracy: 80.00%
        """
        # Ensure 1D arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        return {
            "model": model_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Union[float, str]]:
        """
        Compute comprehensive regression metrics.

        Calculates MSE, RMSE, MAE, and R² score for regression tasks.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values (1D array)
        y_pred : np.ndarray
            Predicted values (1D array)
        model_name : str, default="Model"
            Name identifier for the model being evaluated

        Returns
        -------
        Dict[str, Union[float, str]]
            Dictionary containing:
            - 'model': Model name
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'r2': R-squared coefficient of determination

        Examples
        --------
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        >>> metrics = MetricsCalculator.compute_regression_metrics(
        ...     y_true, y_pred, "RegressionModel"
        ... )
        >>> print(f"R² Score: {metrics['r2']:.4f}")
        R² Score: 0.9867
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "model": model_name,
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }

    @staticmethod
    def print_classification_report(results: Dict[str, float]) -> None:
        """
        Print formatted classification results report.

        Parameters
        ----------
        results : Dict[str, float]
            Results dictionary from compute_classification_metrics
        """
        print(f"\n{results.get('model', 'Model')} 模型评估结果:")
        print(f"  Accuracy:   {results.get('accuracy', 0):.4f}")
        print(f"  Precision:  {results.get('precision', 0):.4f}")
        print(f"  Recall:     {results.get('recall', 0):.4f}")
        print(f"  F1 Score:   {results.get('f1', 0):.4f}")

    @staticmethod
    def print_regression_report(results: Dict[str, float]) -> None:
        """
        Print formatted regression results report.

        Parameters
        ----------
        results : Dict[str, float]
            Results dictionary from compute_regression_metrics
        """
        print(f"\n{results.get('model', 'Model')} 回归模型评估结果:")
        print(f"  MSE:  {results.get('mse', 0):.4f}")
        print(f"  RMSE: {results.get('rmse', 0):.4f}")
        print(f"  MAE:  {results.get('mae', 0):.4f}")
        print(f"  R²:   {results.get('r2', 0):.4f}")

    @classmethod
    def compare_models(
        cls,
        results_list: List[Dict[str, Union[float, str]]],
        task_type: str = "classification"
    ) -> None:
        """
        Print comparison table for multiple models.

        Parameters
        ----------
        results_list : List[Dict[str, float]]
            List of metric dictionaries from different models
        task_type : str, default="classification"
            Type of task: "classification" or "regression"

        Examples
        --------
        >>> results = [
        ...     {"model": "Model A", "accuracy": 0.85, "f1": 0.83},
        ...     {"model": "Model B", "accuracy": 0.90, "f1": 0.89}
        ... ]
        >>> MetricsCalculator.compare_models(results, "classification")
        """
        if not results_list:
            print("No results to compare.")
            return

        if task_type == "classification":
            header = f"{'模型':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}"
            separator = "-" * 52
            print("\n" + "=" * 60)
            print("模型分类性能对比:")
            print("=" * 60)
            print(header)
            print(separator)

            for result in results_list:
                print(
                    f"{result.get('model', ''):<12} "
                    f"{result.get('accuracy', 0):<10.4f} "
                    f"{result.get('precision', 0):<10.4f} "
                    f"{result.get('recall', 0):<10.4f} "
                    f"{result.get('f1', 0):<10.4f}"
                )

        elif task_type == "regression":
            header = f"{'模型':<12} {'MSE':<15} {'RMSE':<15} {'MAE':<15} {'R²':<15}"
            separator = "-" * 67
            print("\n" + "=" * 70)
            print("模型回归性能对比:")
            print("=" * 70)
            print(header)
            print(separator)

            for result in results_list:
                print(
                    f"{result.get('model', ''):<12} "
                    f"{result.get('mse', 0):<15.4f} "
                    f"{result.get('rmse', 0):<15.4f} "
                    f"{result.get('mae', 0):<15.4f} "
                    f"{result.get('r2', 0):<15.4f}"
                )

        else:
            raise ValueError(f"Unknown task type: {task_type}. Use 'classification' or 'regression'")

    @staticmethod
    def calculate_fold_average(
        all_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate average metrics across cross-validation folds.

        Parameters
        ----------
        all_metrics : List[Dict[str, float]]
            List of metric dictionaries from each fold

        Returns
        -------
        Dict[str, float]
            Average of each metric across all folds
        """
        if not all_metrics:
            return {}

        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key != "model":
                avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

        return avg_metrics
