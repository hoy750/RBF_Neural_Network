"""
Visualization Module
====================

This module provides comprehensive visualization tools for RBF neural network
analysis, including training curves, prediction plots, residual analysis,
and performance comparison charts.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


class Visualizer:
    """
    Comprehensive visualization toolkit for RBF neural networks.

    Provides methods for creating publication-quality plots to analyze
    model performance, training dynamics, and prediction quality.

    Examples
    --------
    >>> import numpy as np
    >>> from src.rbf_nn.evaluation.visualization import Visualizer
    >>> viz = Visualizer()
    >>> loss_history = [1.0, 0.5, 0.25, 0.1]
    >>> viz.plot_loss_curve(loss_history)
    """

    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize Visualizer with plotting style.

        Parameters
        ----------
        style : str, default="seaborn"
            Matplotlib style to use for plots
        figsize : Tuple[int, int], default=(10, 6)
            Default figure size (width, height) in inches
        """
        self.style = style
        self.figsize = figsize

        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")
            print(f"Warning: Style '{style}' not found, using default")

    def plot_loss_curve(
        self,
        loss_history: List[float],
        title: str = "Loss Trend during Training",
        xlabel: str = "Iteration",
        ylabel: str = "Loss",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Plot training loss curve over epochs.

        Creates a line plot showing how loss decreases during training,
        useful for monitoring convergence and detecting overfitting.

        Parameters
        ----------
        loss_history : List[float]
            Loss values recorded at each epoch/iteration
        title : str, default="Loss Trend during Training"
            Plot title
        xlabel : str, default="Iteration"
            X-axis label
        ylabel : str, default="Loss"
            Y-axis label
        save_path : str, optional
            Path to save the plot image (e.g., 'loss_curve.png')
        show_plot : bool, default=True
            Whether to display the plot interactively

        Examples
        --------
        >>> viz = Visualizer()
        >>> losses = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15, 0.1]
        >>> viz.plot_loss_curve(losses, save_path="training_loss.png")
        """
        plt.figure(figsize=self.figsize)
        plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=2)

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_actual_vs_predicted(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Actual vs Predicted Values",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create scatter plot of actual vs predicted values.

        Ideal predictions fall on the diagonal y=x line.
        Deviations from this line indicate prediction errors.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values (1D array)
        y_pred : np.ndarray
            Predicted values (1D array)
        title : str, default="Actual vs Predicted Values"
            Plot title
        save_path : str, optional
            Path to save the plot
        show_plot : bool, default=True
            Whether to display the plot

        Examples
        --------
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 1.9, 2.8, 4.2, 4.9])
        >>> viz.plot_actual_vs_predicted(y_true, y_pred)
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        plt.figure(figsize=self.figsize)
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="black", s=60)

        # Plot perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

        plt.xlabel("Actual Values", fontsize=12)
        plt.ylabel("Predicted Values", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residual Plot",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create residual plot (predicted values vs residuals).

        Residuals should be randomly distributed around zero with
        no clear patterns for a well-fitted model.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values
        title : str, default="Residual Plot"
            Plot title
        save_path : str, optional
            Path to save the plot
        show_plot : bool, default=True
            Whether to display the plot

        Examples
        --------
        >>> viz.plot_residuals(y_true, y_pred, "Model Residuals")
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        residuals = y_true - y_pred

        plt.figure(figsize=self.figsize)
        plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="black", s=60)
        plt.axhline(y=0, color="r", linestyle="-", linewidth=2, label="Zero Line")

        plt.xlabel("Predicted Values", fontsize=12)
        plt.ylabel("Residuals", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        bins: int = 30,
        title: str = "Error Distribution",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create histogram of prediction errors.

        Errors should follow approximately normal distribution
        centered at zero for unbiased predictions.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values
        bins : int, default=30
            Number of histogram bins
        title : str, default="Error Distribution"
            Plot title
        save_path : str, optional
            Path to save the plot
        show_plot : bool, default=True
            Whether to display the plot
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        errors = y_true - y_pred

        plt.figure(figsize=self.figsize)
        n, bins_edges, patches = plt.hist(errors, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")

        # Add mean and std lines
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        plt.axvline(mean_error, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_error:.3f}")
        plt.axvline(mean_error + std_error, color="g", linestyle=":", linewidth=1.5, label=f"+1 Std: {std_error:.3f}")
        plt.axvline(mean_error - std_error, color="g", linestyle=":", linewidth=1.5, label=f"-1 Std")

        plt.xlabel("Error", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> float:
        """
        Plot Receiver Operating Characteristic (ROC) curve.

        Converts regression outputs to binary classification using
        median threshold for ROC computation.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values (continuous or binary)
        y_scores : np.ndarray
            Prediction scores/probabilities
        title : str, default="ROC Curve"
            Plot title
        save_path : str, optional
            Path to save the plot
        show_plot : bool, default=True
            Whether to display the plot

        Returns
        -------
        float
            AUC (Area Under Curve) score

        Examples
        --------
        >>> auc_score = viz.plot_roc_curve(y_true, y_pred_proba)
        >>> print(f"AUC Score: {auc_score:.3f}")
        """
        # Convert to binary if needed
        threshold = np.median(np.array(y_true))
        y_binary = (np.array(y_true) > threshold).astype(int)
        y_score = np.array(y_scores).flatten()

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.figure(figsize=self.figsize)
        plt.plot(
            fpr, tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

        return roc_auc

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> float:
        """
        Plot Precision-Recall curve.

        Useful for evaluating models on imbalanced datasets where
        ROC curves can be misleading.

        Parameters
        ----------
        y_true : np.ndarray
            Ground truth values
        y_scores : np.ndarray
            Prediction scores
        title : str, default="Precision-Recall Curve"
            Plot title
        save_path : str, optional
            Path to save the plot
        show_plot : bool, default=True
            Whether to display the plot

        Returns
        -------
        float
            Average precision score
        """
        # Convert to binary
        threshold = np.median(np.array(y_true))
        y_binary = (np.array(y_true) > threshold).astype(int)
        y_score = np.array(y_scores).flatten()

        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_binary, y_score)
        avg_precision = average_precision_score(y_binary, y_score)

        # Plot
        plt.figure(figsize=self.figsize)
        plt.plot(
            recall, precision,
            color="blue",
            lw=2,
            label=f"PR curve (AP = {avg_precision:.2f})"
        )

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall", fontsize=12)
        plt.ylabel("Precision", fontsize=12)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

        return avg_precision

    def plot_metrics_comparison(
        self,
        metrics_list: List[Dict[str, float]],
        metric_names: List[str],
        title: str = "Metrics Comparison Across Folds",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create grouped bar chart comparing metrics across folds/models.

        Parameters
        ----------
        metrics_list : List[Dict[str, float]]
            List of metric dictionaries
        metric_names : List[str]
            Names of metrics to compare
        title : str, default="Metrics Comparison Across Folds"
            Plot title
        save_path : str, optional
            Path to save the plot
        show_plot : bool, default=True
            Whether to display the plot

        Examples
        --------
        >>> metrics = [
        ...     {"RMSE": 0.15, "MAE": 0.12, "R²": 0.95},
        ...     {"RMSE": 0.18, "MAE": 0.14, "R²": 0.93}
        ... ]
        >>> viz.plot_metrics_comparison(metrics, ["RMSE", "MAE", "R²"])
        """
        n_folds = len(metrics_list)
        n_metrics = len(metric_names)

        # Convert to array format
        data = np.zeros((n_folds, n_metrics))
        for i, metrics in enumerate(metrics_list):
            for j, name in enumerate(metric_names):
                data[i, j] = metrics.get(name, 0)

        # Create bar chart
        x = np.arange(n_folds)
        width = 0.2

        fig, ax = plt.subplots(figsize=(max(10, n_folds * 1.5), 6))

        colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))
        for i in range(n_metrics):
            bars = ax.bar(x + i * width, data[:, i], width, label=metric_names[i], color=colors[i])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9
                )

        ax.set_xlabel("Fold / Model", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

    def plot_model_comparison_bar_chart(
        self,
        results: List[Dict[str, Union[str, float]]],
        metric_key: str = "f1",
        title: str = "Model Performance Comparison",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None:
        """
        Create horizontal bar chart comparing multiple models on a single metric.

        Parameters
        ----------
        results : List[Dict[str, Union[str, float]]]
            List of result dictionaries with 'model' name and metric values
        metric_key : str, default="f1"
            Key of the metric to visualize
        title : str, default="Model Performance Comparison"
            Plot title
        save_path : str, optional
            Path to save the plot
        show_plot : bool, default=True
            Whether to display the plot

        Examples
        --------
        >>> results = [
        ...     {"model": "RBF", "accuracy": 0.85, "f1": 0.83},
        ...     {"model": "RF", "accuracy": 0.88, "f1": 0.87},
        ...     {"model": "XGBoost", "accuracy": 0.90, "f1": 0.89}
        ... ]
        >>> viz.plot_model_comparison_bar_chart(results, "f1")
        """
        model_names = [r["model"] for r in results]
        scores = [r.get(metric_key, 0) for r in results]

        fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.2), 6))

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
        bars = ax.barh(model_names, scores, color=colors, edgecolor="black")

        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=11,
                fontweight="bold"
            )

        ax.set_xlabel(metric_key.upper(), fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        ax.set_xlim(0, max(scores) * 1.15)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.tight_layout()
            plt.show()

    @staticmethod
    def close_all_plots() -> None:
        """Close all open matplotlib figures to free memory."""
        plt.close("all")
