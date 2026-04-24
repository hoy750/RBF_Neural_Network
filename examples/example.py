#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RBF Neural Network - Complete Usage Example
============================================

This script demonstrates all major features of the RBF Neural Network library,
including data preprocessing, model training, evaluation, visualization,
cross-validation, and model comparison.

Run this example to see the full workflow in action:
    python examples/example.py

Author: RBF Research Team
Version: 1.0.0
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for imports when running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rbf_nn.core.rbf_network import RBFNeuralNetwork
from src.rbf_nn.core.activations import tanh, de_tanh
from src.rbf_nn.data.preprocessing import DataPreprocessor
from src.rbf_nn.evaluation.metrics import MetricsCalculator
from src.rbf_nn.evaluation.visualization import Visualizer
from src.rbf_nn.models.comparison import ModelComparator
from src.rbf_nn.utils.config import Config


def generate_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 4,
    noise_level: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic dataset for demonstration purposes.

    Creates a dataset with non-linear relationships between features and target,
    suitable for testing RBF neural network capabilities.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples to generate
    n_features : int, default=4
        Number of input features
    noise_level : float, default=0.1
        Amount of Gaussian noise to add to target
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Generated dataset with feature columns and target column
    """
    np.random.seed(random_state)

    # Generate feature names (Co, Cr, Mg, Pb like in original research)
    feature_names = ["Co", "Cr", "Mg", "Pb"][:n_features]

    # Generate features with some correlations
    X = np.random.randn(n_samples, n_features)

    # Create non-linear target variable
    # Complex function that benefits from RBF's ability to capture local patterns
    y = (
        np.sin(X[:, 0]) * 2 +
        X[:, 1] ** 2 * 0.5 +
        np.exp(-X[:, 2]) * 1.5 +
        np.cos(X[:, 3]) * X[:, 0] +
        noise_level * np.random.randn(n_samples)
    )

    # Scale target to reasonable range
    y = (y - y.mean()) / y.std()

    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df["Ti"] = y

    print(f"  Generated synthetic dataset:")
    print(f"   Samples: {n_samples}")
    print(f"   Features: {n_features}")
    print(f"   Target range: [{y.min():.3f}, {y.max():.3f}]")

    return df


def example_1_basic_training():
    """
    Example 1: Basic RBF Network Training and Prediction
    =====================================================
    Demonstrates the simplest use case: load data, preprocess, train, predict.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic RBF Network Training")
    print("=" * 70)

    # Generate sample data
    df = generate_synthetic_data(n_samples=500)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        feature_columns=["Co", "Cr", "Mg", "Pb"],
        target_column="Ti",
        normalization_method="standard"
    )

    # Preprocess data
    X_scaled, y_processed = preprocessor.fit_transform(df, return_binary_target=False)
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X_scaled, y_processed, test_size=0.2
    )

    print(f"\n Data split complete:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")

    # Create and train RBF network
    model = RBFNeuralNetwork(
        n_hidden_units=8,
        max_epochs=50,
        learning_rate=0.001,
        error_threshold=1e-4
    )

    print(f"\n Training RBF Neural Network...")
    print(f"   Architecture: {model}")
    model.fit(X_train, y_train, verbose=True)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate performance
    metrics = MetricsCalculator.compute_regression_metrics(
        y_test, predictions, "RBF Network"
    )
    MetricsCalculator.print_regression_report(metrics)

    return model, metrics


def example_2_visualization():
    """
    Example 2: Comprehensive Visualization Suite
    ==============================================
    Demonstrates all available visualization tools.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Visualization Tools")
    print("=" * 70)

    # Generate data
    df = generate_synthetic_data(n_samples=300)
    preprocessor = DataPreprocessor()
    X_scaled, y = preprocessor.fit_transform(df, return_binary_target=False)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)

    # Train model
    model = RBFNeuralNetwork(max_epochs=30)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_test)

    # Initialize visualizer
    viz = Visualizer(style="seaborn", figsize=(12, 7))

    # Plot 1: Training loss curve
    print("\n Plotting training loss curve...")
    viz.plot_loss_curve(
        model.loss_history_,
        title="RBF Network Convergence"
    )

    # Plot 2: Actual vs Predicted
    print(" Plotting actual vs predicted values...")
    viz.plot_actual_vs_predicted(y_test, predictions)

    # Plot 3: Residual analysis
    print(" Analyzing residuals...")
    viz.plot_residuals(y_test, predictions)

    # Plot 4: Error distribution
    print(" Plotting error distribution...")
    viz.plot_error_distribution(y_test, predictions, bins=25)

    # For classification-style ROC/PR curves, convert to binary
    threshold = np.median(y_test)
    y_binary = (y_test > threshold).astype(int)
    pred_binary = (predictions > threshold).astype(int)

    # Plot 5: ROC curve
    print(" Computing ROC curve...")
    auc_score = viz.plot_roc_curve(y_test, predictions)
    print(f"   AUC Score: {auc_score:.4f}")

    # Plot 6: Precision-Recall curve
    print(" Computing PR curve...")
    ap_score = viz.plot_precision_recall_curve(y_test, predictions)
    print(f"   Average Precision: {ap_score:.4f}")

    # Clean up
    Visualizer.close_all_plots()

    return viz


def example_3_cross_validation():
    """
    Example 3: K-Fold Cross Validation
    ====================================
    Demonstrates robust model evaluation using K-fold CV.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: K-Fold Cross Validation")
    print("=" * 70)

    from sklearn.model_selection import KFold

    # Generate data
    df = generate_synthetic_data(n_samples=400)
    preprocessor = DataPreprocessor()
    X_scaled, y = preprocessor.fit_transform(df, return_binary_target=False)

    # Setup cross-validation
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    print(f"\n Running {k_folds}-fold cross-validation...")

    all_metrics = []
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train model for this fold
        model = RBFNeuralNetwork(
            n_hidden_units=6,
            max_epochs=40
        )
        model.fit(X_train, y_train, verbose=False)

        # Predict and evaluate
        predictions = model.predict(X_test)
        metrics = MetricsCalculator.compute_regression_metrics(
            y_test, predictions, f"Fold {fold_idx + 1}"
        )

        all_metrics.append(metrics)
        fold_results.append(metrics)

        print(f"\n   Fold {fold_idx + 1}/{k_folds}:")
        print(f"      MSE:  {metrics['mse']:.4f}")
        print(f"      RMSE: {metrics['rmse']:.4f}")
        print(f"      MAE:  {metrics['mae']:.4f}")
        print(f"      R²:   {metrics['r2']:.4f}")

    # Calculate average metrics across folds
    avg_metrics = MetricsCalculator.calculate_fold_average(all_metrics)

    print(f"\n{'='*60}")
    print(f"AVERAGE PERFORMANCE ACROSS {k_folds} FOLDS:")
    print(f"{'='*60}")
    print(f"   Mean MSE:  {avg_metrics['mse']:.4f} ± {np.std([m['mse'] for m in all_metrics]):.4f}")
    print(f"   Mean RMSE: {avg_metrics['rmse']:.4f} ± {np.std([m['rmse'] for m in all_metrics]):.4f}")
    print(f"   Mean MAE:  {avg_metrics['mae']:.4f} ± {np.std([m['mae'] for m in all_metrics]):.4f}")
    print(f"   Mean R²:   {avg_metrics['r2']:.4f} ± {np.std([m['r2'] for m in all_metrics]):.4f}")

    # Visualize fold comparison
    if len(fold_results) > 0:
        viz = Visualizer()
        metric_names = ["mse", "rmse", "mae", "r2"]
        viz.plot_metrics_comparison(
            fold_results,
            metric_names,
            title=f"Metrics Across {k_folds} Folds"
        )

    return avg_metrics, all_metrics


def example_4_model_comparison():
    """
    Example 4: Compare RBF with Other ML Models
    ============================================
    Benchmarks RBF against Random Forest, XGBoost, and SVM.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Model Comparison")
    print("=" * 70)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        import xgboost as xgb
    except ImportError:
        print("  Skipping model comparison: scikit-learn/xgboost not installed")
        print("   Install with: pip install scikit-learn xgboost")
        return None

    # Generate data
    df = generate_synthetic_data(n_samples=500)
    preprocessor = DataPreprocessor()
    X_scaled, y_binary = preprocessor.fit_transform(df, return_binary_target=True)
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        X_scaled, y_binary, stratify=True
    )

    # Train RBF model first
    print("\n Training RBF Neural Network...")
    rbf_model = RBFNeuralNetwork(max_epochs=40)
    rbf_model.fit(X_train, y_train.reshape(-1, 1), verbose=False)
    rbf_predictions = rbf_model.predict(X_test)
    rbf_pred_binary = (rbf_predictions > 0).astype(int)

    rbf_result = MetricsCalculator.compute_classification_metrics(
        y_test, rbf_pred_binary, "RBF NN"
    )
    MetricsCalculator.print_classification_report(rbf_result)

    # Compare with other models
    comparator = ModelComparator()
    comparator.add_default_models()

    print("\n🤖 Comparing with baseline models...")
    results = comparator.train_and_evaluate_classification(
        X_train, X_test, y_train, y_test, verbose=True
    )

    # Insert RBF result at beginning for comparison
    all_results = [rbf_result] + results

    # Print comprehensive comparison table
    MetricsCalculator.compare_models(all_results, "classification")

    # Visualize comparison
    viz = Visualizer(figsize=(10, 6))
    viz.plot_model_comparison_bar_chart(all_results, metric_key="accuracy")

    # Find best model
    best = max(all_results, key=lambda x: x["accuracy"])
    print(f"\n Best performing model: {best['model']} "
          f"(Accuracy: {best['accuracy']:.4f}, F1: {best['f1']:.4f})")

    return all_results


def example_5_configuration_management():
    """
    Example 5: Configuration Management
    ====================================
    Shows how to use Config class for experiment management.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Configuration Management")
    print("=" * 70)

    # Create default configuration
    config = Config()
    print("\n Default Configuration:")
    print(config)

    # Customize for specific experiment
    custom_config = Config(
        n_hidden_units=16,
        max_epochs=150,
        learning_rate=0.005,
        error_threshold=0.5e-3,
        k_folds=10,
        verbose=True
    )
    print("\n Custom Experiment Configuration:")
    print(custom_config)

    # Update specific parameters
    custom_config.update(noise_level=0.05, normalization_method="minmax")
    print(f"\n  After update - Noise level: {custom_config.noise_level}")

    # Save configuration
    config_path = "output/example_config.json"
    os.makedirs("output", exist_ok=True)
    custom_config.save_config(config_path)

    # Load configuration
    loaded_config = Config.load_config(config_path)
    print(f"\n Loaded configuration from file:")
    print(f"   Hidden units: {loaded_config.n_hidden_units}")
    print(f"   Learning rate: {loaded_config.learning_rate}")

    # Use configuration with model
    df = generate_synthetic_data(n_samples=200)
    preprocessor = DataPreprocessor(
        normalization_method=loaded_config.normalization_method
    )
    X_scaled, y = preprocessor.fit_transform(df, return_binary_target=False)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)

    model = RBFNeuralNetwork(
        n_hidden_units=loaded_config.n_hidden_units,
        max_epochs=loaded_config.max_epochs,
        learning_rate=loaded_config.learning_rate,
        error_threshold=loaded_config.error_threshold
    )

    print(f"\n Training with loaded configuration...")
    model.fit(X_train, y_train, verbose=True)

    return config, loaded_config


def example_6_activation_functions():
    """
    Example 6: Activation Functions Demonstration
    ===============================================
    Visualizes tanh activation and its derivative.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Activation Functions")
    print("=" * 70)

    import matplotlib.pyplot as plt

    # Generate input range
    x = np.linspace(-5, 5, 200)

    # Compute activations
    tanh_values = tanh(x)
    de_tanh_values = de_tanh(tanh_values)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, tanh_values, "b-", linewidth=2, label="tanh(x)")
    axes[0].set_title("Tanh Activation Function", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("x", fontsize=12)
    axes[0].set_ylabel("tanh(x)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    axes[0].axhline(y=-1, color="r", linestyle="--", alpha=0.5)
    axes[0].axhline(y=1, color="r", linestyle="--", alpha=0.5)
    axes[0].axvline(x=0, color="g", linestyle="--", alpha=0.5)

    axes[1].plot(x, de_tanh_values, "r-", linewidth=2, label="de_tanh(tanh(x))")
    axes[1].set_title("Derivative of Tanh", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("d(tanh)/dx", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)
    axes[1].axhline(y=0, color="k", linestyle="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("output/activation_functions.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nActivation function plots saved to output/activation_functions.png")

    # Print some example values
    print("\nSample values:")
    test_points = [-2, -1, 0, 1, 2]
    for point in test_points:
        t_val = tanh(point)
        dt_val = de_tanh(t_val)
        print(f"   tanh({point:>2}) = {t_val:>7.4f},  derivative = {dt_val:>7.4f}")


def main():
    """
    Main function running all examples sequentially.
    """
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  RBF NEURAL NETWORK LIBRARY - COMPLETE USAGE EXAMPLE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    print("\nThis script demonstrates all major features:")
    print("   1. Basic training and prediction")
    print("   2. Comprehensive visualization suite")
    print("   3. K-fold cross-validation")
    print("   4. Model comparison with ML baselines")
    print("   5. Configuration management")
    print("   6. Activation functions analysis")

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    try:
        # Run each example
        print("\n\nStarting Examples...\n")

        example_1_basic_training()
        example_2_visualization()
        example_3_cross_validation()
        example_4_model_comparison()
        example_5_configuration_management()
        example_6_activation_functions()

        print("\n\n" + "█" * 70)
        print(" ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("█" * 70)
        print("\n Output files saved to: ./output/")
        print("\n Next steps:")
        print("   • Check generated plots in output/ directory")
        print("   • Review printed metrics and comparisons")
        print("   • Modify parameters in examples for experimentation")
        print("   • Read README.md for detailed API documentation")

    except Exception as e:
        print(f"\n Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
