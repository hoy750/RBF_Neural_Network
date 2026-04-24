"""
Unit Tests for RBF Neural Network Library
=========================================

Comprehensive test suite covering all major components:
- Activation functions
- RBF Neural Network core functionality
- Data preprocessing utilities
- Evaluation metrics calculation
- Configuration management

Run with: python -m pytest tests/ -v
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rbf_nn.core.activations import tanh, de_tanh
from src.rbf_nn.core.rbf_network import RBFNeuralNetwork
from src.rbf_nn.data.preprocessing import DataPreprocessor
from src.rbf_nn.evaluation.metrics import MetricsCalculator
from src.rbf_nn.evaluation.visualization import Visualizer
from src.rbf_nn.utils.config import Config


class TestActivationFunctions(unittest.TestCase):
    """Test cases for activation functions (tanh and de_tanh)."""

    def test_tanh_zero(self):
        """Test that tanh(0) = 0."""
        result = tanh(0.0)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_tanh_range(self):
        """Test that tanh output is in [-1, 1]."""
        x = np.array([-100, -10, -1, 0, 1, 10, 100])
        result = tanh(x)
        self.assertTrue(np.all(result >= -1))
        self.assertTrue(np.all(result <= 1))

    def test_tanh_symmetry(self):
        """Test that tanh is an odd function: tanh(-x) = -tanh(x)."""
        x = np.random.randn(100)
        self.assertTrue(np.allclose(tanh(-x), -tanh(x)))

    def test_de_tanh_at_zero(self):
        """Test that de_tanh(0) = 1 (derivative of tanh at zero)."""
        result = de_tanh(0.0)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_de_tanh_formula(self):
        """Test that de_tanh(x) = 1 - x^2."""
        x = np.random.uniform(-0.9, 0.9, 50)
        expected = 1 - x ** 2
        result = de_tanh(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_tanh_numpy_array(self):
        """Test that functions work correctly with numpy arrays."""
        x = np.linspace(-5, 5, 100)
        result = tanh(x)
        self.assertEqual(result.shape, x.shape)
        self.assertIsInstance(result, np.ndarray)


class TestRBFNeuralNetwork(unittest.TestCase):
    """Test cases for RBFNeuralNetwork class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate simple synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 4
        self.X_train = np.random.randn(n_samples, n_features)
        self.y_train = (
            np.sin(self.X_train[:, 0]) +
            0.5 * self.X_train[:, 1] ** 2 +
            0.1 * np.random.randn(n_samples)
        )
        self.X_test = np.random.randn(20, n_features)
        self.y_test = (
            np.sin(self.X_test[:, 0]) +
            0.5 * self.X_test[:, 1] ** 2 +
            0.1 * np.random.randn(20)
        )

    def test_initialization(self):
        """Test model initialization with default parameters."""
        model = RBFNeuralNetwork()
        self.assertEqual(model.n_hidden_units, 8)
        self.assertEqual(model.max_epochs, 100)
        self.assertAlmostEqual(model.learning_rate, 0.001)
        self.assertIsNone(model.w1_)

    def test_custom_initialization(self):
        """Test model initialization with custom parameters."""
        model = RBFNeuralNetwork(
            n_hidden_units=16,
            max_epochs=200,
            learning_rate=0.01,
            error_threshold=1e-4
        )
        self.assertEqual(model.n_hidden_units, 16)
        self.assertEqual(model.max_epochs, 200)
        self.assertAlmostEqual(model.learning_rate, 0.01)

    def test_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        model = RBFNeuralNetwork(max_epochs=5)
        result = model.fit(self.X_train, self.y_train, verbose=False)
        self.assertIs(result, model)

    def test_fit_sets_parameters(self):
        """Test that fit() initializes all network parameters."""
        model = RBFNeuralNetwork(max_epochs=5)
        model.fit(self.X_train, self.y_train, verbose=False)

        self.assertIsNotNone(model.w1_)
        self.assertIsNotNone(model.b1_)
        self.assertIsNotNone(model.w2_)
        self.assertIsNotNone(model.b2_)
        self.assertIsNotNone(model.loss_history_)
        self.assertTrue(len(model.loss_history_) > 0)

    def test_predict_shape(self):
        """Test that predict() returns correct shape."""
        model = RBFNeuralNetwork(max_epochs=5)
        model.fit(self.X_train, self.y_train, verbose=False)
        predictions = model.predict(self.X_test)

        self.assertEqual(predictions.shape[0], self.X_test.shape[0])

    def test_predict_before_fit_raises_error(self):
        """Test that predict() raises error if called before fit()."""
        model = RBFNeuralNetwork()
        with self.assertRaises(RuntimeError):
            model.predict(self.X_test)

    def test_incompatible_shapes_raises_error(self):
        """Test that fit() raises error for incompatible X/y shapes."""
        model = RBFNeuralNetwork()
        X_wrong = np.random.randn(50, 4)
        y_wrong = np.random.randn(30)  # Different number of samples

        with self.assertRaises(ValueError):
            model.fit(X_wrong, y_wrong)

    def test_loss_decreases_during_training(self):
        """Test that loss generally decreases during training."""
        model = RBFNeuralNetwork(max_epochs=20)
        model.fit(self.X_train, self.y_train, verbose=False)

        # Loss should be lower at end than at beginning
        self.assertLess(model.loss_history_[-1], model.loss_history_[0])

    def test_get_params(self):
        """Test get_params() returns complete parameter dictionary."""
        model = RBFNeuralNetwork(max_epochs=5)
        model.fit(self.X_train, self.y_train, verbose=False)
        params = model.get_params()

        self.assertIn("n_hidden_units", params)
        self.assertIn("w1", params)
        self.assertIn("loss_history", params)
        self.assertEqual(params["n_hidden_units"], 8)

    def test_repr(self):
        """Test string representation of model."""
        model = RBFNeuralNetwork(n_hidden_units=12, max_epochs=50)
        repr_str = repr(model)

        self.assertIn("RBFNeuralNetwork", repr_str)
        self.assertIn("12", repr_str)
        self.assertIn("50", repr_str)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""

    def setUp(self):
        """Set up test DataFrame."""
        np.random.seed(42)
        n_samples = 200
        self.df = pd.DataFrame({
            "Co": np.random.randn(n_samples),
            "Cr": np.random.randn(n_samples),
            "Mg": np.random.randn(n_samples),
            "Pb": np.random.randn(n_samples),
            "Ti": np.random.randn(n_samples)
        })

    def test_default_initialization(self):
        """Test default feature and target columns."""
        preprocessor = DataPreprocessor()
        self.assertEqual(preprocessor.feature_columns, ["Co", "Cr", "Mg", "Pb"])
        self.assertEqual(preprocessor.target_column, "Ti")

    def test_custom_columns(self):
        """Test custom feature and target column specification."""
        preprocessor = DataPreprocessor(
            feature_columns=["Co", "Cr"],
            target_column="Mg"
        )
        self.assertEqual(preprocessor.feature_columns, ["Co", "Cr"])
        self.assertEqual(preprocessor.target_column, "Mg")

    def test_fit_transform_output_shapes(self):
        """Test that fit_transform() returns correct shapes."""
        preprocessor = DataPreprocessor()
        X_scaled, y_processed = preprocessor.fit_transform(self.df)

        self.assertEqual(X_scaled.shape[0], len(self.df))
        self.assertEqual(X_scaled.shape[1], 4)
        self.assertEqual(y_processed.shape[0], len(self.df))

    def test_missing_columns_raises_error(self):
        """Test that missing columns raise ValueError."""
        df_incomplete = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        preprocessor = DataPreprocessor()

        with self.assertRaises(ValueError):
            preprocessor.fit_transform(df_incomplete)

    def test_binary_conversion(self):
        """Test binary target conversion using median threshold."""
        preprocessor = DataPreprocessor(classification_threshold="median")
        X_scaled, y_binary = preprocessor.fit_transform(self.df, return_binary_target=True)

        unique_values = set(y_binary)
        self.assertTrue(unique_values.issubset({0, 1}))

    def test_split_data_shapes(self):
        """Test train/test split produces correct shapes."""
        preprocessor = DataPreprocessor()
        X_scaled, y = preprocessor.fit_transform(self.df, return_binary_target=False)
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X_scaled, y, test_size=0.2
        )

        total_samples = len(self.df)
        expected_test_size = int(total_samples * 0.2)
        expected_train_size = total_samples - expected_test_size

        self.assertEqual(X_train.shape[0], expected_train_size)
        self.assertEqual(X_test.shape[0], expected_test_size)

    def test_add_noise_increases_variance(self):
        """Test that add_noise increases data variance."""
        original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy = DataPreprocessor.add_noise(original, noise_level=0.5)

        self.assertNotEqual(np.var(original), np.var(noisy))

    def test_get_feature_stats_after_fit(self):
        """Test get_feature_stats() after fitting."""
        preprocessor = DataPreprocessor()
        preprocessor.fit_transform(self.df)
        stats = preprocessor.get_feature_stats()

        self.assertIn("feature_columns", stats)
        self.assertIn("normalization_method", stats)


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for MetricsCalculator class."""

    def setUp(self):
        """Set up sample predictions and ground truth."""
        self.y_true_classification = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        self.y_pred_classification = np.array([0, 1, 0, 0, 1, 0, 1, 1])

        self.y_true_regression = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred_regression = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

    def test_classification_metrics_range(self):
        """Test that classification metrics are in valid ranges."""
        results = MetricsCalculator.compute_classification_metrics(
            self.y_true_classification,
            self.y_pred_classification,
            "TestModel"
        )

        self.assertGreaterEqual(results["accuracy"], 0)
        self.assertLessEqual(results["accuracy"], 1)
        self.assertGreaterEqual(results["precision"], 0)
        self.assertLessEqual(results["precision"], 1)
        self.assertGreaterEqual(results["recall"], 0)
        self.assertLessEqual(results["recall"], 1)
        self.assertGreaterEqual(results["f1"], 0)
        self.assertLessEqual(results["f1"], 1)

    def test_regression_metrics_positive_mse(self):
        """Test that MSE is always non-negative."""
        results = MetricsCalculator.compute_regression_metrics(
            self.y_true_regression,
            self.y_pred_regression,
            "TestModel"
        )

        self.assertGreaterEqual(results["mse"], 0)
        self.assertGreaterEqual(results["rmse"], 0)
        self.assertGreaterEqual(results["mae"], 0)

    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        perfect_y = np.array([1, 2, 3, 4, 5])
        results = MetricsCalculator.compute_regression_metrics(
            perfect_y,
            perfect_y,
            "PerfectModel"
        )

        self.assertAlmostEqual(results["mse"], 0.0, places=6)
        self.assertAlmostEqual(results["r2"], 1.0, places=6)

    def test_compare_models_prints_table(self):
        """Test that compare_models doesn't raise errors."""
        results_list = [
            {"model": "Model A", "accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1": 0.85},
            {"model": "Model B", "accuracy": 0.90, "precision": 0.89, "recall": 0.91, "f1": 0.90},
        ]

        # Should not raise any errors
        try:
            MetricsCalculator.compare_models(results_list, "classification")
        except Exception as e:
            self.fail(f"compare_models raised {type(e).__name__} unexpectedly!")

    def test_calculate_fold_average(self):
        """Test fold average calculation."""
        all_metrics = [
            {"mse": 0.1, "rmse": 0.316, "mae": 0.25, "r2": 0.95},
            {"mse": 0.15, "rmse": 0.387, "mae": 0.28, "r2": 0.93},
            {"mse": 0.12, "rmse": 0.346, "mae": 0.26, "r2": 0.94},
        ]

        avg = MetricsCalculator.calculate_fold_average(all_metrics)

        self.assertAlmostEqual(avg["mse"], 0.123333, places=4)
        self.assertAlmostEqual(avg["r2"], 0.94, places=4)


class TestVisualizer(unittest.TestCase):
    """Test cases for Visualizer class."""

    def setUp(self):
        """Set up visualizer instance."""
        self.viz = Visualizer(show_plot=False)
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 7.0, 7.8])
        self.loss_history = [1.0, 0.7, 0.5, 0.3, 0.2, 0.15]

    def test_visualizer_initialization(self):
        """Test Visualizer can be initialized."""
        viz = Visualizer()
        self.assertIsNotNone(viz)

    def test_plot_loss_curve_creates_figure(self):
        """Test loss curve plotting."""
        import matplotlib.pyplot as plt

        self.viz.plot_loss_curve(
            self.loss_history,
            show_plot=False
        )

        # Check figure was created
        figs = [plt.figure(num) for num in plt.get_fignums()]
        self.assertTrue(len(figs) > 0)
        plt.close("all")

    def test_plot_actual_vs_predicted(self):
        """Test actual vs predicted scatter plot."""
        import matplotlib.pyplot as plt

        self.viz.plot_actual_vs_predicted(
            self.y_true,
            self.y_pred,
            show_plot=False
        )

        figs = [plt.figure(num) for num in plt.get_fignums()]
        self.assertTrue(len(figs) > 0)
        plt.close("all")

    def test_close_all_plots(self):
        """Test close_all_plots() removes all figures."""
        import matplotlib.pyplot as plt

        # Create some figures
        plt.figure()
        plt.figure()
        self.assertEqual(len(plt.get_fignums()), 2)

        # Close them
        Visualizer.close_all_plots()
        self.assertEqual(len(plt.get_fignums()), 0)


class TestConfig(unittest.TestCase):
    """Test cases for Config dataclass."""

    def setUp(self):
        """Create temporary directory for config files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.test_dir)

    def test_default_config_values(self):
        """Test default configuration values."""
        config = Config()

        self.assertEqual(config.n_hidden_units, 8)
        self.assertEqual(config.max_epochs, 100)
        self.assertAlmostEqual(config.learning_rate, 0.001)
        self.assertEqual(config.test_size, 0.2)

    def test_to_dict_and_from_dict(self):
        """Test serialization to/from dictionary."""
        config = Config(n_hidden_units=16, learning_rate=0.01)
        config_dict = config.to_dict()

        restored_config = Config.from_dict(config_dict)

        self.assertEqual(restored_config.n_hidden_units, 16)
        self.assertAlmostEqual(restored_config.learning_rate, 0.01)

    def test_save_and_load_config(self):
        """Test saving and loading configuration file."""
        config_path = os.path.join(self.test_dir, "test_config.json")

        original_config = Config(
            n_hidden_units=32,
            max_epochs=200,
            learning_rate=0.005
        )
        original_config.save_config(config_path)

        loaded_config = Config.load_config(config_path)

        self.assertEqual(loaded_config.n_hidden_units, 32)
        self.assertEqual(loaded_config.max_epochs, 200)
        self.assertAlmostEqual(loaded_config.learning_rate, 0.005)

    def test_load_nonexistent_file_raises_error(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Config.load_config("nonexistent_config.json")

    def test_update_method(self):
        """Test updating specific configuration values."""
        config = Config()
        updated = config.update(n_hidden_units=20, learning_rate=0.02)

        self.assertEqual(updated.n_hidden_units, 20)
        self.assertAlmostEqual(updated.learning_rate, 0.02)
        # Other values should remain unchanged
        self.assertEqual(updated.max_epochs, 100)

    def test_update_invalid_key_raises_error(self):
        """Test updating with invalid key raises AttributeError."""
        config = Config()

        with self.assertRaises(AttributeError):
            config.update(invalid_parameter=42)

    def test_repr_formatting(self):
        """Test string representation format."""
        config = Config(n_hidden_units=12)
        repr_str = repr(config)

        self.assertIn("Config(", repr_str)
        self.assertIn("n_hidden_units=12", repr_str)


class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test complete training and evaluation pipeline."""
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 150
        df = pd.DataFrame({
            "Co": np.random.randn(n_samples),
            "Cr": np.random.randn(n_samples),
            "Mg": np.random.randn(n_samples),
            "Pb": np.random.randn(n_samples),
            "Ti": np.random.randn(n_samples)
        })

        # Preprocess
        preprocessor = DataPreprocessor()
        X_scaled, y = preprocessor.fit_transform(df, return_binary_target=False)
        X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)

        # Train model
        model = RBFNeuralNetwork(max_epochs=10)
        model.fit(X_train, y_train, verbose=False)

        # Predict
        predictions = model.predict(X_test)

        # Evaluate
        metrics = MetricsCalculator.compute_regression_metrics(
            y_test, predictions, "Integration Test"
        )

        # Verify we got valid metrics
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("r2", metrics)
        self.assertGreater(metrics["mse"], 0)

    def test_pipeline_with_binary_target(self):
        """Test pipeline with binary classification target."""
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            "Co": np.random.randn(n_samples),
            "Cr": np.random.randn(n_samples),
            "Mg": np.random.randn(n_samples),
            "Pb": np.random.randn(n_samples),
            "Ti": np.random.randn(n_samples)
        })

        preprocessor = DataPreprocessor()
        X_scaled, y_binary = preprocessor.fit_transform(df, return_binary_target=True)
        X_train, X_test, y_train, y_test = preprocessor.split_data(
            X_scaled, y_binary, stratify=True
        )

        model = RBFNeuralNetwork(max_epochs=5)
        model.fit(X_train, y_train.reshape(-1, 1), verbose=False)
        predictions = model.predict(X_test)

        # Convert to binary
        pred_binary = (predictions > 0).astype(int)

        metrics = MetricsCalculator.compute_classification_metrics(
            y_test, pred_binary, "Binary Classification"
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
