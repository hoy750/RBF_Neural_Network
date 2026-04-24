# RBF Neural Network Library 🧠


A comprehensive, production-ready Python library for **Radial Basis Function (RBF) Neural Networks** with state-of-the-art training algorithms, comprehensive evaluation metrics, rich visualization tools, and seamless integration with popular machine learning frameworks.

## ✨ Features

- 🎯 **Complete RBF Implementation**: Full forward/backward propagation with gradient descent optimization
- 📊 **Dual Task Support**: Both regression and classification tasks out-of-the-box
- 🔄 **K-Fold Cross Validation**: Robust model evaluation with cross-validation support
- 📈 **Rich Visualization**: Training curves, ROC/PR curves, residual plots, and more
- 🤖 **Model Comparison**: Built-in comparison with Random Forest, XGBoost, and SVM
- ⚡ **High Performance**: Optimized NumPy operations for efficient computation
- 🔧 **Fully Configurable**: Flexible hyperparameter tuning via configuration files
- 📝 **Well Documented**: Comprehensive docstrings and type hints throughout
- 🧪 **Test Coverage**: Unit tests ensuring code reliability

## 🚀 Quick Start


### Basic Usage

```python
import numpy as np
import pandas as pd
from src.rbf_nn.core.rbf_network import RBFNeuralNetwork
from src.rbf_nn.data.preprocessing import DataPreprocessor
from src.rbf_nn.evaluation.metrics import MetricsCalculator
from src.rbf_nn.evaluation.visualization import Visualizer

# Load your data
df = pd.read_csv("your_data.csv")

# Preprocess data
preprocessor = DataPreprocessor(
    feature_columns=["feature1", "feature2", "feature3", "feature4"],
    target_column="target"
)
X_scaled, y_processed = preprocessor.fit_transform(df)

# Split into train/test sets
X_train, X_test, y_train, y_test = preprocessor.split_data(
    X_scaled, y_processed, test_size=0.2
)

# Create and train RBF network
model = RBFNeuralNetwork(
    n_hidden_units=8,
    max_epochs=100,
    learning_rate=0.001
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
metrics = MetricsCalculator.compute_regression_metrics(
    y_test, predictions, "RBF Model"
)
MetricsCalculator.print_regression_report(metrics)

# Visualize results
viz = Visualizer()
viz.plot_loss_curve(model.loss_history_)
viz.plot_actual_vs_predicted(y_test, predictions)
```

## 📁 Project Structure

```
rbf-neural-network/
├── src/
│   └── rbf_nn/                    # Main package
│       ├── __init__.py            # Package initialization
│       ├── core/                  # Core neural network modules
│       │   ├── activations.py     # Activation functions (tanh, de_tanh)
│       │   └── rbf_network.py     # Main RBFNeuralNetwork class
│       ├── data/                  # Data processing utilities
│       │   └── preprocessing.py   # DataPreprocessor class
│       ├── models/                # ML model comparison
│       │   └── comparison.py      # ModelComparator class
│       ├── evaluation/            # Evaluation & visualization
│       │   ├── metrics.py         # MetricsCalculator class
│       │   └── visualization.py   # Visualizer class
│       └── utils/                 # Utility modules
│           └── config.py          # Configuration management
├── tests/                         # Unit tests
│   └── test_*.py
├── examples/                      # Example scripts
│   └── example.py                 # Complete usage example
├── docs/                          # Documentation
├── data/                          # Data files
├── setup.py                       # Package setup configuration
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
└── .gitignore                     # Git ignore rules
```

## 🎓 Core Concepts

### RBF Neural Network Architecture

The library implements a three-layer RBF neural network:

1. **Input Layer**: Receives feature vectors
2. **Hidden Layer (RBF Layer)**:
   - Uses K-means clustering to initialize radial basis function centers
   - Computes Gaussian activation based on distance from centers
   - Width parameters control the spread of each RBF neuron
3. **Output Layer**:
   - Linear combination of hidden layer outputs
   - Tanh activation for output transformation

### Training Algorithm

1. **Initialization**:
   - Use K-means clustering to determine RBF centers
   - Calculate width parameters using P-nearest-neighbor heuristic
   - Initialize output weights with Xavier/Glorot initialization

2. **Forward Propagation**:
   - Compute RBF activations in hidden layer
   - Apply tanh activation in output layer

3. **Backward Propagation**:
   - Compute gradients using chain rule
   - Update all parameters via gradient descent

4. **Convergence Check**:
   - Stop when loss < threshold or max epochs reached

## 📖 API Documentation

### `RBFNeuralNetwork` Class

Main class for creating and training RBF neural networks.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_hidden_units` | int | 8 | Number of RBF neurons in hidden layer |
| `max_epochs` | int | 100 | Maximum training iterations |
| `error_threshold` | float | 0.65e-3 | Convergence criterion |
| `learning_rate` | float | 0.001 | Gradient descent step size |
| `random_state` | int | 42 | Reproducibility seed |

#### Key Methods

##### `fit(X_train, y_train, verbose=True)`
Train the RBF network on provided data.

**Parameters:**
- `X_train`: Feature matrix (n_samples, n_features)
- `y_train`: Target values (n_samples,) or (n_samples, 1)
- `verbose`: Print training progress

**Returns:** self (for method chaining)

##### `predict(X_test)`
Generate predictions for new data.

**Parameters:**
- `X_test`: Test feature matrix (n_samples, n_features)

**Returns:** Predictions array (n_samples,)

##### `get_params()`
Retrieve all model parameters and configuration.

**Returns:** Dictionary with model state

### `DataPreprocessor` Class

Handles data normalization, feature extraction, and train/test splitting.

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `feature_columns` | list | ["Co","Cr","Mg","Pb"] | Input feature column names |
| `target_column` | str | "Ti" | Target variable name |
| `normalization_method` | str | "standard" | "standard" or "minmax" |
| `classification_threshold` | str | "median" | "median" or "mean" |

#### Key Methods

- `fit_transform(df)`: Fit preprocessor and transform data
- `transform(df)`: Transform new data using fitted parameters
- `split_data(X, y)`: Split into train/test sets
- `add_noise(y, level)`: Add controlled noise to targets

### `MetricsCalculator` Class

Comprehensive evaluation metrics for both classification and regression.

#### Static Methods

- `compute_classification_metrics(y_true, y_pred, name)` → Dict
- `compute_regression_metrics(y_true, y_pred, name)` → Dict
- `compare_models(results_list, task_type)`: Print comparison table
- `calculate_fold_average(all_metrics)` → Dict

### `Visualizer` Class

Publication-quality visualization tools.

#### Methods

- `plot_loss_curve(loss_history)`: Training loss over iterations
- `plot_actual_vs_predicted(y_true, y_pred)`: Scatter plot with diagonal
- `plot_residuals(y_true, y_pred)`: Residual analysis plot
- `plot_error_distribution(y_true, y_pred)`: Error histogram
- `plot_roc_curve(y_true, y_scores)`: ROC curve with AUC
- `plot_precision_recall_curve(y_true, y_scores)`: PR curve with AP
- `plot_model_comparison_bar_chart(results, metric)`: Horizontal bar chart

### `ModelComparator` Class

Compare RBF network against other ML algorithms.

#### Methods

- `add_model(name, model)`: Register a scikit-learn model
- `add_default_models()`: Add RF, XGBoost, SVM baselines
- `train_and_evaluate_classification(...)`: Run classification comparison
- `train_and_evaluate_regression(...)`: Run regression comparison
- `get_best_model(task_type, metric)`: Find top-performing model

### `Config` Dataclass

Centralized configuration management with JSON serialization.

#### Usage

```python
from src.rbf_nn.utils.config import Config

# Create default config
config = Config()

# Customize parameters
config.update(n_hidden_units=16, learning_rate=0.005)

# Save to file
config.save_config("my_experiment.json")

# Load from file
loaded_config = Config.load_config("my_experiment.json")
```

## 💡 Advanced Examples

### Example 1: K-Fold Cross Validation

```python
from sklearn.model_selection import KFold
from src.rbf_nn.core.rbf_network import RBFNeuralNetwork
from src.rbf_nn.evaluation.metrics import MetricsCalculator

kf = KFold(n_splits=10, shuffle=True, random_state=42)
all_metrics = []

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RBFNeuralNetwork(n_hidden_units=8)
    model.fit(X_train, y_train, verbose=False)

    predictions = model.predict(X_test)
    metrics = MetricsCalculator.compute_regression_metrics(
        y_test, predictions, f"Fold {fold_idx + 1}"
    )
    all_metrics.append(metrics)

avg_metrics = MetricsCalculator.calculate_fold_average(all_metrics)
print(f"Average RMSE: {avg_metrics['rmse']:.4f}")
print(f"Average R²: {avg_metrics['r2']:.4f}")
```

### Example 2: Model Comparison

```python
from src.rbf_nn.models.comparison import ModelComparator
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

comparator = ModelComparator()

# Add custom models
comparator.add_model("Random Forest", RandomForestClassifier(n_estimators=100))
comparator.add_model("XGBoost", XGBClassifier(n_estimators=100))
comparator.add_model("SVM", SVC(kernel='rbf'))

# Compare on classification task
classification_results = comparator.train_and_evaluate_classification(
    X_train, X_test, y_train, y_test
)

# Get best model
best = comparator.get_best_model("classification", "accuracy")
print(f"Best model: {best['model']} with accuracy {best['accuracy']:.4f}")
```

### Example 3: Complete Visualization Pipeline

```python
from src.rbf_nn.evaluation.visualization import Visualizer

viz = Visualizer(style="seaborn", figsize=(12, 8))

# Plot training dynamics
viz.plot_loss_curve(
    model.loss_history_,
    title="RBF Network Training Loss",
    save_path="output/training_curve.png"
)

# Analyze predictions
viz.plot_actual_vs_predicted(y_test, predictions, title="Prediction Quality")
viz.plot_residuals(y_test, predictions, title="Residual Analysis")
viz.plot_error_distribution(y_test, predictions, bins=40)

# Classification metrics (if applicable)
auc_score = viz.plot_roc_curve(y_test_binary, pred_scores, title="ROC Curve")
ap_score = viz.plot_precision_recall_curve(y_test_binary, pred_scores)

print(f"AUC: {auc_score:.4f}, Average Precision: {ap_score:.4f}")

# Clean up
Visualizer.close_all_plots()
```

### Example 4: Custom Configuration

```python
from src.rbf_nn.utils.config import Config, create_experiment_config

# High-capacity experiment
experiment_config = create_experiment_config(
    "high_capacity_rbf",
    n_hidden_units=32,
    max_epochs=200,
    learning_rate=0.005,
    error_threshold=0.5e-3
)
experiment_config.save_config("experiments/high_capacity.json")

# Quick training experiment
quick_config = create_experiment_config(
    "quick_test",
    max_epochs=50,
    n_hidden_units=4,
    verbose=False
)
```

## 🔧 Configuration Reference

All configurable parameters and their defaults:

### Network Architecture

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `n_hidden_units` | 8 | 2-64 | Number of RBF neurons |
| `max_epochs` | 100 | 10-10000 | Maximum iterations |
| `error_threshold` | 0.65e-3 | 1e-6 - 1e-1 | Stop loss threshold |
| `learning_rate` | 0.001 | 1e-5 - 1.0 | Gradient step size |

### Data Processing

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `test_size` | 0.2 | 0.1-0.5 | Test set proportion |
| `normalization_method` | "standard" | "standard"/"minmax" | Scaling method |
| `noise_level` | 0.03 | 0.0-1.0 | Augmentation noise |

### Cross-Validation

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `k_folds` | 10 | 3-20 | CV fold count |
| `random_state` | 42 | Any int | Reproducibility seed |

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src/rbf_nn --cov-report=html

# Run specific test file
python -m pytest tests/test_core.py -v

# Run tests matching pattern
python -m pytest tests/ -k "rbf" -v
```

Test coverage should be maintained above 80%.



### Code Style Guidelines

- Follow [PEP 8](https://pep8.org/) standards
- Use type hints for all function signatures
- Write docstrings following NumPy style
- Keep functions under 50 lines when possible
- Maintain test coverage > 80%

## 📚 References

- Broomhead, D. S., & Lowe, D. (1988). Radial basis functions, multi-variable functional interpolation and adaptive networks.
- Haykin, S. (1999). Neural Networks: A Comprehensive Approach.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn team for excellent ML infrastructure
- Matplotlib community for powerful visualization tools
- NumPy developers for high-performance computing
- Research community for RBF network theory

---

⭐ If this project helped you, please give it a star!
