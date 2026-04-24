"""
RBF Neural Network Library
==========================

A comprehensive library for Radial Basis Function (RBF) neural networks,
including training, prediction, evaluation, and visualization capabilities.

Features:
- RBF neural network implementation with customizable parameters
- K-fold cross-validation support
- Multiple evaluation metrics (MSE, RMSE, MAE, R²)
- Rich visualization tools (loss curves, ROC, PR curves)
- Comparison with other ML models (Random Forest, XGBoost, SVM)

Author: RBF Research Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "RBF Research Team"

from src.rbf_nn.core.activations import tanh, de_tanh
from src.rbf_nn.core.rbf_network import RBFNeuralNetwork
from src.rbf_nn.data.preprocessing import DataPreprocessor
from src.rbf_nn.evaluation.metrics import MetricsCalculator
from src.rbf_nn.evaluation.visualization import Visualizer

__all__ = [
    "tanh",
    "de_tanh",
    "RBFNeuralNetwork",
    "DataPreprocessor",
    "MetricsCalculator",
    "Visualizer",
]
