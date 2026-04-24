"""Core module for RBF Neural Network"""

from src.rbf_nn.core.activations import tanh, de_tanh
from src.rbf_nn.core.rbf_network import RBFNeuralNetwork

__all__ = ["tanh", "de_tanh", "RBFNeuralNetwork"]
