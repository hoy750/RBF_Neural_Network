"""
Activation Functions Module
==========================

This module provides activation functions used in RBF neural networks,
including hyperbolic tangent (tanh) and its derivative.
"""

import numpy as np
from typing import Union


def tanh(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Compute the hyperbolic tangent activation function.

    The tanh function squashes input values to the range [-1, 1],
    making it useful for neural network hidden layers.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Input value(s) to apply tanh activation to

    Returns
    -------
    Union[np.ndarray, float]
        Tanh activation output in range [-1, 1]

    Examples
    --------
    >>> import numpy as np
    >>> from src.rbf_nn.core.activations import tanh
    >>> tanh(0)
    0.0
    >>> tanh(1.0)
    0.7615941559557649

    Notes
    -----
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def de_tanh(x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Compute the derivative of the hyperbolic tangent function.

    This derivative is used during backpropagation to compute gradients.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Input value(s) (typically the output of tanh activation)

    Returns
    -------
    Union[np.ndarray, float]
        Derivative of tanh at the given point(s)

    Examples
    --------
    >>> import numpy as np
    >>> from src.rbf_nn.core.activations import de_tanh
    >>> de_tanh(0)
    1.0
    >>> de_tanh(0.5)
    0.75

    Notes
    -----
    Formula: d(tanh)/dx = 1 - tanh²(x)

    This is computed efficiently as: 1 - x² when x is already
    the output of a tanh activation.
    """
    return 1 - x ** 2
