"""
RBF Neural Network Core Implementation
=======================================

This module provides the main RBFNeuralNetwork class that implements
a complete Radial Basis Function neural network with training and
prediction capabilities.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from sklearn.cluster import KMeans

from src.rbf_nn.core.activations import tanh, de_tanh


class RBFNeuralNetwork:
    """
    Radial Basis Function (RBF) Neural Network implementation.

    This class implements an RBF neural network with:
    - K-means clustering for initializing RBF centers
    - Gradient descent-based backpropagation for training
    - Support for both regression and classification tasks
    - Configurable network architecture and training parameters

    Attributes
    ----------
    n_hidden_units : int
        Number of hidden units (RBF neurons)
    max_epochs : int
        Maximum number of training epochs
    error_threshold : float
        Training stopping criterion based on loss
    learning_rate : float
        Learning rate for gradient descent
    random_state : int
        Random seed for reproducibility

    Examples
    --------
    >>> import numpy as np
    >>> from src.rbf_nn.core.rbf_network import RBFNeuralNetwork
    >>> # Create and train model
    >>> model = RBFNeuralNetwork(n_hidden_units=8, max_epochs=100)
    >>> X_train = np.random.randn(100, 4)
    >>> y_train = np.random.randn(100, 1)
    >>> model.fit(X_train, y_train)
    >>> # Make predictions
    >>> X_test = np.random.randn(20, 4)
    >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        n_hidden_units: int = 8,
        max_epochs: int = 100,
        error_threshold: float = 0.65e-3,
        learning_rate: float = 0.001,
        random_state: int = 42
    ):
        """
        Initialize RBF Neural Network with specified parameters.

        Parameters
        ----------
        n_hidden_units : int, default=8
            Number of RBF neurons in the hidden layer
        max_epochs : int, default=100
            Maximum number of training iterations
        error_threshold : float, default=0.65e-3
            Stop training if loss falls below this threshold
        learning_rate : float, default=0.001
            Step size for gradient descent updates
        random_state : int, default=42
            Random seed for reproducible results
        """
        self.n_hidden_units = n_hidden_units
        self.max_epochs = max_epochs
        self.error_threshold = error_threshold
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Model parameters (initialized during training)
        self.w1_: Optional[np.matrix] = None  # RBF center weights
        self.b1_: Optional[np.matrix] = None  # RBF width parameters
        self.w2_: Optional[np.matrix] = None  # Output layer weights
        self.b2_: Optional[np.matrix] = None  # Output layer bias
        self.loss_history_: List[float] = []  # Training loss history

        # Set random seed for reproducibility
        np.random.seed(self.random_state)

    def _initialize_centers(
        self,
        X_train: np.ndarray
    ) -> np.matrix:
        """
        Initialize RBF centers using K-means clustering.

        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (n_samples, n_features)

        Returns
        -------
        np.matrix
            Cluster centers as RBF neuron positions
        """
        kmeans = KMeans(
            n_clusters=self.n_hidden_units,
            max_iter=10000,
            random_state=self.random_state
        )
        kmeans.fit(X_train)
        return np.matrix(kmeans.cluster_centers_)

    def _initialize_widths(
        self,
        w1: np.matrix
    ) -> np.matrix:
        """
        Calculate RBF width parameters based on inter-center distances.

        Uses the P-nearest-neighbor heuristic where width is calculated
        based on maximum distance to other centers.

        Parameters
        ----------
        w1 : np.matrix
            RBF center matrix of shape (n_hidden_units, n_features)

        Returns
        -------
        np.matrix
            Width parameter matrix of shape (n_hidden_units, 1)
        """
        b1 = np.matrix(np.zeros((self.n_hidden_units, 1)))

        for i in range(self.n_hidden_units):
            max_dist = 0.0
            for j in range(self.n_hidden_units):
                dist = np.sqrt(np.sum(np.square(w1[i, :] - w1[j, :])))
                if dist > max_dist:
                    max_dist = dist
            b1[i] = max_dist / np.sqrt(2 * self.n_hidden_units)

        return b1

    def _initialize_output_layer(
        self,
        n_input_features: int,
        n_output_features: int = 1
    ) -> Tuple[np.matrix, np.matrix]:
        """
        Initialize output layer weights using Xavier/Glorot initialization.

        Parameters
        ----------
        n_input_features : int
            Number of input features to output layer (= n_hidden_units)
        n_output_features : int, default=1
            Number of output features

        Returns
        -------
        Tuple[np.matrix, np.matrix]
            Weight matrix w2 and bias vector b2
        """
        scale = np.sqrt(3.0 / ((n_input_features + n_output_features) * 0.5))

        w2 = np.random.uniform(
            low=-scale,
            high=scale,
            size=[n_input_features, n_output_features]
        )
        b2 = np.random.uniform(
            low=-scale,
            high=scale,
            size=[n_output_features, 1]
        )

        return np.matrix(w2), np.matrix(b2)

    def _forward_pass(
        self,
        X: np.matrix,
        w1: np.matrix,
        b1: np.matrix,
        w2: np.matrix,
        b2: np.matrix
    ) -> np.matrix:
        """
        Perform forward propagation through the network.

        Computes RBF activations in hidden layer and applies tanh
        activation in output layer.

        Parameters
        ----------
        X : np.matrix
            Input data matrix of shape (n_samples, n_features)
        w1, b1, w2, b2 : np.matrix
            Network parameters

        Returns
        -------
        np.matrix
            Network output predictions
        """
        n_samples = X.shape[0]
        hidden_output = np.matrix(np.zeros((n_samples, self.n_hidden_units)))

        # Compute RBF activations for each sample and each hidden unit
        for i in range(n_samples):
            for j in range(self.n_hidden_units):
                distance = (X[i, :] - w1[j, :]) * (X[i, :] - w1[j, :]).T
                width_term = 2 * b1[j, :] * b1[j, :]
                hidden_output[i, j] = np.exp((-1.0) * (distance / width_term))

        # Apply tanh activation in output layer
        output = tanh(hidden_output * w2 + b2)
        return output

    def _backward_pass(
        self,
        X: np.matrix,
        y_true: np.matrix,
        output: np.matrix,
        hidden_output: np.matrix,
        w1: np.matrix,
        b1: np.matrix,
        w2: np.matrix
    ) -> Tuple[np.matrix, np.matrix, np.matrix, np.matrix]:
        """
        Perform backward propagation to compute gradients.

        Implements gradient descent with chain rule to compute
        partial derivatives for all network parameters.

        Parameters
        ----------
        X : np.matrix
            Input data matrix
        y_true : np.matrix
            True target values
        output : np.matrix
            Predicted output from forward pass
        hidden_output : np.matrix
            Hidden layer RBF activations
        w1, b1, w2 : np.matrix
            Current network parameters

        Returns
        -------
        Tuple[np.matrix, np.matrix, np.matrix, np.matrix]
            Gradients for w1, b1, w2, b2 respectively
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Compute error
        error = y_true - output

        # Compute tanh derivative
        output_array = np.array(output.T)
        delta = de_tanh(output_array).transpose()

        # Initialize gradient accumulators
        grad_w1 = np.zeros((self.n_hidden_units, n_features))
        grad_b1 = np.zeros((self.n_hidden_units, 1))
        grad_w2 = np.zeros((self.n_hidden_units, 1))
        grad_b2 = np.zeros((1, 1))

        # Accumulate gradients across all samples
        for j in range(self.n_hidden_units):
            sum_w1 = 0.0
            sum_b1 = 0.0
            sum_w2 = 0.0
            sum_b2 = 0.0

            for i in range(n_samples):
                sum_w1 += (
                    error[i, :]
                    * delta[i, :]
                    * hidden_output[i, j]
                    * (X[i, :] - w1[j, :])
                )
                sum_b1 += (
                    error[i, :]
                    * delta[i, :]
                    * hidden_output[i, j]
                    * (X[i, :] - w1[j, :])
                    * (X[i, :] - w1[j, :]).T
                )
                sum_w2 += error[i, :] * delta[i, :] * hidden_output[i, j]
                sum_b2 += error[i, :] * delta[i, :]

            # Apply chain rule for RBF derivatives
            grad_w1[j, :] = (w2[j, :] / (b1[j, :] * b1[j, :])) * sum_w1
            grad_b1[j, :] = (
                w2[j, :] / (b1[j, :] * b1[j, :] * b1[j, :])
            ) * sum_b1
            grad_w2[j, :] = sum_w2
            grad_b2 = sum_b2

        return grad_w1, grad_b1, grad_w2, grad_b2

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True
    ) -> "RBFNeuralNetwork":
        """
        Train the RBF neural network on provided data.

        This method initializes network parameters and performs
        iterative gradient descent optimization until convergence
        or maximum epochs reached.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix of shape (n_samples, n_features)
        y_train : np.ndarray
            Training target values of shape (n_samples,) or (n_samples, 1)
        verbose : bool, default=True
            Whether to print training progress

        Returns
        -------
        self : RBFNeuralNetwork
            Returns self for method chaining

        Raises
        ------
        ValueError
            If X_train and y_train have incompatible shapes

        Examples
        --------
        >>> model = RBFNeuralNetwork(n_hidden_units=8)
        >>> X = np.random.randn(100, 4)
        >>> y = np.random.randn(100, 1)
        >>> model.fit(X, y, verbose=False)
        <src.rbf_nn.core.rbf_network.RBFNeuralNetwork object at ...>
        """
        # Validate input shapes
        X_train = np.array(X_train)
        y_train = np.array(y_train).reshape(-1, 1) if y_train.ndim == 1 else y_train

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples. "
                f"Got {X_train.shape[0]} and {y_train.shape[0]}"
            )

        n_samples, n_features = X_train.shape
        n_outputs = y_train.shape[1] if y_train.ndim > 1 else 1

        # Convert to matrices for computation
        X_matrix = np.matrix(X_train)
        y_matrix = np.matrix(y_train)

        # Initialize network parameters
        self.w1_ = self._initialize_centers(X_train)
        self.b1_ = self._initialize_widths(self.w1_)
        self.w2_, self.b2_ = self._initialize_output_layer(
            self.n_hidden_units, n_outputs
        )

        # Reset loss history
        self.loss_history_ = []

        # Training loop
        for epoch in range(self.max_epochs):
            # Forward pass
            hidden_output = np.matrix(np.zeros((n_samples, self.n_hidden_units)))
            for i in range(n_samples):
                for j in range(self.n_hidden_units):
                    distance = (
                        X_matrix[i, :] - self.w1_[j, :]
                    ) * (X_matrix[i, :] - self.w1_[j, :]).T
                    width_term = 2 * self.b1_[j, :] * self.b1_[j, :]
                    hidden_output[i, j] = np.exp((-1.0) * (distance / width_term))

            output = tanh(hidden_output * self.w2_ + self.b2_)

            # Compute loss
            error = y_matrix - output
            loss = float(np.sum(np.square(error)))
            self.loss_history_.append(loss)

            # Check convergence
            if loss < self.error_threshold:
                if verbose:
                    print(f"Converged at epoch {epoch + 1}, loss: {loss:.6f}")
                break

            # Backward pass and update weights
            grad_w1, grad_b1, grad_w2, grad_b2 = self._backward_pass(
                X_matrix, y_matrix, output, hidden_output,
                self.w1_, self.b1_, self.w2_
            )

            # Update parameters using gradient descent
            self.w1_ += self.learning_rate * grad_w1
            self.b1_ += self.learning_rate * grad_b1
            self.w2_ += self.learning_rate * grad_w2
            self.b2_ += self.learning_rate * grad_b2

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {loss:.6f}")

        if verbose:
            print(f"Training completed. Final loss: {self.loss_history_[-1]:.6f}")

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained RBF neural network.

        Parameters
        ----------
        X_test : np.ndarray
            Test feature matrix of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,)

        Raises
        ------
        RuntimeError
            If model has not been trained yet

        Examples
        --------
        >>> model = RBFNeuralNetwork()
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> predictions.shape
        (n_samples,)
        """
        if self.w1_ is None:
            raise RuntimeError("Model must be trained before making predictions. Call fit() first.")

        X_test = np.array(X_test)
        n_samples = X_test.shape[0]
        X_matrix = np.matrix(X_test)

        # Forward pass through trained network
        hidden_output = np.matrix(np.zeros((n_samples, self.n_hidden_units)))
        for i in range(n_samples):
            for j in range(self.n_hidden_units):
                distance = (X_matrix[i, :] - self.w1_[j, :]) * (X_matrix[i, :] - self.w1_[j, :]).T
                width_term = 2 * self.b1_[j, :] * self.b1_[j, :]
                hidden_output[i, j] = np.exp((-1.0) * (distance / width_term))

        output = tanh(hidden_output * self.w2_ + self.b2_)
        return np.array(output).flatten()

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters and configuration.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all model parameters and config
        """
        return {
            "n_hidden_units": self.n_hidden_units,
            "max_epochs": self.max_epochs,
            "error_threshold": self.error_threshold,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "w1": self.w1_,
            "b1": self.b1_,
            "w2": self.w2_,
            "b2": self.b2_,
            "loss_history": self.loss_history_
        }

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"RBFNeuralNetwork("
            f"n_hidden_units={self.n_hidden_units}, "
            f"max_epochs={self.max_epochs}, "
            f"learning_rate={self.learning_rate})"
        )
