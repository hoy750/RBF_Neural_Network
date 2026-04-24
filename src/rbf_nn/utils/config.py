"""
Configuration Management Module
===============================

This module provides centralized configuration management for RBF neural
network parameters, allowing easy customization and reproducibility.
"""

import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Config:
    """
    Configuration dataclass for RBF Neural Network parameters.

    This class encapsulates all configurable parameters for the RBF network,
    including architecture settings, training hyperparameters, and data
    processing options. Supports serialization to/from JSON files.

    Attributes
    ----------
    n_hidden_units : int
        Number of RBF neurons in hidden layer
    max_epochs : int
        Maximum number of training iterations
    error_threshold : float
        Convergence criterion for stopping training
    learning_rate : float
        Gradient descent step size
    random_state : int
        Seed for reproducible random initialization
    test_size : float
        Proportion of data held out for testing
    normalization_method : str
        Feature scaling method ("standard" or "minmax")
    k_folds : int
        Number of folds for cross-validation
    noise_level : float
        Noise magnitude for data augmentation
    verbose : bool
        Whether to print detailed training progress

    Examples
    --------
    >>> config = Config(n_hidden_units=16, learning_rate=0.01)
    >>> config.save_config("my_config.json")
    >>> loaded_config = Config.load_config("my_config.json")
    """

    # Network Architecture
    n_hidden_units: int = 8
    max_epochs: int = 100
    error_threshold: float = 0.65e-3
    learning_rate: float = 0.001
    random_state: int = 42

    # Data Splitting
    test_size: float = 0.2
    k_folds: int = 10

    # Data Processing
    normalization_method: str = "standard"
    noise_level: float = 0.03

    # Output Settings
    verbose: bool = True
    save_plots: bool = False
    output_dir: str = "./output"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of configuration
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config instance from dictionary.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration values

        Returns
        -------
        Config
            New Config instance with provided values
        """
        return cls(**config_dict)

    def save_config(self, filepath: str) -> None:
        """
        Save configuration to JSON file.

        Parameters
        ----------
        filepath : str
            Path to output JSON file

        Examples
        --------
        >>> config = Config()
        >>> config.save_config("config.json")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)

        print(f"Configuration saved to {filepath}")

    @classmethod
    def load_config(cls, filepath: str) -> "Config":
        """
        Load configuration from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON configuration file

        Returns
        -------
        Config
            Loaded configuration instance

        Raises
        ------
        FileNotFoundError
            If configuration file does not exist
        json.JSONDecodeError
            If file contains invalid JSON

        Examples
        --------
        >>> config = Config.load_config("config.json")
        >>> print(config.n_hidden_units)
        8
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def update(self, **kwargs: Any) -> "Config":
        """
        Update configuration with new values.

        Allows partial updates to specific parameters while keeping
        others unchanged.

        Parameters
        ----------
        **kwargs : Any
            Keyword arguments matching Config attributes

        Returns
        -------
        Config
            Updated configuration (self)

        Examples
        --------
        >>> config = Config()
        >>> config.update(n_hidden_units=16, learning_rate=0.01)
        >>> print(config.n_hidden_units)
        16
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Unknown config parameter: {key}")

        return self

    def __repr__(self) -> str:
        """Return formatted string representation."""
        lines = ["Config("]
        for key, value in self.to_dict().items():
            lines.append(f"    {key}={value!r},")
        lines.append(")")
        return "\n".join(lines)


def get_default_config() -> Config:
    """
    Get default configuration instance.

    Convenience function returning a Config with default values
    suitable for most use cases.

    Returns
    -------
    Config
        Default configuration instance
    """
    return Config()


def create_experiment_config(
    experiment_name: str,
    **overrides: Any
) -> Config:
    """
    Create a named experiment configuration.

    Useful for managing multiple experiments with different
    hyperparameter settings.

    Parameters
    ----------
    experiment_name : str
        Name identifier for the experiment
    **overrides : Any
        Configuration overrides for this experiment

    Returns
    -------
    Config
        Customized configuration for the experiment

    Examples
    --------
    >>> config = create_experiment_config(
    ...     "high_capacity",
    ...     n_hidden_units=32,
    ...     learning_rate=0.005,
    ...     max_epochs=200
    ... )
    """
    config = Config(**overrides)
    config.experiment_name = experiment_name
    return config
