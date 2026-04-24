"""
RBF Neural Network Library - Setup Configuration
=================================================

A comprehensive Python library for Radial Basis Function (RBF) neural networks
with training, evaluation, visualization, and model comparison capabilities.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rbf-neural-network",
    version="1.0.0",
    author="RBF Research Team",
    author_email="research@rbf-nn.com",
    description="A comprehensive RBF Neural Network library for research and production",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rbf-neural-network",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
        "ml-comparison": [
            "scikit-learn>=1.2.0",
            "xgboost>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rbf-train=src.rbf_nn.cli:train_model",
            "rbf-evaluate=src.rbf_nn.cli:evaluate_model",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="neural-network, rbf, machine-learning, deep-learning, classification, regression",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/rbf-neural-network/issues",
        "Documentation": "https://rbf-neural-network.readthedocs.io/",
        "Source Code": "https://github.com/yourusername/rbf-neural-network",
    },
)
