#!/usr/bin/env python3
"""
Setup script for BondX MLOps Module

This script handles the installation and setup of the MLOps module.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open(this_directory / "requirements.txt") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="bondx-mlops",
    version="1.0.0",
    author="BondX Development Team",
    author_email="dev@bondx.com",
    description="End-to-end MLOps capabilities for BondX including experiment tracking, model registry, drift detection, and canary deployments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bondx/bondx-mlops",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "mlflow": ["mlflow>=1.20.0"],
        "wandb": ["wandb>=0.12.0"],
        "viz": ["matplotlib>=3.5.0", "seaborn>=0.11.0", "plotly>=5.0.0"],
        "test": ["pytest>=6.2.0", "pytest-cov>=3.0.0", "pytest-mock>=3.6.0"],
        "dev": ["black>=22.0.0", "flake8>=4.0.0", "mypy>=0.950"],
        "docs": ["sphinx>=4.5.0", "sphinx-rtd-theme>=1.0.0"],
        "monitoring": ["prometheus-client>=0.12.0"],
        "security": ["cryptography>=3.4.0"],
        "database": ["sqlalchemy>=1.4.0", "alembic>=1.7.0"],
        "aws": ["boto3>=1.24.0"],
        "gcp": ["google-cloud-storage>=2.0.0"],
        "azure": ["azure-storage-blob>=12.0.0"],
        "queue": ["redis>=4.0.0", "celery>=5.2.0"],
        "web": ["fastapi>=0.78.0", "uvicorn>=0.17.0"],
    },
    entry_points={
        "console_scripts": [
            "bondx-mlops=bondx.mlops.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "bondx.mlops": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "mlops",
        "machine learning",
        "experiment tracking",
        "model registry",
        "drift detection",
        "canary deployment",
        "bond trading",
        "financial modeling",
        "risk management",
    ],
    project_urls={
        "Bug Reports": "https://github.com/bondx/bondx-mlops/issues",
        "Source": "https://github.com/bondx/bondx-mlops",
        "Documentation": "https://bondx-mlops.readthedocs.io/",
    },
    zip_safe=False,
)
