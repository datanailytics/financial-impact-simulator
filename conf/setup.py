#!/usr/bin/env python
"""
Setup script for Financial Simulator package.

This file provides backward compatibility for installations that require setup.py,
though pyproject.toml is the preferred configuration method.
"""

import os
import sys
from pathlib import Path

from setuptools import find_packages, setup

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Read version from package
version_file = project_root / "financial_simulator" / "__init__.py"
version = None
if version_file.exists():
    with open(version_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

if not version:
    version = "1.0.0"

# Read README for long description
readme_file = project_root / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

# Core dependencies
install_requires = [
    # Core dependencies
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    "scipy>=1.10.0,<2.0.0",
    "scikit-learn>=1.3.0,<2.0.0",
    
    # Financial libraries
    "yfinance>=0.2.28",
    "pandas-datareader>=0.10.0",
    "pyportfolioopt>=1.5.5",
    "quantlib>=1.30",
    
    # Visualization
    "matplotlib>=3.7.0,<4.0.0",
    "seaborn>=0.12.0,<1.0.0",
    "plotly>=5.14.0,<6.0.0",
    
    # Web framework
    "fastapi>=0.100.0,<1.0.0",
    "uvicorn[standard]>=0.23.0,<1.0.0",
    "pydantic>=2.0.0,<3.0.0",
    "pydantic-settings>=2.0.0,<3.0.0",
    
    # Database
    "sqlalchemy>=2.0.0,<3.0.0",
    "alembic>=1.11.0,<2.0.0",
    "psycopg2-binary>=2.9.0,<3.0.0",
    "redis>=4.5.0,<5.0.0",
    
    # Task queue
    "celery>=5.3.0,<6.0.0",
    
    # Security
    "cryptography>=41.0.0",
    "python-jose[cryptography]>=3.3.0",
    "passlib[bcrypt]>=1.7.4",
    
    # HTTP clients
    "httpx>=0.24.0,<1.0.0",
    "requests>=2.31.0,<3.0.0",
    
    # Utilities
    "python-dotenv>=1.0.0",
    "click>=8.1.0,<9.0.0",
    "rich>=13.4.0,<14.0.0",
    "pytz>=2023.3",
    "python-dateutil>=2.8.0,<3.0.0",
    
    # Configuration
    "pyyaml>=6.0.0,<7.0.0",
    "toml>=0.10.2,<1.0.0",
]

# Development dependencies
dev_requires = [
    # Testing
    "pytest>=7.4.0,<8.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "pytest-asyncio>=0.21.0,<1.0.0",
    "pytest-mock>=3.11.0,<4.0.0",
    
    # Code quality
    "black>=23.7.0,<24.0.0",
    "isort>=5.12.0,<6.0.0",
    "flake8>=6.1.0,<7.0.0",
    "pylint>=2.17.0,<3.0.0",
    "mypy>=1.5.0,<2.0.0",
    "bandit>=1.7.5,<2.0.0",
    "safety>=2.3.0,<3.0.0",
    
    # Development tools
    "ipython>=8.14.0,<9.0.0",
    "jupyter>=1.0.0,<2.0.0",
    "pre-commit>=3.3.0,<4.0.0",
]

# Machine learning dependencies
ml_requires = [
    "tensorflow>=2.13.0,<3.0.0",
    "torch>=2.0.0,<3.0.0",
    "lightgbm>=4.0.0,<5.0.0",
    "xgboost>=1.7.0,<2.0.0",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=7.1.0,<8.0.0",
    "sphinx-rtd-theme>=1.3.0,<2.0.0",
    "myst-parser>=2.0.0,<3.0.0",
]

# Custom setup command to handle environment setup
from setuptools import Command


class InitializeProject(Command):
    """Custom command to initialize project structure and dependencies."""
    
    description = "Initialize project structure and install pre-commit hooks"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        """Run initialization tasks."""
        import subprocess
        
        # Create necessary directories
        directories = [
            "data/dev",
            "data/test",
            "data/prod",
            "models/dev",
            "models/test",
            "models/prod",
            "logs",
            "reports",
            "notebooks",
            "scripts",
            "tests/unit",
            "tests/integration",
            "tests/fixtures",
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Copy .env.example to .env if it doesn't exist
        env_example = Path(".env.example")
        env_file = Path(".env")
        if env_example.exists() and not env_file.exists():
            import shutil
            shutil.copy(env_example, env_file)
            print("Created .env file from .env.example")
        
        # Install pre-commit hooks
        try:
            subprocess.run(["pre-commit", "install"], check=True)
            print("Installed pre-commit hooks")
        except subprocess.CalledProcessError:
            print("Failed to install pre-commit hooks. Please run 'pre-commit install' manually.")
        
        # Initialize git if not already initialized
        if not Path(".git").exists():
            subprocess.run(["git", "init"], check=True)
            print("Initialized git repository")
        
        print("\nProject initialization complete!")
        print("Next steps:")
        print("1. Update .env with your configuration")
        print("2. Run 'pip install -e .[dev]' to install development dependencies")
        print("3. Run 'pre-commit run --all-files' to check code quality")


# Package discovery configuration
def find_package_data():
    """Find all non-Python files to include in the package."""
    package_data = {
        "financial_simulator": [
            "*.yaml",
            "*.yml",
            "*.json",
            "*.toml",
            "static/**/*",
            "templates/**/*",
            "config/*",
        ]
    }
    return package_data


# Main setup configuration
setup(
    # Basic information
    name="financial-simulator",
    version=version,
    author="Your Team Name",
    author_email="team@example.com",
    description="A comprehensive financial analysis and portfolio simulation platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/financial-simulator",
    license="MIT",
    
    # Package configuration
    packages=find_packages(
        exclude=["tests", "tests.*", "docs", "scripts", "notebooks"]
    ),
    package_data=find_package_data(),
    include_package_data=True,
    python_requires=">=3.11",
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "ml": ml_requires,
        "docs": docs_requires,
        "all": dev_requires + ml_requires + docs_requires,
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "financial-simulator=financial_simulator.cli:main",
            "fs-worker=financial_simulator.worker:start_worker",
            "fs-scheduler=financial_simulator.scheduler:start_scheduler",
            "fs-migrate=financial_simulator.database:run_migrations",
            "fs-init-db=financial_simulator.database:init_database",
        ],
    },
    
    # Custom commands
    cmdclass={
        "init": InitializeProject,
    },
    
    # PyPI classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Framework :: FastAPI",
        "Natural Language :: English",
    ],
    
    # Additional metadata
    project_urls={
        "Documentation": "https://financial-simulator.readthedocs.io",
        "Source": "https://github.com/yourorg/financial-simulator",
        "Issues": "https://github.com/yourorg/financial-simulator/issues",
        "Changelog": "https://github.com/yourorg/financial-simulator/blob/main/CHANGELOG.md",
    },
    
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
    ],
    
    # Build configuration
    zip_safe=False,
    platforms=["any"],
)

# Development installation instructions
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Financial Simulator Installation")
    print("="*60)
    print("\nFor development installation with all extras:")
    print("  pip install -e '.[all]'")
    print("\nFor production installation:")
    print("  pip install .")
    print("\nTo initialize the project structure:")
    print("  python setup.py init")
    print("\nFor more information, see README.md")
    print("="*60 + "\n")