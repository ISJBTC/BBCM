#!/usr/bin/env python
"""
Setup script for blood-based cancer mathematical model.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements(filename="requirements.txt"):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="blood-cancer-model",
    version="1.0.0",
    author="[Your Name]",
    author_email="[your.email@domain.com]",
    description="Machine Learning Enhanced Blood-Based Cancer Mathematical Model for Precision Oncology",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/blood-cancer-model",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/blood-cancer-model/issues",
        "Documentation": "https://github.com/yourusername/blood-cancer-model/wiki",
        "Source Code": "https://github.com/yourusername/blood-cancer-model",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.12b0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "sphinx>=4.3.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "visualization": [
            "plotly>=5.3.0",
            "kaleido>=0.2.1",
        ],
        "web": [
            "flask>=2.0.0",
            "flask-cors>=3.0.10",
        ],
    },
    entry_points={
        "console_scripts": [
            "cancer-model-train=complete_ml_cancer_analysis:main",
            "cancer-model-stability=stability_simulation:main",
            "cancer-model-validate=mathematical_validation:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.csv", "*.json", "*.tex", "*.html"],
    },
    zip_safe=False,
    keywords=[
        "cancer modeling",
        "mathematical oncology",
        "machine learning",
        "precision medicine",
        "biomarkers",
        "dynamical systems",
        "stability analysis",
        "clinical decision support",
        "breast cancer",
        "treatment optimization",
    ],
)
