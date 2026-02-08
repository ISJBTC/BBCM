"""
Blood-Based Cancer Mathematical Model

A comprehensive framework integrating mathematical modeling, machine learning,
and clinical decision support for precision oncology.
"""

__version__ = "1.0.0"
__author__ = "[Your Name]"
__email__ = "[your.email@domain.com]"

from .production_system import ClinicalCancerModel, BloodPanel
from .mathematical_validation import MathematicalAnalysis
from .biological_validation import BiologicalValidator
from .stability_simulation import ComprehensiveDocumentationGenerator

__all__ = [
    "ClinicalCancerModel",
    "BloodPanel",
    "MathematicalAnalysis",
    "BiologicalValidator",
    "ComprehensiveDocumentationGenerator",
]
