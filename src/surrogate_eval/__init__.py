"""Evaluation and diagnostics for paired perturbation responses."""

from .metrics import compare, evaluate
from .schema import PairedReference, Prediction, load_prediction, load_reference

__version__ = "0.1.0"
__all__ = [
    "PairedReference",
    "Prediction",
    "load_reference",
    "load_prediction",
    "evaluate",
    "compare",
]
