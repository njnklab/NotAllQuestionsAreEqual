"""Sanitized SNIE network architecture.

This package contains only architecture-level code. It intentionally excludes
data-source definitions, training scripts, private run artifacts, and fitted
parameters.
"""

from .conditioning import ConditionalMeanTable
from .data import SplitIndices, TensorBundle
from .encoders import FeatureTokenEncoder, TokenEncoder
from .experiments import ArchitectureVariant, ExperimentSpec, make_variant_specs, run_experiment_matrix
from .losses import SNIELossWeights
from .model import SNIEArchitectureConfig, SNIEModel
from .trainer import TrainingConfig, train_snie_model

__all__ = [
    "ArchitectureVariant",
    "ConditionalMeanTable",
    "ExperimentSpec",
    "FeatureTokenEncoder",
    "SNIELossWeights",
    "SNIEArchitectureConfig",
    "SNIEModel",
    "SplitIndices",
    "TensorBundle",
    "TokenEncoder",
    "TrainingConfig",
    "make_variant_specs",
    "run_experiment_matrix",
    "train_snie_model",
]
