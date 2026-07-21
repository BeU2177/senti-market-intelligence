"""Feature engineering package for Senti Market Intelligence."""

from features.feature_metadata import FeatureMetadata, FeatureCategory, FEATURE_REGISTRY
from features.target_builder import TargetBuilder
from features.leakage_validator import LeakageValidator, LeakageValidationReport
from features.feature_pipeline import FeaturePipeline, FeatureDataset, FEATURE_VERSION

__all__ = [
    "FeatureMetadata",
    "FeatureCategory",
    "FEATURE_REGISTRY",
    "TargetBuilder",
    "LeakageValidator",
    "LeakageValidationReport",
    "FeaturePipeline",
    "FeatureDataset",
    "FEATURE_VERSION",
]
