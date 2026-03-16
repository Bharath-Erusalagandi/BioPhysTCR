"""
GARSEF Feature Extraction Modules

This package provides feature extraction for:
- ESM2: Sequence embeddings (1280-dim)
- SaProt: Structure-aware embeddings (446-dim)
- Physicochemical: APBS, SASA, B-factors (8-dim per residue)
"""

from .esm2_extractor import (
    ESM2Extractor,
    load_esm2_embeddings,
)

from .saprot_extractor import (
    SaProtExtractor,
    FoldseekRunner,
    save_saprot_features,
    load_saprot_features,
)

from .physicochemical import (
    PhysicochemicalExtractor,
    FEATURE_NAMES as PHYSICOCHEMICAL_FEATURE_NAMES,
    normalize_features,
    load_physicochemical_features,
    save_physicochemical_features,
)

__all__ = [
    "ESM2Extractor",
    "load_esm2_embeddings",
    "SaProtExtractor",
    "FoldseekRunner",
    "save_saprot_features",
    "load_saprot_features",
    "PhysicochemicalExtractor",
    "PHYSICOCHEMICAL_FEATURE_NAMES",
    "normalize_features",
    "load_physicochemical_features",
    "save_physicochemical_features",
]
