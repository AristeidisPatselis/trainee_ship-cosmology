# models/registry.py
"""Registry for cosmological models."""

from typing import Dict, Type, Optional
from .base import CosmologicalModel
from .lcdm import LCDM
from .delta import DeltaLCDM
from .delta4 import Delta4LCDM  # Create this
from .hdot import HDotAlpha
from .delta_alpha import DeltaAlphaLCDM  # Create this
from .delta4_alpha import Delta4AlphaLCDM  # Create this


_MODEL_REGISTRY = {
    'lcdm': LCDM,
    'delta': DeltaLCDM,
    'delta4': Delta4LCDM,
    'hdot_alpha': HDotAlpha,
    'delta_alpha': DeltaAlphaLCDM,
    'delta4_alpha': Delta4AlphaLCDM,
}


def get_model_class(name: str) -> Type[CosmologicalModel]:
    """Get a model class by name."""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return _MODEL_REGISTRY[name]


def get_model(name: str, **kwargs) -> CosmologicalModel:
    """Create a model instance by name."""
    model_class = get_model_class(name)
    return model_class(**kwargs)


def list_models() -> list:
    """List available model names."""
    return list(_MODEL_REGISTRY.keys())


def register_model(name: str, model_class: Type[CosmologicalModel]) -> None:
    """Register a new model."""
    if name in _MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' already registered")
    _MODEL_REGISTRY[name] = model_class