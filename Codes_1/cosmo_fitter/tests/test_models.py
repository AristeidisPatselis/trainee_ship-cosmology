# tests/test_models.py
"""Tests for cosmological models."""

import pytest
import numpy as np
from models.registry import get_model, list_models


def test_model_registry():
    """Test model registry."""
    models = list_models()
    assert 'lcdm' in models
    assert 'delta' in models
    
    # Get LCDM model
    lcdm = get_model('lcdm')
    assert lcdm.name == "LCDM"
    assert lcdm.param_names == ['H0', 'Om']
    assert lcdm.n_params == 2
    
    # Get delta model
    delta = get_model('delta')
    assert delta.name == "delta-LCDM"
    assert delta.param_names == ['H0', 'Om', 'delta']
    assert delta.n_params == 3


def test_lcdm_H():
    """Test LCDM H(z) calculation."""
    lcdm = get_model('lcdm')
    z = np.array([0.0, 0.5, 1.0])
    params = np.array([70.0, 0.3])
    
    H = lcdm.H(z, params)
    
    # H(0) should be H0
    assert np.isclose(H[0], 70.0)
    
    # Should be positive and increasing
    assert np.all(H > 0)
    assert np.all(np.diff(H) > 0)


def test_delta_lcdm_lcdm_limit():
    """Test that delta-LCDM reduces to LCDM at delta=0."""
    delta = get_model('delta')
    z = np.array([0.0, 0.5, 1.0])
    
    # LCDM parameters
    lcdm_params = np.array([70.0, 0.3])
    
    # Get LCDM from lcdm model
    lcdm = get_model('lcdm')
    H_lcdm = lcdm.H(z, lcdm_params)
    
    # Delta-LCDM with delta=0 should match
    delta_params = np.array([70.0, 0.3, 0.0])
    H_delta = delta.H(z, delta_params)
    
    np.testing.assert_allclose(H_delta, H_lcdm, rtol=1e-10)