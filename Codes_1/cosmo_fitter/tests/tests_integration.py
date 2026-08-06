# tests/test_integration.py
"""Integration tests for the full pipeline."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from models.registry import get_model
from data.loader import Dataset
from inference.optimizer import find_best_fit
from inference.mcmc import run_mcmc


@pytest.fixture
def mock_data():
    """Create mock data for testing."""
    z = np.linspace(0, 1, 10)
    # Mock LCDM with H0=70, Om=0.3
    H0, Om = 70.0, 0.3
    H = H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))
    sigma = 2.0 * np.ones_like(z)
    return Dataset(z=z, H=H, sigma=sigma, name='mock')


def test_lcdm_fit(mock_data):
    """Test fitting LCDM to mock data."""
    model = get_model('lcdm')
    z, H, sigma = mock_data.z, mock_data.H, mock_data.sigma
    
    best_params, chi2, _ = find_best_fit(
        model, z, H, sigma,
        n_starts=2, verbose=False
    )
    
    # Should recover true parameters
    assert np.isclose(best_params[0], 70.0, rtol=0.05)
    assert np.isclose(best_params[1], 0.3, rtol=0.05)


def test_mcmc_runs(mock_data):
    """Test that MCMC runs without errors."""
    model = get_model('lcdm')
    z, H, sigma = mock_data.z, mock_data.H, mock_data.sigma
    
    best_params = np.array([70.0, 0.3])
    
    sampler, flat_samples = run_mcmc(
        model, z, H, sigma, best_params,
        nwalkers=8, nsteps=100, discard=20, thin=2,
        progress=False
    )
    
    assert flat_samples.shape[0] > 0
    assert flat_samples.shape[1] == 2