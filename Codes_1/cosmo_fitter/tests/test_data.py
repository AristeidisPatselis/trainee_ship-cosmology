# tests/test_data.py
"""Tests for data loading module."""

import pytest
from pathlib import Path
import tempfile
import numpy as np
from data.loader import Dataset, load_dataset, get_dataset, list_datasets


def test_dataset_validation():
    """Test dataset validation."""
    # Valid dataset
    dataset = Dataset(
        z=np.array([0.1, 0.2, 0.3]),
        H=np.array([70, 75, 80]),
        sigma=np.array([1, 2, 3])
    )
    assert dataset.validate() is True
    assert dataset.n_points == 3
    
    # Invalid: mismatched lengths
    with pytest.raises(ValueError):
        Dataset(
            z=np.array([0.1, 0.2]),
            H=np.array([70, 75, 80]),
            sigma=np.array([1, 2, 3])
        ).validate()
    
    # Invalid: negative redshift
    with pytest.raises(ValueError):
        Dataset(
            z=np.array([-0.1, 0.2, 0.3]),
            H=np.array([70, 75, 80]),
            sigma=np.array([1, 2, 3])
        ).validate()


def test_dataset_registry():
    """Test dataset registry."""
    datasets = list_datasets()
    assert 'cosmic_chronometers' in datasets
    
    # This will fail if actual data files aren't present, so we skip actual loading
    # in unit tests. Integration tests should handle this.