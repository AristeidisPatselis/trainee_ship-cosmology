# tests/test_config.py
"""Tests for the configuration system."""

import pytest
from pathlib import Path
from config.manager import Config, get_config


def test_config_loading():
    """Test loading configuration from YAML."""
    # Create a temporary config file
    import tempfile
    import yaml
    
    test_config = {
        'data': {
            'base_dir': '/test/data',
            'files': {'z': 'z.txt', 'H': 'H.txt'}
        },
        'models': {'h0_bounds': [50, 100], 'om_bounds': [0.01, 0.99]},
        'mcmc': {'nwalkers': 10, 'nsteps': 100}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        temp_path = Path(f.name)
    
    try:
        config = Config.from_yaml(temp_path)
        assert config.data.base_dir == Path('/test/data')
        assert config.models.h0_bounds == (50.0, 100.0)
        assert config.mcmc.nwalkers == 10
    finally:
        temp_path.unlink()


def test_get_config():
    """Test getting the global configuration."""
    from config.manager import reset_config
    
    reset_config()
    config = get_config()
    assert config is not None
    assert hasattr(config, 'data')
    assert hasattr(config, 'models')
    assert hasattr(config, 'mcmc')