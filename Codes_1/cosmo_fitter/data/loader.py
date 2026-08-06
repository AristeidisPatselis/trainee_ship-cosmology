# data/loader.py
"""Unified data loading for cosmological datasets."""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from config.manager import get_config


@dataclass
class Dataset:
    """Container for cosmological dataset."""
    z: np.ndarray
    H: np.ndarray
    sigma: np.ndarray
    name: str = "cosmic_chronometers"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.metadata['n_points'] = len(self.z)
        self.metadata['z_range'] = (float(self.z.min()), float(self.z.max()))
    
    @property
    def n_points(self) -> int:
        return len(self.z)
    
    def validate(self) -> bool:
        """Validate dataset consistency."""
        if not (len(self.z) == len(self.H) == len(self.sigma)):
            raise ValueError(
                f"Mismatched lengths: z={len(self.z)}, H={len(self.H)}, sigma={len(self.sigma)}"
            )
        if np.any(self.z < 0):
            raise ValueError("Negative redshifts found")
        if np.any(self.H <= 0):
            raise ValueError("Non-positive H values found")
        if np.any(self.sigma <= 0):
            raise ValueError("Non-positive sigma values found")
        return True


def find_file_recursively(filename: str, base_dir: Path) -> Path:
    """Search for a file recursively in base_dir and its subdirectories."""
    # First check directly
    direct_path = base_dir / filename
    if direct_path.exists():
        return direct_path
    
    # Search recursively
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return Path(root) / filename
    
    # If not found, raise error with helpful message
    raise FileNotFoundError(
        f"Could not find '{filename}' in '{base_dir}' or its subdirectories.\n"
        f"Available files in {base_dir} and subdirectories:\n"
        f"{_list_available_files(base_dir)}"
    )


def _list_available_files(base_dir: Path, max_files: int = 20) -> str:
    """List available .txt files in base_dir and subdirectories."""
    files = []
    for root, dirs, filenames in os.walk(base_dir):
        for f in filenames:
            if f.endswith('.txt'):
                rel_path = os.path.relpath(os.path.join(root, f), base_dir)
                files.append(rel_path)
                if len(files) >= max_files:
                    files.append("... and more")
                    return "\n".join(files)
    return "\n".join(files) if files else "No .txt files found"


def load_clean_data(filename: str, base_dir: Path) -> np.ndarray:
    """Load numeric data from a file, handling formatting issues."""
    filepath = find_file_recursively(filename, base_dir)
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            # Handle possible bracket artifacts (e.g., "[12] 67.3")
            clean_line = line.split(']')[-1].strip()
            if clean_line:
                data.append(float(clean_line))
    
    return np.array(data)


def load_dataset(
    name: str = "cosmic_chronometers",
    config: Optional = None,
    z_file: Optional[str] = None,
    H_file: Optional[str] = None,
    sigma_file: Optional[str] = None,
    base_dir: Optional[Path] = None
) -> Dataset:
    """
    Load a cosmological dataset.
    
    Args:
        name: Dataset name
        config: Configuration object (uses global if None)
        z_file: Redshift file name (overrides config)
        H_file: H(z) file name (overrides config)
        sigma_file: Sigma file name (overrides config)
        base_dir: Base directory (overrides config)
    
    Returns:
        Dataset object containing z, H, sigma arrays
    """
    if config is None:
        config = get_config()
    
    # Use provided values or config defaults
    files = config.data.files
    z_file = z_file or files.get('z')
    H_file = H_file or files.get('H')
    sigma_file = sigma_file or files.get('sigma')
    base_dir = base_dir or config.data.base_dir
    
    if not all([z_file, H_file, sigma_file]):
        raise ValueError(
            "Missing file specifications. Provide z_file, H_file, sigma_file "
            "or ensure they are in the configuration."
        )
    
    print(f"Loading dataset '{name}' from: {base_dir}")
    print(f"  z: {z_file}")
    print(f"  H: {H_file}")
    print(f"  sigma: {sigma_file}")
    
    z = load_clean_data(z_file, base_dir)
    H = load_clean_data(H_file, base_dir)
    sigma = load_clean_data(sigma_file, base_dir)
    
    dataset = Dataset(z=z, H=H, sigma=sigma, name=name)
    dataset.validate()
    
    print(f"  Loaded {dataset.n_points} data points")
    print(f"  Redshift range: [{dataset.metadata['z_range'][0]:.3f}, {dataset.metadata['z_range'][1]:.3f}]")
    
    return dataset


# Dataset registry for predefined datasets
_DATASET_REGISTRY = {
    'cosmic_chronometers': {
        'z_file': 'c_z_vals.txt',
        'H_file': 'c_H_vals.txt',
        'sigma_file': 'c_sigma_vals.txt'
    },
    # Add more datasets here as you get them
}


def get_dataset(name: str, **kwargs) -> Dataset:
    """
    Get a dataset by name from the registry.
    
    Args:
        name: Dataset name (e.g., 'cosmic_chronometers')
        **kwargs: Additional arguments to pass to load_dataset
    
    Returns:
        Dataset object
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_DATASET_REGISTRY.keys())}")
    
    registry_entry = _DATASET_REGISTRY[name]
    # Merge registry defaults with kwargs
    for key, value in registry_entry.items():
        if key not in kwargs:
            kwargs[key] = value
    
    return load_dataset(name=name, **kwargs)


def list_datasets() -> list:
    """List available dataset names."""
    return list(_DATASET_REGISTRY.keys())


def register_dataset(name: str, **kwargs) -> None:
    """Register a new dataset."""
    _DATASET_REGISTRY[name] = kwargs