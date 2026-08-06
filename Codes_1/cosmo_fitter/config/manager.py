# config/manager.py
"""Configuration management for the cosmology fitting framework."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import yaml
import json


@dataclass
class DataConfig:
    """Data file configuration."""
    base_dir: Path
    files: Dict[str, str]
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataConfig':
        return cls(
            base_dir=Path(data.get('base_dir', '')),
            files=data.get('files', {})
        )
    
    def get_file_path(self, file_key: str) -> Path:
        """Get full path for a data file."""
        if file_key not in self.files:
            raise KeyError(f"Unknown data file key: {file_key}")
        return self.base_dir / self.files[file_key]


@dataclass
class MCMCConfig:
    """MCMC sampler configuration."""
    nwalkers: int = 32
    nsteps: int = 3000
    discard: int = 500
    thin: int = 15
    parallel: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MCMCConfig':
        return cls(**data)


@dataclass
class ContourConfig:
    """Contour plotting configuration."""
    grid_size: int = 60
    levels_2d: List[float] = field(default_factory=lambda: [2.30, 6.18, 11.83])
    levels_1d: List[float] = field(default_factory=lambda: [1.0, 4.0, 9.0])
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ContourConfig':
        return cls(**data)


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    n_multistart: int = 8
    de_maxiter: int = 200
    de_popsize: int = 20
    nelder_mead_maxiter: int = 5000
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationConfig':
        return cls(**data)


@dataclass
class PlottingConfig:
    """Plotting configuration."""
    dpi: int = 300
    figsize: List[int] = field(default_factory=lambda: [10, 8])
    use_latex: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlottingConfig':
        return cls(**data)


@dataclass
class ModelConfig:
    """Model-specific configuration."""
    h0_bounds: Tuple[float, float] = (40.0, 100.0)
    om_bounds: Tuple[float, float] = (0.01, 0.99)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        h0 = tuple(data.get('h0_bounds', [40.0, 100.0]))
        om = tuple(data.get('om_bounds', [0.01, 0.99]))
        return cls(h0_bounds=h0, om_bounds=om)


@dataclass
class Config:
    """Master configuration class."""
    data: DataConfig
    models: ModelConfig
    mcmc: MCMCConfig
    contours: ContourConfig
    optimization: OptimizationConfig
    plotting: PlottingConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            data=DataConfig.from_dict(data.get('data', {})),
            models=ModelConfig.from_dict(data.get('models', {})),
            mcmc=MCMCConfig.from_dict(data.get('mcmc', {})),
            contours=ContourConfig.from_dict(data.get('contours', {})),
            optimization=OptimizationConfig.from_dict(data.get('optimization', {})),
            plotting=PlottingConfig.from_dict(data.get('plotting', {}))
        )
    
    @classmethod
    def from_json(cls, json_path: Path) -> 'Config':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Similar to from_yaml but for JSON
        return cls.from_yaml(json_path)  # For now, just use YAML
    
    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            'data': {
                'base_dir': str(self.data.base_dir),
                'files': self.data.files
            },
            'models': {
                'h0_bounds': list(self.models.h0_bounds),
                'om_bounds': list(self.models.om_bounds)
            },
            'mcmc': {
                'nwalkers': self.mcmc.nwalkers,
                'nsteps': self.mcmc.nsteps,
                'discard': self.mcmc.discard,
                'thin': self.mcmc.thin,
                'parallel': self.mcmc.parallel
            },
            'contours': {
                'grid_size': self.contours.grid_size,
                'levels_2d': self.contours.levels_2d,
                'levels_1d': self.contours.levels_1d
            },
            'optimization': {
                'n_multistart': self.optimization.n_multistart,
                'de_maxiter': self.optimization.de_maxiter,
                'de_popsize': self.optimization.de_popsize,
                'nelder_mead_maxiter': self.optimization.nelder_mead_maxiter
            },
            'plotting': {
                'dpi': self.plotting.dpi,
                'figsize': self.plotting.figsize,
                'use_latex': self.plotting.use_latex
            }
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_output_dir(self, model_name: str, base_output_dir: Path) -> Path:
        """Get output directory for a specific model."""
        output_dir = base_output_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


# Global configuration instance - load once and reuse
_global_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None) -> Config:
    """Get the global configuration instance."""
    global _global_config
    
    if _global_config is None:
        if config_path is None:
            # Try to find default config
            default_path = Path(__file__).parent / 'defaults.yaml'
            if default_path.exists():
                config_path = default_path
            else:
                raise FileNotFoundError(
                    "No configuration file found. Please provide config_path."
                )
        _global_config = Config.from_yaml(config_path)
    
    return _global_config


def reset_config() -> None:
    """Reset the global configuration."""
    global _global_config
    _global_config = None