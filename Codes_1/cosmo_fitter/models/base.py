# models/base.py
"""Base classes for cosmological models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any, Union
import numpy as np

from config.manager import Config, get_config


@dataclass
class ModelInfo:
    """Information about a cosmological model."""
    name: str
    param_names: List[str]
    param_labels: List[str]
    param_bounds: List[Tuple[float, float]]
    n_params: int
    description: str = ""
    reference: str = ""
    lcdm_limit_params: Optional[Dict[str, float]] = None


class CosmologicalModel(ABC):
    """
    Abstract base class for all cosmological models.
    
    A model implements H(z) for given parameters and provides metadata
    for fitting and plotting.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._setup_matplotlib()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name (e.g., 'delta-LCDM', 'H_dot-alpha')."""
        pass
    
    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """Parameter names (e.g., ['H0', 'Om', 'delta'])."""
        pass
    
    @property
    def param_labels(self) -> List[str]:
        """LaTeX labels for parameters."""
        # Default implementation - can be overridden
        labels = {
            'H0': r'$H_0$ [km/s/Mpc]',
            'Om': r'$\Omega_{m,0}$',
            'delta': r'$\delta$',
            'alpha': r'$\alpha$',
            'b': r'$b$'
        }
        return [labels.get(name, name) for name in self.param_names]
    
    @property
    @abstractmethod
    def param_bounds(self) -> List[Tuple[float, float]]:
        """Parameter bounds for fitting."""
        pass
    
    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        return len(self.param_names)
    
    @property
    def model_info(self) -> ModelInfo:
        """Complete model information."""
        return ModelInfo(
            name=self.name,
            param_names=self.param_names,
            param_labels=self.param_labels,
            param_bounds=self.param_bounds,
            n_params=self.n_params,
            description=self._get_description(),
            reference=self._get_reference(),
            lcdm_limit_params=self._get_lcdm_limit_params()
        )
    
    def _get_description(self) -> str:
        """Get model description (override in subclasses)."""
        return ""
    
    def _get_reference(self) -> str:
        """Get model reference (override in subclasses)."""
        return ""
    
    def _get_lcdm_limit_params(self) -> Optional[Dict[str, float]]:
        """Get parameters for LCDM limit (override if applicable)."""
        return None
    
    def _setup_matplotlib(self):
        """Configure matplotlib for plotting (called once per model instance)."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib import rc
            if self.config.plotting.use_latex:
                rc('text', usetex=True)
                rc('font', family='serif')
        except Exception:
            # Fallback to mathtext
            from matplotlib import rc
            rc('text', usetex=False)
            rc('font', family='DejaVu Sans')
    
    @abstractmethod
    def H(self, z: Union[float, np.ndarray], params: np.ndarray) -> np.ndarray:
        """
        Compute H(z) for given parameters.
        
        Args:
            z: Redshift(s)
            params: Model parameters in order of param_names
        
        Returns:
            H(z) values
        """
        pass
    
    def chi2(self, params: np.ndarray, z: np.ndarray, H_obs: np.ndarray, 
             sigma: np.ndarray) -> float:
        """
        Compute chi-squared for given parameters and data.
        
        Args:
            params: Model parameters
            z: Redshift values
            H_obs: Observed H(z)
            sigma: Observation uncertainties
        
        Returns:
            Chi-squared value
        """
        if not self.within_bounds(params):
            return 1e12
        
        H_model = self.H(z, params)
        if np.any(~np.isfinite(H_model)) or np.any(H_model <= 0):
            return 1e12
        
        return float(np.sum(((H_obs - H_model) / sigma) ** 2))
    
    def within_bounds(self, params: np.ndarray) -> bool:
        """Check if parameters are within bounds."""
        for p, (lo, hi) in zip(params, self.param_bounds):
            if not (lo <= p <= hi):
                return False
        return True
    
    def lcdm_limit(self, params: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if parameters represent LCDM limit."""
        if self._get_lcdm_limit_params() is None:
            return False
        
        limit = self._get_lcdm_limit_params()
        for i, name in enumerate(self.param_names):
            if name in limit:
                if abs(params[i] - limit[name]) > tolerance:
                    return False
        return True
    
    def H_lcdm(self, z: np.ndarray, H0: float, Om: float) -> np.ndarray:
        """
        Standard flat LCDM Hubble parameter (for baseline comparisons).
        
        Args:
            z: Redshift(s)
            H0: Hubble constant
            Om: Matter density parameter
        
        Returns:
            H(z) for LCDM
        """
        return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))
    
    def chi2_lcdm(self, params: Tuple[float, float], z: np.ndarray,
                  H_obs: np.ndarray, sigma: np.ndarray) -> float:
        """Chi-squared for LCDM (for model comparison)."""
        H0, Om = params
        if H0 <= 0 or not (0 < Om < 1):
            return 1e12
        H_model = self.H_lcdm(z, H0, Om)
        return float(np.sum(((H_obs - H_model) / sigma) ** 2))
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"