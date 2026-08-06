# models/lcdm.py
"""Standard flat LambdaCDM model."""

import numpy as np
from typing import Union, List, Tuple
from .base import CosmologicalModel


class LCDM(CosmologicalModel):
    """Standard flat LCDM model with parameters (H0, Om)."""
    
    @property
    def name(self) -> str:
        return "LCDM"
    
    @property
    def param_names(self) -> List[str]:
        return ['H0', 'Om']
    
    @property
    def param_bounds(self) -> List[Tuple[float, float]]:
        return [
            self.config.models.h0_bounds,
            self.config.models.om_bounds
        ]
    
    def _get_description(self) -> str:
        return "Standard flat LambdaCDM model"
    
    def _get_reference(self) -> str:
        return "Standard cosmological model"
    
    def _get_lcdm_limit_params(self):
        # LCDM is the limit of itself
        return {'H0': None, 'Om': None}  # All values are the LCDM limit
    
    def H(self, z: Union[float, np.ndarray], params: np.ndarray) -> np.ndarray:
        """
        Compute H(z) for LCDM.
        
        Args:
            z: Redshift(s)
            params: [H0, Om]
        """
        z = np.atleast_1d(z)
        H0, Om = params
        return self.H_lcdm(z, H0, Om)