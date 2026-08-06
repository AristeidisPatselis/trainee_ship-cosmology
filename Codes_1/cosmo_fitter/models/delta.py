# models/delta.py
"""Delta-LCDM model with free delta."""

import numpy as np
from typing import Union, List, Tuple
from functools import lru_cache
from scipy.optimize import brentq
from .base import CosmologicalModel


class DeltaLCDM(CosmologicalModel):
    """
    Delta-LCDM model: H(z)^2 = H0^2 * [Om*(1+z)^3 + (1-Om)*(H/H0)^delta]
    Parameters: (H0, Om, delta)
    """
    
    @property
    def name(self) -> str:
        return "delta-LCDM"
    
    @property
    def param_names(self) -> List[str]:
        return ['H0', 'Om', 'delta']
    
    @property
    def param_bounds(self) -> List[Tuple[float, float]]:
        return [
            self.config.models.h0_bounds,
            self.config.models.om_bounds,
            (-3.0, 3.0)  # delta bounds
        ]
    
    def _get_description(self) -> str:
        return "Modified Friedmann equation with free delta: H^2 = H0^2[Om(1+z)^3 + (1-Om)(H/H0)^delta]"
    
    def _get_reference(self) -> str:
        return "Generalized dark energy parametrization"
    
    def _get_lcdm_limit_params(self):
        return {'delta': 0.0}
    
    def _H_single(self, z: float, H0: float, Om: float, delta: float) -> float:
        """Solve for H(z) at a single redshift."""
        if delta == 0.0:
            inside = Om * (1 + z) ** 3 + (1 - Om)
            return H0 * np.sqrt(inside) if inside > 0 else np.nan
        
        def eq(H):
            inside = Om * (1 + z) ** 3 + (1 - Om) * (H / H0) ** delta
            if inside <= 0 or not np.isfinite(inside):
                return 1e10
            return H - H0 * np.sqrt(inside)
        
        # Start bracket around LCDM value
        guess = H0 * np.sqrt(max(Om * (1 + z) ** 3 + (1 - Om), 1e-8))
        lo, hi = 0.05 * guess, 20.0 * guess
        
        try:
            flo, fhi = eq(lo), eq(hi)
            tries = 0
            while flo * fhi > 0 and tries < 40:
                lo *= 0.7
                hi *= 1.4
                flo, fhi = eq(lo), eq(hi)
                tries += 1
            if flo * fhi > 0:
                return np.nan
            return brentq(eq, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200)
        except Exception:
            return np.nan
    
    @lru_cache(maxsize=4096)
    def _H_single_cached(self, z: float, H0: float, Om: float, delta: float) -> float:
        return self._H_single(z, H0, Om, delta)
    
    def H(self, z: Union[float, np.ndarray], params: np.ndarray) -> np.ndarray:
        """
        Compute H(z) for delta-LCDM.
        
        Args:
            z: Redshift(s)
            params: [H0, Om, delta]
        """
        z = np.atleast_1d(z)
        H0, Om, delta = params
        
        if not (H0 > 0 and 0 < Om < 1):
            return np.full_like(z, np.nan)
        
        # Use cached version for speed
        out = [self._H_single_cached(float(zi), float(H0), float(Om), float(delta))
               for zi in z]
        return np.array(out)