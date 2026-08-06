# models/hdot.py
"""H_dot-alpha model: H(z)^2 = H0^2*Om*(1+z)^3 - alpha*(1+z)*H*dH/dz"""

import numpy as np
from typing import Union, List, Tuple
from functools import lru_cache
from scipy.integrate import solve_ivp
from .base import CosmologicalModel


class HDotAlpha(CosmologicalModel):
    """
    H_dot-alpha model with free alpha.
    Parameters: (H0, Om, alpha)
    """
    
    @property
    def name(self) -> str:
        return "H_dot-alpha"
    
    @property
    def param_names(self) -> List[str]:
        return ['H0', 'Om', 'alpha']
    
    @property
    def param_bounds(self) -> List[Tuple[float, float]]:
        return [
            self.config.models.h0_bounds,
            self.config.models.om_bounds,
            (0.01, 6.0)  # alpha bounds
        ]
    
    def _get_description(self) -> str:
        return ("Modified Friedmann equation: H^2 = H0^2*Om*(1+z)^3 "
                "- alpha*(1+z)*H*dH/dz")
    
    def _get_lcdm_limit_params(self):
        return {'alpha': 0.0}  # But note: alpha=0 is singular!
    
    def _rhs_u(self, z, u, H0, Om, alpha):
        """RHS for u=H^2 ODE."""
        u_safe = min(max(u[0], 1e-8), 1e10)
        x = 1.0 + z
        dudz = (2.0 / (alpha * x)) * (H0**2 * Om * x**3 - u_safe)
        return [np.clip(dudz, -1e12, 1e12)]
    
    @lru_cache(maxsize=256)
    def _H_cached(self, z_tuple, H0, Om, alpha):
        """Cached H(z) computation."""
        z_eval = np.array(z_tuple)
        if alpha <= 0 or H0 <= 0 or Om <= 0:
            return tuple([np.nan] * len(z_eval))
        
        z_max = max(z_eval.max(), 1e-6)
        t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))
        
        try:
            sol = solve_ivp(
                self._rhs_u, (0.0, z_max), [H0**2],
                args=(H0, Om, alpha),
                t_eval=t_eval,
                method='LSODA',
                rtol=1e-8, atol=1e-10,
                max_step=0.05,
            )
        except Exception:
            return tuple([np.nan] * len(z_eval))
        
        if not sol.success:
            return tuple([np.nan] * len(z_eval))
        
        u_of_z = np.interp(z_eval, sol.t, sol.y[0])
        if np.any(~np.isfinite(u_of_z)) or np.any(u_of_z <= 0):
            return tuple([np.nan] * len(z_eval))
        
        return tuple(np.sqrt(u_of_z))
    
    def H(self, z: Union[float, np.ndarray], params: np.ndarray) -> np.ndarray:
        z = np.atleast_1d(z)
        H0, Om, alpha = params
        
        if not (H0 > 0 and 0 < Om < 1):
            return np.full_like(z, np.nan)
        
        z_tuple = tuple(z)
        result = self._H_cached(z_tuple, float(H0), float(Om), float(alpha))
        return np.array(result)