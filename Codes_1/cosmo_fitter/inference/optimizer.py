# inference/optimizer.py
"""Optimization routines for cosmological models."""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from scipy.optimize import minimize, differential_evolution
from tqdm import tqdm

from models.base import CosmologicalModel
from config.manager import get_config


def find_best_fit(
    model: CosmologicalModel,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray,
    n_starts: Optional[int] = None,
    de_maxiter: Optional[int] = None,
    de_popsize: Optional[int] = None,
    nm_maxiter: Optional[int] = None,
    verbose: bool = True
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Find the best-fit parameters for a model.
    
    Uses differential evolution followed by Nelder-Mead refinement.
    
    Args:
        model: Cosmological model
        z: Redshift values
        H_obs: Observed H(z)
        sigma: Uncertainties
        n_starts: Number of multi-start runs
        de_maxiter: Max iterations for differential evolution
        de_popsize: Population size for differential evolution
        nm_maxiter: Max iterations for Nelder-Mead
        verbose: Print progress
    
    Returns:
        Tuple of (best_params, chi2, metadata)
    """
    config = get_config()
    opt_config = config.optimization
    
    n_starts = n_starts or opt_config.n_multistart
    de_maxiter = de_maxiter or opt_config.de_maxiter
    de_popsize = de_popsize or opt_config.de_popsize
    nm_maxiter = nm_maxiter or opt_config.nelder_mead_maxiter
    
    bounds = model.param_bounds
    param_names = model.param_names
    
    if verbose:
        print(f"  Running differential evolution...")
    
    def chi2_func(params):
        return model.chi2(params, z, H_obs, sigma)
    
    # Differential evolution
    de_result = differential_evolution(
        chi2_func,
        bounds=bounds,
        seed=42,
        maxiter=de_maxiter,
        tol=1e-8,
        polish=True,
        popsize=de_popsize,
    )
    
    best_params = de_result.x
    best_chi2 = de_result.fun
    
    if verbose:
        print(f"  Running {n_starts} multi-start local optimizations...")
    
    rng = np.random.default_rng(42)
    starts = [best_params] + [
        [rng.uniform(lo, hi) for (lo, hi) in bounds]
        for _ in range(n_starts)
    ]
    
    local_results = []
    for x0 in tqdm(starts, desc="  Local optimizations", disable=not verbose):
        res = minimize(
            chi2_func,
            x0,
            method='Nelder-Mead',
            bounds=bounds,
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': nm_maxiter}
        )
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2 = res.fun
            best_params = res.x
    
    if verbose:
        spreads = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        if spreads.size:
            print(f"  Multi-start scan: {spreads.size}/{len(starts)} runs converged")
            print(f"    chi2 range: [{spreads.min():.3f}, {spreads.max():.3f}]")
            if spreads.max() - spreads.min() > 0.5:
                print("    -> spread suggests degenerate/multi-modal surface")
    
    metadata = {
        'de_success': de_result.success,
        'n_converged': len([r for r in local_results if np.isfinite(r.fun)]),
        'n_starts': len(starts),
    }
    
    return best_params, best_chi2, metadata


def curve_fit_uncertainties(
    model: CosmologicalModel,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray,
    p0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate parameter uncertainties using curve_fit.
    
    Args:
        model: Cosmological model
        z: Redshift values
        H_obs: Observed H(z)
        sigma: Uncertainties
        p0: Initial parameter guess
    
    Returns:
        Tuple of (popt, perr, pcov)
    """
    from scipy.optimize import curve_fit
    
    def model_func(z_array, *params):
        H = model.H(z_array, np.array(params))
        if np.any(~np.isfinite(H)):
            return np.full_like(np.atleast_1d(z_array), 1e6, dtype=float)
        return H
    
    bounds = list(zip(*model.param_bounds))
    
    popt, pcov = curve_fit(
        model_func,
        z, H_obs,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000,
    )
    
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov