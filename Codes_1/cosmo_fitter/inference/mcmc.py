# inference/mcmc.py
"""MCMC sampling routines for cosmological models."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import emcee
from tqdm import tqdm

from models.base import CosmologicalModel
from config.manager import get_config


def log_prior(params: np.ndarray, model: CosmologicalModel) -> float:
    """Log-prior for parameters."""
    if not model.within_bounds(params):
        return -np.inf
    return 0.0


def log_likelihood(
    params: np.ndarray,
    model: CosmologicalModel,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray
) -> float:
    """Log-likelihood function."""
    chi2_val = model.chi2(params, z, H_obs, sigma)
    if chi2_val >= 1e11:
        return -np.inf
    return -0.5 * chi2_val


def log_probability(
    params: np.ndarray,
    model: CosmologicalModel,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray
) -> float:
    """Log-probability (prior + likelihood)."""
    lp = log_prior(params, model)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, model, z, H_obs, sigma)


def run_mcmc(
    model: CosmologicalModel,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray,
    start_params: np.ndarray,
    nwalkers: Optional[int] = None,
    nsteps: Optional[int] = None,
    discard: Optional[int] = None,
    thin: Optional[int] = None,
    parallel: Optional[bool] = None,
    progress: bool = True
) -> Tuple[emcee.EnsembleSampler, np.ndarray]:
    """
    Run MCMC sampling for a model.
    
    Args:
        model: Cosmological model
        z: Redshift values
        H_obs: Observed H(z)
        sigma: Uncertainties
        start_params: Starting parameters
        nwalkers: Number of walkers
        nsteps: Number of steps
        discard: Number of burn-in steps to discard
        thin: Thinning factor
        parallel: Use parallel processing
        progress: Show progress bar
    
    Returns:
        Tuple of (sampler, flat_samples)
    """
    config = get_config()
    mcmc_config = config.mcmc
    
    nwalkers = nwalkers or mcmc_config.nwalkers
    nsteps = nsteps or mcmc_config.nsteps
    discard = discard or mcmc_config.discard
    thin = thin or mcmc_config.thin
    parallel = parallel if parallel is not None else mcmc_config.parallel
    
    ndim = model.n_params
    bounds = model.param_bounds
    
    # Initialize walkers around the starting point
    spread = np.array([(hi - lo) * 0.02 for (lo, hi) in bounds])
    pos = np.zeros((nwalkers, ndim))
    
    for i in range(nwalkers):
        pos[i] = start_params + spread * np.random.randn(ndim)
        for j, (lo, hi) in enumerate(bounds):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-6, hi - 1e-6)
    
    # Setup parallel processing
    pool = None
    if parallel:
        try:
            import multiprocessing
            n_cpus = multiprocessing.cpu_count()
            n_threads = max(1, min(n_cpus, nwalkers // 2))
            if n_threads > 1:
                pool = multiprocessing.Pool(processes=n_threads)
                print(f"  Using {n_threads} CPU cores for MCMC")
        except Exception as e:
            print(f"  Parallel processing not available: {e}")
    
    # Create sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim,
        log_probability,
        args=(model, z, H_obs, sigma),
        pool=pool
    )
    
    # Run MCMC
    print(f"  Running MCMC ({nwalkers} walkers x {nsteps} steps)...")
    sampler.run_mcmc(pos, nsteps, progress=progress)
    
    if pool is not None:
        pool.close()
    
    # Get samples
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    return sampler, flat_samples


def get_mcmc_summary(flat_samples: np.ndarray, param_names: list) -> Dict[str, dict]:
    """
    Get summary statistics from MCMC samples.
    
    Args:
        flat_samples: MCMC samples
        param_names: Parameter names
    
    Returns:
        Dictionary with summary statistics for each parameter
    """
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    
    summary = {}
    for i, name in enumerate(param_names):
        lo, med, hi = percentiles[:, i]
        summary[name] = {
            'median': med,
            'lower_error': med - lo,
            'upper_error': hi - med,
            'std': np.std(flat_samples[:, i]),
            'percentiles': {'16': lo, '50': med, '84': hi}
        }
    
    return summary


def compute_dic(
    flat_samples: np.ndarray,
    model: CosmologicalModel,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Deviance Information Criterion (DIC).
    
    DIC = D(theta_bar) + 2*pD, where pD = D_bar - D(theta_bar)
    and D = -2*log-likelihood.
    
    Args:
        flat_samples: MCMC samples
        model: Cosmological model
        z: Redshift values
        H_obs: Observed H(z)
        sigma: Uncertainties
    
    Returns:
        Tuple of (DIC, pD)
    """
    def D(params):
        return -2 * log_likelihood(params, model, z, H_obs, sigma)
    
    D_samples = np.array([D(theta) for theta in flat_samples])
    D_bar = np.mean(D_samples)
    
    theta_bar = np.mean(flat_samples, axis=0)
    D_hat = D(theta_bar)
    
    pD = D_bar - D_hat
    dic = D_hat + 2 * pD
    
    return dic, pD