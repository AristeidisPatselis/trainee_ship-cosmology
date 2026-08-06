# plotting/contours.py
"""Contour plotting utilities."""

import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from models.base import CosmologicalModel
from .base import setup_plotting, get_figsize, save_figure, format_parameter_label


def plot_contours(
    model: CosmologicalModel,
    flat_samples: np.ndarray,
    param_indices: Tuple[int, int],
    output_dir: Path,
    filename: Optional[str] = None,
    truth_params: Optional[np.ndarray] = None,
    levels: Optional[List[float]] = None,
    show_titles: bool = True,
) -> plt.Figure:
    """
    Plot 2D confidence contours from MCMC samples.
    
    Args:
        model: Cosmological model
        flat_samples: MCMC samples
        param_indices: Indices of parameters to plot
        output_dir: Output directory
        filename: Output filename
        truth_params: True parameters to mark
        levels: Contour levels
        show_titles: Show parameter labels
    
    Returns:
        Matplotlib figure
    """
    import corner
    
    setup_plotting()
    
    if filename is None:
        filename = f"corner_{model.name}.png"
    
    param_names = model.param_names
    param_labels = [format_parameter_label(name) for name in param_names]
    
    # Get parameter labels for selected indices
    labels = [param_labels[i] for i in param_indices]
    
    # Truth values if provided
    truths = [truth_params[i] for i in param_indices] if truth_params is not None else None
    
    fig = corner.corner(
        flat_samples[:, param_indices],
        labels=labels,
        truths=truths,
        show_titles=show_titles,
        levels=levels or [0.68, 0.95, 0.997],
        quantiles=[0.16, 0.5, 0.84],
        title_kwargs={"fontsize": 12},
    )
    
    save_figure(fig, filename, output_dir)
    return fig


def plot_delta_chi2_contours(
    model: CosmologicalModel,
    best_params: np.ndarray,
    chi2_best: float,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray,
    param_pairs: List[Tuple[int, int]],
    output_dir: Path,
    grid_size: Optional[int] = None,
) -> None:
    """
    Plot Delta-chi2 contours for parameter pairs.
    
    Args:
        model: Cosmological model
        best_params: Best-fit parameters
        chi2_best: Best-fit chi2
        z: Redshift values
        H_obs: Observed H(z)
        sigma: Uncertainties
        param_pairs: List of parameter index pairs
        output_dir: Output directory
        grid_size: Grid size for contour calculation
    """
    from config.manager import get_config
    
    setup_plotting()
    config = get_config()
    grid_size = grid_size or config.contours.grid_size
    
    param_names = model.param_names
    param_labels = [format_parameter_label(name) for name in param_names]
    
    for i, j in param_pairs:
        fig, ax = plt.subplots(figsize=get_figsize())
        
        # Create grid
        bounds = model.param_bounds
        p1_lo, p1_hi = bounds[i]
        p2_lo, p2_hi = bounds[j]
        
        # Center grid on best fit
        p1_best, p2_best = best_params[i], best_params[j]
        p1_range = min(p1_hi - p1_lo, 4 * abs(p1_best) if p1_best != 0 else 2)
        p2_range = min(p2_hi - p2_lo, 4 * abs(p2_best) if p2_best != 0 else 2)
        
        p1_min = max(p1_lo, p1_best - p1_range)
        p1_max = min(p1_hi, p1_best + p1_range)
        p2_min = max(p2_lo, p2_best - p2_range)
        p2_max = min(p2_hi, p2_best + p2_range)
        
        p1_grid = np.linspace(p1_min, p1_max, grid_size)
        p2_grid = np.linspace(p2_min, p2_max, grid_size)
        X, Y = np.meshgrid(p1_grid, p2_grid)
        
        # Compute chi2 grid
        chi2_grid = np.zeros_like(X)
        other_params = [k for k in range(model.n_params) if k not in (i, j)]
        
        for idx, (p1_val, p2_val) in enumerate(zip(X.ravel(), Y.ravel())):
            params = best_params.copy()
            params[i] = p1_val
            params[j] = p2_val
            chi2_grid.ravel()[idx] = model.chi2(params, z, H_obs, sigma)
        
        delta_chi2 = chi2_grid - chi2_best
        
        # Plot contours
        levels = config.contours.levels_2d
        from .base import add_confidence_contours
        add_confidence_contours(ax, X, Y, delta_chi2, levels)
        
        # Mark best fit
        ax.plot(p1_best, p2_best, 'k*', ms=14, label='best fit')
        
        ax.set_xlabel(param_labels[i])
        ax.set_ylabel(param_labels[j])
        ax.set_title(fr'$\Delta\chi^2$ contours: {param_labels[i]} vs {param_labels[j]}')
        ax.legend()
        
        filename = f"contour_{param_names[i]}_{param_names[j]}.png"
        save_figure(fig, filename, output_dir)