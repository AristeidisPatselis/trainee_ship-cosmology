# plotting/base.py
"""Base plotting utilities."""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from config.manager import get_config


def setup_plotting(use_latex: Optional[bool] = None) -> None:
    """
    Setup matplotlib for plotting.
    
    Args:
        use_latex: Whether to use LaTeX rendering
    """
    config = get_config()
    use_latex = use_latex if use_latex is not None else config.plotting.use_latex
    
    try:
        if use_latex:
            rc('text', usetex=True)
            rc('font', family='serif')
    except Exception:
        rc('text', usetex=False)
        rc('font', family='DejaVu Sans')
    
    rc('figure', dpi=config.plotting.dpi)


def get_figsize(scale: float = 1.0) -> tuple:
    """Get default figure size."""
    config = get_config()
    return tuple(s * scale for s in config.plotting.figsize)


def save_figure(fig: plt.Figure, filename: str, output_dir: Path) -> None:
    """Save figure with consistent settings."""
    config = get_config()
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=config.plotting.dpi, bbox_inches='tight')
    print(f"  Saved: {filepath}")


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[tuple] = None,
    **kwargs
) -> tuple:
    """Create subplots with default settings."""
    if figsize is None:
        figsize = get_figsize()
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, axes


def format_parameter_label(param_name: str) -> str:
    """Format a parameter name for plotting."""
    labels = {
        'H0': r'$H_0$ [km/s/Mpc]',
        'Om': r'$\Omega_{m,0}$',
        'delta': r'$\delta$',
        'alpha': r'$\alpha$',
        'b': r'$b$',
    }
    return labels.get(param_name, param_name)


def add_confidence_contours(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    levels: list,
    colors: Optional[list] = None,
    labels: Optional[dict] = None,
    fill: bool = True,
    **kwargs
) -> None:
    """
    Add confidence contours to a plot.
    
    Args:
        ax: Matplotlib axes
        X: X grid
        Y: Y grid
        Z: Z values (delta chi2)
        levels: Contour levels
        colors: Colors for contours
        labels: Labels for contours
        fill: Whether to fill between contours
    """
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    if labels is None:
        labels = {
            levels[0]: r'1$\sigma$',
            levels[1]: r'2$\sigma$',
            levels[2]: r'3$\sigma$'
        }
    
    # Filled contours
    if fill:
        filled_levels = [0] + levels + [max(Z.max(), levels[-1] + 1)]
        ax.contourf(
            X, Y, Z,
            levels=filled_levels,
            colors=['#08306b', '#4292c6', '#9ecae1', 'white'],
            alpha=0.3,
            **kwargs
        )
    
    # Contour lines
    cs = ax.contour(X, Y, Z, levels=levels, colors=colors, linewidths=1.5, **kwargs)
    ax.clabel(cs, cs.levels, inline=True, fmt=labels, fontsize=10)