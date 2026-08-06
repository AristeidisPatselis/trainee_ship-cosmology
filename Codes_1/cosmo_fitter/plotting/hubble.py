# plotting/hubble.py
"""Hubble diagram plotting utilities."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from models.base import CosmologicalModel
from .base import setup_plotting, get_figsize, save_figure


def plot_hubble_diagram(
    model: CosmologicalModel,
    best_params: np.ndarray,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray,
    output_dir: Path,
    filename: Optional[str] = None,
    show_lcdm: bool = True,
) -> None:
    """
    Plot Hubble diagram with residuals.
    
    Args:
        model: Cosmological model
        best_params: Best-fit parameters
        z: Redshift values
        H_obs: Observed H(z)
        sigma: Uncertainties
        output_dir: Output directory
        filename: Output filename
        show_lcdm: Show LCDM comparison
    """
    setup_plotting()
    
    if filename is None:
        filename = f"hubble_{model.name}.png"
    
    # Compute model predictions
    z_smooth = np.linspace(0, z.max() * 1.05, 300)
    H_model_smooth = model.H(z_smooth, best_params)
    H_model_data = model.H(z, best_params)
    
    # Residuals
    residuals = H_obs - H_model_data
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=get_figsize(), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    
    # Main plot
    ax1.errorbar(z, H_obs, yerr=sigma, fmt='o', color='crimson',
                 ms=4, capsize=2, label='Cosmic chronometer data')
    ax1.plot(z_smooth, H_model_smooth, color='navy', lw=2,
             label=rf'{model.name} fit')
    
    # LCDM comparison
    if show_lcdm:
        H0, Om = best_params[0], best_params[1]  # Assumes H0, Om are first two params
        H_lcdm_smooth = model.H_lcdm(z_smooth, H0, Om)
        ax1.plot(z_smooth, H_lcdm_smooth, color='green', lw=1.5, ls='--',
                 label=r'$\Lambda$CDM (same $H_0,\Omega_{m,0}$)')
    
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title(f'Hubble diagram: {model.name}')
    ax1.legend()
    
    # Residuals
    ax2.errorbar(z, residuals, yerr=sigma, fmt='o', color='crimson', ms=4, capsize=2)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')
    
    fig.tight_layout()
    save_figure(fig, filename, output_dir)


def plot_hubble_diagram_comparison(
    model_results: dict,
    z: np.ndarray,
    H_obs: np.ndarray,
    sigma: np.ndarray,
    output_dir: Path,
    filename: str = "hubble_comparison.png",
) -> None:
    """
    Plot Hubble diagram comparing multiple models.
    
    Args:
        model_results: Dictionary mapping model names to fit results
        z: Redshift values
        H_obs: Observed H(z)
        sigma: Uncertainties
        output_dir: Output directory
        filename: Output filename
    """
    setup_plotting()
    
    z_smooth = np.linspace(0, z.max() * 1.05, 300)
    
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=get_figsize(), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    
    # Data
    ax1.errorbar(z, H_obs, yerr=sigma, fmt='o', color='k',
                 ms=4, capsize=2, label='Data', zorder=5)
    ax2.axhline(0, color='gray', lw=1.5, ls='--')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot each model
    for (name, result), color in zip(model_results.items(), colors):
        best_params = result['best_params']
        H_model_smooth = result['model'].H(z_smooth, best_params)
        H_model_data = result['model'].H(z, best_params)
        
        ax1.plot(z_smooth, H_model_smooth, color=color, lw=2,
                 label=f'{name}')
        ax2.plot(z, H_obs - H_model_data, 'o', color=color, ms=4, alpha=0.7)
    
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble diagram: model comparison')
    ax1.legend()
    
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')
    
    fig.tight_layout()
    save_figure(fig, filename, output_dir)