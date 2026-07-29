#!/usr/bin/env python3
"""
H_dot_lcdm_full.py
==================
Complete implementation of the modified Friedmann equation fit with H_dot term:

    H(z)^2 = H0^2 * [Om*(1+z)^3 + b*H(z)^delta] - alpha*(1+z)*H(z)*dH/dz

where b = (1 - Om)*H0^(-delta) (derived from z=0 condition).

This program:
1. Loads cosmic chronometer H(z) data
2. Solves the ODE numerically for H(z)
3. Fits the 4-parameter model (Om, H0, delta, alpha) using differential evolution
4. Performs MCMC sampling for parameter uncertainties
5. Generates confidence contours and diagnostic plots

All plots are saved as PNG files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm
import emcee
import corner
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

def setup_matplotlib():
    """Configure matplotlib with LaTeX if available."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
        fig_test = plt.figure()
        plt.text(0.5, 0.5, r"$\alpha$")
        fig_test.canvas.draw()
        plt.close(fig_test)
    except Exception:
        print("Note: LaTeX unavailable, using mathtext.")
        rc('text', usetex=False)
        rc('font', family='DejaVu Sans')


def load_clean_data(filename, script_dir=None):
    """Load numeric data, handling stray bracket artifacts."""
    if script_dir is None:
        script_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(script_dir, filename)
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.split(']')[-1].strip()
            if clean_line:
                data.append(float(clean_line))
    return np.array(data)


def load_all_data():
    """Load z, H(z), and sigma_H from disk."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    z_vals = load_clean_data('z_vals.txt', script_dir)
    H_vals = load_clean_data('H_vals.txt', script_dir)
    sigma_vals = load_clean_data('sigma_vals.txt', script_dir)
    return z_vals, H_vals, sigma_vals


# =============================================================================
# 2. ODE SOLVER FOR THE H_dot MODEL
# =============================================================================

def H_of_z_Hdot(z_array, Om_m0, H0, b, delta, alpha):
    """
    Solve the ODE:
        (alpha*(1+z)/2) * du/dz = H0^2*Om*(1+z)^3 + H0^2*b*u^(delta/2) - u
    with u(0) = H0^2, where u = H^2.
    
    Returns H(z) = sqrt(u(z)).
    """
    z_array = np.atleast_1d(z_array).astype(float)
    z_max = max(z_array.max(), 1e-6)
    
    # Clip alpha to avoid division by zero
    if alpha < 1e-10:
        alpha = 1e-10
    
    def rhs(zz, uu):
        uu_safe = min(max(uu[0], 1e-10), 1e10)
        term = b * uu_safe**(delta / 2.0)
        dudz = (H0**2 * Om_m0 * (1 + zz)**3 + H0**2 * term - uu_safe) / (alpha * (1 + zz) / 2.0)
        return [np.clip(dudz, -1e10, 1e10)]
    
    try:
        sol = solve_ivp(
            rhs, [0.0, z_max], [H0**2],
            t_eval=np.sort(np.unique(np.append(z_array, 0.0))),
            rtol=1e-6, atol=1e-8, method='LSODA', max_step=0.05
        )
    except Exception:
        return np.full_like(z_array, np.nan)
    
    if not sol.success:
        return np.full_like(z_array, np.nan)
    
    u_of_z = np.interp(z_array, sol.t, sol.y[0])
    if np.any(~np.isfinite(u_of_z)) or np.any(u_of_z <= 0):
        return np.full_like(z_array, np.nan)
    return np.sqrt(u_of_z)


# =============================================================================
# 3. CHI-SQUARED FUNCTIONS
# =============================================================================

def chisq_Hdot(pars, z_data, H_data, sigma_data):
    """
    Chi-squared for the 4-parameter H_dot model.
    pars = [Om_m0, H0, delta, alpha]
    """
    Om_m0, H0, delta, alpha = pars
    
    # b is derived from the z=0 condition (same as Part 2)
    b = (1 - Om_m0) * H0**(-delta)
    
    theorH = H_of_z_Hdot(z_data, Om_m0, H0, b, delta, alpha)
    if not np.all(np.isfinite(theorH)):
        return 1e12
    
    return np.sum(((H_data - theorH) / sigma_data)**2)


def chisq_Hdot_2p(delta_alpha, Om_m0_fixed, H0_fixed, z_data, H_data, sigma_data):
    """
    2-parameter chi-squared with Om and H0 fixed.
    delta_alpha = [delta, alpha]
    """
    delta, alpha = delta_alpha
    b = (1 - Om_m0_fixed) * H0_fixed**(-delta)
    theorH = H_of_z_Hdot(z_data, Om_m0_fixed, H0_fixed, b, delta, alpha)
    if not np.all(np.isfinite(theorH)):
        return 1e12
    return np.sum(((H_data - theorH) / sigma_data)**2)


# =============================================================================
# 4. BASE ΛCDM MODEL (FOR COMPARISON)
# =============================================================================

def lcdm_H(z, Om, H0):
    """Standard ΛCDM H(z)."""
    return H0 * np.sqrt(Om * (1+z)**3 + (1 - Om))


def chisq_lcdm(pars, z_data, H_data, sigma_data):
    """Chi-squared for ΛCDM."""
    Om, H0 = pars
    theorH = lcdm_H(z_data, Om, H0)
    return np.sum(((H_data - theorH) / sigma_data)**2)


# =============================================================================
# 5. MCMC SAMPLING
# =============================================================================

def run_mcmc_4param(z_data, H_data, sigma_data, popt, nwalkers=32, nsteps=2000,
                    discard=500, thin=15):
    """
    Run MCMC for the 4-parameter H_dot model.
    
    Parameters:
        popt: [Om_m0, H0, delta, alpha] best-fit values
    Returns:
        flat_samples: (n_samples, 4) array of MCMC samples
        percentiles: dict with 16/50/84 percentiles for each parameter
    """
    
    def log_prior(theta):
        Om_m0, H0, delta, alpha = theta
        if (0.05 < Om_m0 < 0.95 and 55 < H0 < 90 and
            -2.5 < delta < 2.5 and 0.05 < alpha < 5.0):
            return 0.0
        return -np.inf
    
    def log_likelihood(theta):
        Om_m0, H0, delta, alpha = theta
        b = (1 - Om_m0) * H0**(-delta)
        theorH = H_of_z_Hdot(z_data, Om_m0, H0, b, delta, alpha)
        if not np.all(np.isfinite(theorH)):
            return -np.inf
        return -0.5 * np.sum(((H_data - theorH) / sigma_data)**2)
    
    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    ndim = 4
    # Initial positions: small ball around best-fit
    pos = popt + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([1, 10, 0.1, 0.1])
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=False)
    
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    percentiles = {}
    labels = [r'$\Omega_{m,0}$', r'$H_0$', r'$\delta$', r'$\alpha$']
    for i, label in enumerate(labels):
        p = np.percentile(flat_samples[:, i], [16, 50, 84])
        percentiles[label] = p
    
    return flat_samples, percentiles


def run_mcmc_2param(delta_alpha_best, Om_fixed, H0_fixed, z_data, H_data, sigma_data,
                    nwalkers=24, nsteps=1000, discard=250, thin=8):
    """
    Run MCMC for the 2-parameter (delta, alpha) model with Om and H0 fixed.
    """
    
    def log_prior(theta):
        delta, alpha = theta
        if (-2.5 < delta < 2.5 and 0.05 < alpha < 5.0):
            return 0.0
        return -np.inf
    
    def log_likelihood(theta):
        delta, alpha = theta
        c = chisq_Hdot_2p([delta, alpha], Om_fixed, H0_fixed, z_data, H_data, sigma_data)
        if c >= 1e12:
            return -np.inf
        return -0.5 * c
    
    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    ndim = 2
    pos = delta_alpha_best + 1e-2 * np.random.randn(nwalkers, ndim) * np.array([1, 1])
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=False)
    
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    percentiles = {}
    labels = [r'$\delta$', r'$\alpha$']
    for i, label in enumerate(labels):
        p = np.percentile(flat_samples[:, i], [16, 50, 84])
        percentiles[label] = p
    
    return flat_samples, percentiles


# =============================================================================
# 6. PLOTTING FUNCTIONS
# =============================================================================

def plot_H_fit(z_data, H_data, sigma_data, Om_fit, H0_fit, delta_fit, alpha_fit,
               z_plot=None, filename='H_dot_fit.png'):
    """Plot data and best-fit H(z) curves."""
    if z_plot is None:
        z_plot = np.linspace(0.001, 2.0, 200)
    
    b_fit = (1 - Om_fit) * H0_fit**(-delta_fit)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Data points
    ax.errorbar(z_data, H_data, yerr=sigma_data, fmt='o', color='navy',
                capsize=3, label='Data')
    
    # H_dot model fit
    H_fit = H_of_z_Hdot(z_plot, Om_fit, H0_fit, b_fit, delta_fit, alpha_fit)
    ax.plot(z_plot, H_fit, 'r-', lw=2, label=r'$H$-dot model')
    
    # ΛCDM comparison (with same Om, H0)
    H_lcdm = lcdm_H(z_plot, Om_fit, H0_fit)
    ax.plot(z_plot, H_lcdm, 'k--', lw=1.5, label=r'$\Lambda$CDM')
    
    ax.set_xlabel(r'$z$', fontsize=12)
    ax.set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Best fit: $\\Omega_m={Om_fit:.3f}$, $H_0={H0_fit:.1f}$, '
                 f'$\\delta={delta_fit:.3f}$, $\\alpha={alpha_fit:.3f}$',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_contour_delta_alpha(Om_fixed, H0_fixed, z_data, H_data, sigma_data,
                             filename='contour_delta_alpha.png'):
    """2D χ² contour in (delta, alpha) space."""
    n_grid = 50
    delta_grid = np.linspace(-2.0, 2.0, n_grid)
    alpha_grid = np.linspace(0.05, 3.0, n_grid)
    
    Z = np.empty((n_grid, n_grid))
    for i, a in enumerate(alpha_grid):
        for j, d in enumerate(delta_grid):
            Z[i, j] = chisq_Hdot_2p([d, a], Om_fixed, H0_fixed,
                                     z_data, H_data, sigma_data)
    
    chi2_min = Z.min()
    levels = chi2_min + np.array([0, 2.30, 6.18, 11.83])  # 1σ, 2σ, 3σ for 2 DOF
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(delta_grid, alpha_grid, Z, levels=levels, cmap='viridis', alpha=0.7)
    cs = ax.contour(delta_grid, alpha_grid, Z, levels=levels, colors='white', linewidths=1.5)
    ax.clabel(cs, fmt={levels[1]: r'1$\sigma$', levels[2]: r'2$\sigma$',
                       levels[3]: r'3$\sigma$'}, fontsize=10)
    
    # Mark minimum
    idx = np.unravel_index(Z.argmin(), Z.shape)
    ax.scatter([delta_grid[idx[1]]], [alpha_grid[idx[0]]],
               marker='*', color='white', s=200, edgecolor='black', zorder=5)
    
    ax.set_xlabel(r'$\delta$', fontsize=12)
    ax.set_ylabel(r'$\alpha$', fontsize=12)
    ax.set_title(r'$\Delta\chi^2$ contours: $\delta$ vs $\alpha$', fontsize=13)
    fig.colorbar(cf, ax=ax, label=r'$\chi^2$')
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_corner(flat_samples, labels, truths=None, filename='corner_plot.png'):
    """Corner plot for MCMC samples."""
    fig = corner.corner(flat_samples, labels=labels, truths=truths,
                        show_titles=True, title_kwargs={"fontsize": 10})
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_parameter_comparison(lcdm_result, hdot_result, filename='param_comparison.png'):
    """Compare ΛCDM and Hdot model parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    labels = [r'$\Omega_{m,0}$', r'$H_0$', r'$\delta$', r'$\alpha$']
    values_lcdm = [lcdm_result[0], lcdm_result[1], 0, 0]  # δ=0, α=0 for ΛCDM
    values_hdot = [hdot_result[0], hdot_result[1], hdot_result[2], hdot_result[3]]
    
    for i, ax in enumerate(axes.flat):
        x = np.arange(2)
        ax.bar(x, [values_lcdm[i], values_hdot[i]], color=['steelblue', 'crimson'], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(['ΛCDM', 'H-dot'])
        ax.set_ylabel(labels[i])
        ax.set_title(labels[i])
    
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def plot_chi2_profile_alpha(delta_fixed, Om_fixed, H0_fixed, z_data, H_data, sigma_data,
                            alpha_range=(0.01, 3.0), n_points=50,
                            filename='chi2_profile_alpha.png'):
    """1D χ² profile over alpha with delta fixed at best value."""
    alphas = np.linspace(*alpha_range, n_points)
    chi2_vals = np.array([chisq_Hdot_2p([delta_fixed, a], Om_fixed, H0_fixed,
                                         z_data, H_data, sigma_data)
                          for a in alphas])
    
    chi2_min = chi2_vals.min()
    delta_chi2 = chi2_vals - chi2_min
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, delta_chi2, 'navy', lw=2)
    ax.axhline(1, color='gray', ls='--', lw=0.8, label=r'1$\sigma$')
    ax.axhline(4, color='gray', ls='--', lw=0.8, label=r'2$\sigma$')
    ax.axhline(9, color='gray', ls='--', lw=0.8, label=r'3$\sigma$')
    ax.set_xlabel(r'$\alpha$', fontsize=12)
    ax.set_ylabel(r'$\Delta\chi^2(\alpha)$', fontsize=12)
    ax.set_title(r'Profile likelihood: $\Delta\chi^2$ vs $\alpha$ (fixed $\delta$)', fontsize=13)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return alphas, delta_chi2


# =============================================================================
# 7. MAIN PROGRAM
# =============================================================================

def main():
    print("="*70)
    print("H_dot Modified Friedmann Equation Fitter")
    print("="*70)
    
    # Setup
    setup_matplotlib()
    
    # Load data
    z_data, H_data, sigma_data = load_all_data()
    print(f"\nLoaded {len(z_data)} data points")
    print(f"z range: [{z_data.min():.3f}, {z_data.max():.3f}]")
    print(f"H range: [{H_data.min():.1f}, {H_data.max():.1f}]")
    
    # ========================================================================
    # Step 1: ΛCDM fit (for comparison)
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 1: ΛCDM fit")
    print("-"*50)
    
    result_lcdm = minimize(chisq_lcdm, [0.3, 70.0],
                           args=(z_data, H_data, sigma_data),
                           method='L-BFGS-B',
                           bounds=[(0.01, 0.99), (50, 100)])
    
    Om_lcdm, H0_lcdm = result_lcdm.x
    chi2_lcdm = result_lcdm.fun
    dof_lcdm = len(z_data) - 2
    print(f"Ω_m = {Om_lcdm:.4f}")
    print(f"H_0 = {H0_lcdm:.2f}")
    print(f"χ² = {chi2_lcdm:.2f}")
    print(f"χ²/dof = {chi2_lcdm/dof_lcdm:.3f}")
    
    # ========================================================================
    # Step 2: Full 4-parameter H_dot fit
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 2: 4-parameter H_dot fit (differential evolution)")
    print("-"*50)
    
    # Global optimization with differential evolution
    bounds = [(0.05, 0.95), (55, 90), (-2.5, 2.5), (0.05, 5.0)]
    
    result = differential_evolution(
        chisq_Hdot,
        bounds=bounds,
        args=(z_data, H_data, sigma_data),
        seed=42,
        maxiter=150,
        tol=1e-6,
        polish=True
    )
    
    if result.success:
        print("Optimization converged successfully")
    else:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    Om_hdot, H0_hdot, delta_hdot, alpha_hdot = result.x
    chi2_hdot = result.fun
    dof_hdot = len(z_data) - 4
    
    print(f"Ω_m = {Om_hdot:.4f}")
    print(f"H_0 = {H0_hdot:.2f}")
    print(f"δ   = {delta_hdot:.4f}")
    print(f"α   = {alpha_hdot:.4f}")
    print(f"χ²  = {chi2_hdot:.2f}")
    print(f"χ²/dof = {chi2_hdot/dof_hdot:.3f}")
    
    # ========================================================================
    # Step 3: MCMC for 4-parameter model
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 3: MCMC sampling (4 parameters)")
    print("-"*50)
    
    popt_4 = [Om_hdot, H0_hdot, delta_hdot, alpha_hdot]
    flat_samples_4, percentiles_4 = run_mcmc_4param(z_data, H_data, sigma_data, popt_4,
                                                     nwalkers=32, nsteps=2000)
    
    Om_mcmc = percentiles_4[r'$\Omega_{m,0}$']
    H0_mcmc = percentiles_4[r'$H_0$']
    delta_mcmc = percentiles_4[r'$\delta$']
    alpha_mcmc = percentiles_4[r'$\alpha$']
    
    print(f"Ω_m = {Om_mcmc[1]:.4f} (+{Om_mcmc[2]-Om_mcmc[1]:.4f} / -{Om_mcmc[1]-Om_mcmc[0]:.4f})")
    print(f"H_0 = {H0_mcmc[1]:.2f} (+{H0_mcmc[2]-H0_mcmc[1]:.2f} / -{H0_mcmc[1]-H0_mcmc[0]:.2f})")
    print(f"δ   = {delta_mcmc[1]:.4f} (+{delta_mcmc[2]-delta_mcmc[1]:.4f} / -{delta_mcmc[1]-delta_mcmc[0]:.4f})")
    print(f"α   = {alpha_mcmc[1]:.4f} (+{alpha_mcmc[2]-alpha_mcmc[1]:.4f} / -{alpha_mcmc[1]-alpha_mcmc[0]:.4f})")
    
    # Check if δ is consistent with 0
    n_sigma_delta = abs(delta_mcmc[1]) / ((delta_mcmc[2] - delta_mcmc[0]) / 2)
    print(f"δ is {n_sigma_delta:.2f}σ away from 0 (ΛCDM limit)")
    
    # ========================================================================
    # Step 4: 2-parameter fit with Om, H0 fixed
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 4: 2-parameter fit (δ, α) with Om, H0 fixed")
    print("-"*50)
    
    # Use MCMC best values for Om, H0
    Om_fixed = Om_mcmc[1]
    H0_fixed = H0_mcmc[1]
    
    # Find best (δ, α) via minimization
    result_2p = minimize(
        lambda x: chisq_Hdot_2p(x, Om_fixed, H0_fixed, z_data, H_data, sigma_data),
        [delta_mcmc[1], alpha_mcmc[1]],
        method='L-BFGS-B',
        bounds=[(-2.5, 2.5), (0.05, 5.0)]
    )
    
    delta_best_2p, alpha_best_2p = result_2p.x
    chi2_2p = result_2p.fun
    print(f"δ (best)   = {delta_best_2p:.4f}")
    print(f"α (best)   = {alpha_best_2p:.4f}")
    print(f"χ² (2-param) = {chi2_2p:.2f}")
    
    # MCMC for 2-parameter model
    flat_samples_2, percentiles_2 = run_mcmc_2param(
        [delta_best_2p, alpha_best_2p], Om_fixed, H0_fixed,
        z_data, H_data, sigma_data
    )
    
    delta_2p_mcmc = percentiles_2[r'$\delta$']
    alpha_2p_mcmc = percentiles_2[r'$\alpha$']
    
    print(f"δ (MCMC) = {delta_2p_mcmc[1]:.4f} (+{delta_2p_mcmc[2]-delta_2p_mcmc[1]:.4f} / -{delta_2p_mcmc[1]-delta_2p_mcmc[0]:.4f})")
    print(f"α (MCMC) = {alpha_2p_mcmc[1]:.4f} (+{alpha_2p_mcmc[2]-alpha_2p_mcmc[1]:.4f} / -{alpha_2p_mcmc[1]-alpha_2p_mcmc[0]:.4f})")
    
    # ========================================================================
    # Step 5: Generate plots
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 5: Generating plots")
    print("-"*50)
    
    # Plot 1: H(z) fit
    plot_H_fit(z_data, H_data, sigma_data,
               Om_hdot, H0_hdot, delta_hdot, alpha_hdot,
               filename='H_dot_fit.png')
    print("  - H_dot_fit.png")
    
    # Plot 2: Corner plot (4 parameters)
    plot_corner(flat_samples_4,
                [r'$\Omega_{m,0}$', r'$H_0$', r'$\delta$', r'$\alpha$'],
                truths=[Om_hdot, H0_hdot, delta_hdot, alpha_hdot],
                filename='corner_4param.png')
    print("  - corner_4param.png")
    
    # Plot 3: Corner plot (2 parameters)
    plot_corner(flat_samples_2,
                [r'$\delta$', r'$\alpha$'],
                truths=[delta_best_2p, alpha_best_2p],
                filename='corner_2param.png')
    print("  - corner_2param.png")
    
    # Plot 4: Contour (δ, α)
    plot_contour_delta_alpha(Om_fixed, H0_fixed, z_data, H_data, sigma_data,
                             filename='contour_delta_alpha.png')
    print("  - contour_delta_alpha.png")
    
    # Plot 5: χ² profile over α
    plot_chi2_profile_alpha(delta_best_2p, Om_fixed, H0_fixed,
                            z_data, H_data, sigma_data,
                            filename='chi2_profile_alpha.png')
    print("  - chi2_profile_alpha.png")
    
    # Plot 6: Parameter comparison
    plot_parameter_comparison([Om_lcdm, H0_lcdm],
                              [Om_hdot, H0_hdot, delta_hdot, alpha_hdot],
                              filename='param_comparison.png')
    print("  - param_comparison.png")
    
    # ========================================================================
    # Step 6: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nΛCDM fit:")
    print(f"  Ω_m = {Om_lcdm:.4f}")
    print(f"  H_0 = {H0_lcdm:.2f} km/s/Mpc")
    print(f"  χ²/dof = {chi2_lcdm/dof_lcdm:.3f}")
    
    print("\nH_dot model fit (4 parameters):")
    print(f"  Ω_m = {Om_hdot:.4f}")
    print(f"  H_0 = {H0_hdot:.2f} km/s/Mpc")
    print(f"  δ   = {delta_hdot:.4f}")
    print(f"  α   = {alpha_hdot:.4f}")
    print(f"  χ²/dof = {chi2_hdot/dof_hdot:.3f}")
    
    print("\nMCMC results (4 parameters):")
    print(f"  Ω_m = {Om_mcmc[1]:.4f} (+{Om_mcmc[2]-Om_mcmc[1]:.4f} / -{Om_mcmc[1]-Om_mcmc[0]:.4f})")
    print(f"  H_0 = {H0_mcmc[1]:.2f} (+{H0_mcmc[2]-H0_mcmc[1]:.2f} / -{H0_mcmc[1]-H0_mcmc[0]:.2f})")
    print(f"  δ   = {delta_mcmc[1]:.4f} (+{delta_mcmc[2]-delta_mcmc[1]:.4f} / -{delta_mcmc[1]-delta_mcmc[0]:.4f})")
    print(f"  α   = {alpha_mcmc[1]:.4f} (+{alpha_mcmc[2]-alpha_mcmc[1]:.4f} / -{alpha_mcmc[1]-alpha_mcmc[0]:.4f})")
    
    print(f"\nδ is {n_sigma_delta:.2f}σ away from 0 (ΛCDM limit)")
    
    if abs(delta_mcmc[1]) > 2 * ((delta_mcmc[2] - delta_mcmc[0]) / 2):
        print("\n⚠️ δ > 2σ from 0: possible evidence for evolving dark energy!")
    else:
        print("\n✓ δ is consistent with 0: data do not require the H_dot extension.")
    
    print("\nAll plots saved. Done!")


if __name__ == "__main__":
    main()