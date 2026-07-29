#!/usr/bin/env python3
"""
H_dot_alpha_fit.py
==================
Fit only the alpha parameter in the modified Friedmann equation:

    H(z)^2 = H0^2 * [Om*(1+z)^3 + b*H(z)^delta] - alpha*(1+z)*H(z)*dH/dz

where b = (1 - Om)*H0^(-delta).

This program fixes Om, H0, and delta at their best-fit values from the bH^δ model,
and performs a 1-parameter fit for alpha, including:
- Chi-squared minimization
- MCMC sampling for uncertainties
- Profile likelihood plots
- Confidence intervals

All plots are saved as PNG files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, brentq
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
# 3. CHI-SQUARED FUNCTION
# =============================================================================

def chisq_alpha(alpha, Om_fixed, H0_fixed, delta_fixed, z_data, H_data, sigma_data):
    """
    Chi-squared for alpha only, with Om, H0, delta fixed.
    """
    b = (1 - Om_fixed) * H0_fixed**(-delta_fixed)
    theorH = H_of_z_Hdot(z_data, Om_fixed, H0_fixed, b, delta_fixed, alpha)
    if not np.all(np.isfinite(theorH)):
        return 1e12
    return np.sum(((H_data - theorH) / sigma_data)**2)


def lcdm_H(z, Om, H0):
    """Standard ΛCDM H(z)."""
    return H0 * np.sqrt(Om * (1+z)**3 + (1 - Om))


def chisq_lcdm(pars, z_data, H_data, sigma_data):
    """Chi-squared for ΛCDM."""
    Om, H0 = pars
    theorH = lcdm_H(z_data, Om, H0)
    return np.sum(((H_data - theorH) / sigma_data)**2)


# =============================================================================
# 4. bH^δ MODEL (FOR GETTING BEST-FIT Om, H0, delta)
# =============================================================================

def H_of_z_bHdelta(z, Om_m0, H0, delta):
    """
    Algebraic bH^δ model (no H_dot term).
    Solves: H^2 = H0^2*Om*(1+z)^3 + H0^2*b*H^δ
    """
    b = (1 - Om_m0) * H0**(-delta)
    rhs_matter = H0**2 * Om_m0 * (1 + z)**3
    
    def f(Hval):
        return Hval**2 - H0**2 * b * Hval**delta - rhs_matter
    
    E_guess = np.sqrt(Om_m0*(1+z)**3 + (1-Om_m0))
    H_guess = max(H0*E_guess, 1e-6)
    lo, hi = H_guess, H_guess
    for _ in range(80):
        lo /= 1.5
        hi *= 1.5
        if np.sign(f(lo)) != np.sign(f(hi)):
            try:
                return brentq(f, lo, hi, xtol=1e-10)
            except:
                return np.nan
    return np.nan


def chisq_bHdelta(pars, z_data, H_data, sigma_data):
    """Chi-squared for bH^δ model."""
    Om_m0, H0, delta = pars
    sum_chisq = 0.0
    for z, H_obs, sigma in zip(z_data, H_data, sigma_data):
        theorH = H_of_z_bHdelta(z, Om_m0, H0, delta)
        if not np.isfinite(theorH):
            return 1e12
        sum_chisq += (H_obs - theorH)**2 / sigma**2
    return sum_chisq


# =============================================================================
# 5. MCMC SAMPLING FOR ALPHA
# =============================================================================

def run_mcmc_alpha(Om_fixed, H0_fixed, delta_fixed, z_data, H_data, sigma_data,
                   alpha_best, nwalkers=32, nsteps=2000, discard=500, thin=15):
    """
    Run MCMC for alpha only.
    """
    def log_prior(alpha):
        if 0.01 < alpha < 5.0:
            return 0.0
        return -np.inf
    
    def log_likelihood(alpha):
        c = chisq_alpha(alpha[0], Om_fixed, H0_fixed, delta_fixed,
                        z_data, H_data, sigma_data)
        if c >= 1e12:
            return -np.inf
        return -0.5 * c
    
    def log_prob(theta):
        lp = log_prior(theta[0])
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)
    
    ndim = 1
    pos = alpha_best + 1e-2 * np.random.randn(nwalkers, ndim)
    pos = np.clip(pos, 0.01, 5.0)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=False)
    
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    percentiles = np.percentile(flat_samples[:, 0], [16, 50, 84])
    
    return flat_samples, percentiles


# =============================================================================
# 6. PLOTTING FUNCTIONS
# =============================================================================

def plot_chi2_alpha(alpha_grid, chi2_vals, alpha_best, alpha_err_low, alpha_err_high,
                    filename='chi2_alpha.png'):
    """Plot χ² vs alpha with confidence levels."""
    chi2_min = chi2_vals.min()
    delta_chi2 = chi2_vals - chi2_min
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(alpha_grid, delta_chi2, 'navy', lw=2.5, label=r'$\Delta\chi^2(\alpha)$')
    
    # Confidence levels for 1 DOF
    ax.axhline(1, color='gray', ls='--', lw=1.2, alpha=0.7, label=r'1$\sigma$')
    ax.axhline(4, color='gray', ls='--', lw=1.2, alpha=0.7, label=r'2$\sigma$')
    ax.axhline(9, color='gray', ls='--', lw=1.2, alpha=0.7, label=r'3$\sigma$')
    
    # Best fit and uncertainty
    ax.axvline(alpha_best, color='crimson', ls=':', lw=2, label=f'Best: $\\alpha={alpha_best:.3f}$')
    if alpha_err_low > 0 and alpha_err_high > 0:
        ax.axvspan(alpha_best - alpha_err_low, alpha_best + alpha_err_high,
                   alpha=0.2, color='crimson', label=r'1$\sigma$ region')
    
    ax.set_xlabel(r'$\alpha$', fontsize=13)
    ax.set_ylabel(r'$\Delta\chi^2(\alpha)$', fontsize=13)
    ax.set_title(r'Profile likelihood: $\Delta\chi^2$ vs $\alpha$', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, max(delta_chi2.max() * 0.9, 10))
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  - {filename}")


def plot_alpha_posterior(alpha_samples, alpha_best, alpha_err_low, alpha_err_high,
                         filename='alpha_posterior.png'):
    """Plot alpha posterior distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    n, bins, _ = ax.hist(alpha_samples, bins=40, color='steelblue', alpha=0.8,
                         density=True, label='Posterior samples')
    
    # Best fit line
    ax.axvline(alpha_best, color='crimson', ls='--', lw=2, label=f'Best: $\\alpha={alpha_best:.3f}$')
    
    # 1σ region
    if alpha_err_low > 0 and alpha_err_high > 0:
        ax.axvspan(alpha_best - alpha_err_low, alpha_best + alpha_err_high,
                   alpha=0.2, color='crimson', label=r'1$\sigma$ region')
    
    # Gaussian fit
    mu = alpha_best
    sigma = (alpha_err_low + alpha_err_high) / 2
    x = np.linspace(alpha_samples.min(), alpha_samples.max(), 200)
    y = norm.pdf(x, mu, sigma)
    ax.plot(x, y, 'r-', lw=1.5, label='Gaussian fit')
    
    ax.set_xlabel(r'$\alpha$', fontsize=13)
    ax.set_ylabel('Posterior density', fontsize=13)
    ax.set_title(r'Posterior distribution of $\alpha$', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  - {filename}")


def plot_H_fit_comparison(z_data, H_data, sigma_data, Om_fixed, H0_fixed, delta_fixed,
                          alpha_best, z_plot=None, filename='H_fit_comparison.png'):
    """Plot data and H(z) curves for different alpha values."""
    if z_plot is None:
        z_plot = np.linspace(0.001, 2.0, 200)
    
    b_fixed = (1 - Om_fixed) * H0_fixed**(-delta_fixed)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data points
    ax.errorbar(z_data, H_data, yerr=sigma_data, fmt='o', color='navy',
                capsize=3, label='Data', markersize=8)
    
    # Best fit H_dot model
    H_best = H_of_z_Hdot(z_plot, Om_fixed, H0_fixed, b_fixed, delta_fixed, alpha_best)
    ax.plot(z_plot, H_best, 'r-', lw=2.5, label=f'$\\alpha={alpha_best:.3f}$ (best)')
    
    # Different alpha values to show effect
    for alpha_val in [0.0, 0.5, 1.0, 2.0]:
        if alpha_val != alpha_best:
            H_alpha = H_of_z_Hdot(z_plot, Om_fixed, H0_fixed, b_fixed, delta_fixed, alpha_val)
            if np.all(np.isfinite(H_alpha)):
                ax.plot(z_plot, H_alpha, '--', lw=1.5, alpha=0.6,
                       label=f'$\\alpha={alpha_val:.1f}$')
    
    # ΛCDM comparison
    H_lcdm = lcdm_H(z_plot, Om_fixed, H0_fixed)
    ax.plot(z_plot, H_lcdm, 'k-.', lw=2, label=r'$\Lambda$CDM', alpha=0.7)
    
    ax.set_xlabel(r'$z$', fontsize=13)
    ax.set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=13)
    ax.set_title(f'Best fit: $\\Omega_m={Om_fixed:.3f}$, $H_0={H0_fixed:.1f}$, '
                 f'$\\delta={delta_fixed:.3f}$, $\\alpha={alpha_best:.3f}$',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  - {filename}")


def print_confidence_intervals(alpha_best, alpha_err_low, alpha_err_high,
                               percentiles=None):
    """Print alpha confidence intervals."""
    print("\nAlpha confidence intervals:")
    print(f"  Best fit: {alpha_best:.4f}")
    print(f"  1σ: [{alpha_best - alpha_err_low:.4f}, {alpha_best + alpha_err_high:.4f}]")
    print(f"      ({alpha_err_low:.4f} / +{alpha_err_high:.4f})")
    
    if percentiles is not None:
        print(f"  MCMC 68%: [{percentiles[0]:.4f}, {percentiles[2]:.4f}]")
        print(f"  MCMC median: {percentiles[1]:.4f}")
    
    # Check if alpha is consistent with 0
    n_sigma = abs(alpha_best) / max(alpha_err_low, alpha_err_high)
    print(f"\n  alpha is {n_sigma:.2f}σ away from 0")
    
    if n_sigma > 3:
        print("  ⚠️ alpha > 3σ from 0: significant evidence for H_dot term!")
    elif n_sigma > 2:
        print("  ⚠️ alpha > 2σ from 0: some evidence for H_dot term")
    else:
        print("  ✓ alpha consistent with 0: no significant evidence for H_dot term")


# =============================================================================
# 7. MAIN PROGRAM
# =============================================================================

def main():
    print("="*70)
    print("Alpha Parameter Fitter for H_dot Model")
    print("="*70)
    
    # Setup
    setup_matplotlib()
    
    # Load data
    z_data, H_data, sigma_data = load_all_data()
    print(f"\nLoaded {len(z_data)} data points")
    print(f"z range: [{z_data.min():.3f}, {z_data.max():.3f}]")
    print(f"H range: [{H_data.min():.1f}, {H_data.max():.1f}]")
    
    # ========================================================================
    # Step 1: Fit bH^δ model to get Om, H0, delta
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 1: Fitting bH^δ model (Om, H0, delta)")
    print("-"*50)
    
    result_bHdelta = minimize(
        chisq_bHdelta,
        [0.3, 70.0, 0.0],
        args=(z_data, H_data, sigma_data),
        method='L-BFGS-B',
        bounds=[(0.01, 0.99), (50, 100), (-3.0, 3.0)]
    )
    
    Om_fixed, H0_fixed, delta_fixed = result_bHdelta.x
    chi2_bHdelta = result_bHdelta.fun
    dof_bHdelta = len(z_data) - 3
    
    print(f"Ω_m = {Om_fixed:.4f}")
    print(f"H_0 = {H0_fixed:.2f}")
    print(f"δ   = {delta_fixed:.4f}")
    print(f"χ²  = {chi2_bHdelta:.2f}")
    print(f"χ²/dof = {chi2_bHdelta/dof_bHdelta:.3f}")
    
    # Also fit ΛCDM for comparison
    result_lcdm = minimize(
        chisq_lcdm,
        [0.3, 70.0],
        args=(z_data, H_data, sigma_data),
        method='L-BFGS-B',
        bounds=[(0.01, 0.99), (50, 100)]
    )
    Om_lcdm, H0_lcdm = result_lcdm.x
    chi2_lcdm = result_lcdm.fun
    
    print(f"\nΛCDM comparison: χ² = {chi2_lcdm:.2f}")
    print(f"  Δχ² (bH^δ - ΛCDM) = {chi2_bHdelta - chi2_lcdm:.2f}")
    
    # ========================================================================
    # Step 2: Fit alpha
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 2: Fitting alpha (with Om, H0, δ fixed)")
    print("-"*50)
    
    # Minimize for alpha
    result_alpha = minimize(
        lambda x: chisq_alpha(x[0], Om_fixed, H0_fixed, delta_fixed,
                              z_data, H_data, sigma_data),
        [0.1],
        method='L-BFGS-B',
        bounds=[(0.01, 5.0)]
    )
    
    alpha_best = result_alpha.x[0]
    chi2_alpha = result_alpha.fun
    dof_alpha = len(z_data) - 4
    
    print(f"α (best) = {alpha_best:.4f}")
    print(f"χ² = {chi2_alpha:.2f}")
    print(f"χ²/dof = {chi2_alpha/dof_alpha:.3f}")
    print(f"Δχ² (from bH^δ) = {chi2_alpha - chi2_bHdelta:.2f}")
    
    # ========================================================================
    # Step 3: Compute χ² profile for alpha
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 3: Computing χ² profile")
    print("-"*50)
    
    alpha_grid = np.linspace(0.01, 3.0, 200)
    chi2_grid = np.array([
        chisq_alpha(a, Om_fixed, H0_fixed, delta_fixed, z_data, H_data, sigma_data)
        for a in alpha_grid
    ])
    
    # Find 1σ confidence interval from Δχ² = 1
    chi2_min = chi2_grid.min()
    delta_chi2 = chi2_grid - chi2_min
    
    # Find where Δχ² crosses 1
    alpha_plus = None
    alpha_minus = None
    
    # Find lower bound
    mask_low = (alpha_grid < alpha_best) & (delta_chi2 <= 1.05)
    if np.any(mask_low):
        alpha_minus = alpha_grid[mask_low].min()
        # Interpolate for more accuracy
        idx = np.where(mask_low)[0]
        if len(idx) > 0:
            i = idx[0]
            if i > 0 and delta_chi2[i] > 1:
                alpha_minus = np.interp(1, [delta_chi2[i-1], delta_chi2[i]],
                                        [alpha_grid[i-1], alpha_grid[i]])
    
    # Find upper bound
    mask_high = (alpha_grid > alpha_best) & (delta_chi2 <= 1.05)
    if np.any(mask_high):
        idx = np.where(mask_high)[0]
        if len(idx) > 0:
            i = idx[-1]
            if i < len(alpha_grid) - 1 and delta_chi2[i+1] > 1:
                alpha_plus = np.interp(1, [delta_chi2[i], delta_chi2[i+1]],
                                       [alpha_grid[i], alpha_grid[i+1]])
    
    # Use grid points if interpolation failed
    if alpha_minus is None:
        alpha_minus = alpha_grid[delta_chi2 <= 1][0]
    if alpha_plus is None:
        alpha_plus = alpha_grid[delta_chi2 <= 1][-1]
    
    alpha_err_low = alpha_best - alpha_minus
    alpha_err_high = alpha_plus - alpha_best
    
    print(f"α = {alpha_best:.4f} +{alpha_err_high:.4f} / -{alpha_err_low:.4f}")
    
    # ========================================================================
    # Step 4: MCMC for alpha
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 4: MCMC sampling for alpha")
    print("-"*50)
    
    alpha_samples, percentiles = run_mcmc_alpha(
        Om_fixed, H0_fixed, delta_fixed,
        z_data, H_data, sigma_data,
        alpha_best,
        nwalkers=32, nsteps=3000
    )
    
    alpha_median = percentiles[1]
    alpha_low_68 = alpha_median - percentiles[0]
    alpha_high_68 = percentiles[2] - alpha_median
    
    print(f"α (MCMC) = {alpha_median:.4f} +{alpha_high_68:.4f} / -{alpha_low_68:.4f}")
    
    # Check if alpha is consistent with 0
    n_sigma_mcmc = abs(alpha_median) / max(alpha_low_68, alpha_high_68)
    print(f"α is {n_sigma_mcmc:.2f}σ away from 0")
    
    # ========================================================================
    # Step 5: Generate plots
    # ========================================================================
    print("\n" + "-"*50)
    print("Step 5: Generating plots")
    print("-"*50)
    
    # Plot 1: χ² profile
    plot_chi2_alpha(alpha_grid, chi2_grid, alpha_best,
                    alpha_err_low, alpha_err_high,
                    filename='chi2_alpha.png')
    
    # Plot 2: Posterior distribution
    plot_alpha_posterior(alpha_samples, alpha_best,
                         alpha_err_low, alpha_err_high,
                         filename='alpha_posterior.png')
    
    # Plot 3: H(z) fit comparison
    plot_H_fit_comparison(z_data, H_data, sigma_data,
                          Om_fixed, H0_fixed, delta_fixed, alpha_best,
                          filename='H_fit_comparison.png')
    
    # ========================================================================
    # Step 6: Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nFixed parameters from bH^δ fit:")
    print(f"  Ω_m = {Om_fixed:.4f}")
    print(f"  H_0 = {H0_fixed:.2f} km/s/Mpc")
    print(f"  δ   = {delta_fixed:.4f}")
    
    print(f"\nAlpha fit results:")
    print(f"  α (best)  = {alpha_best:.4f}")
    print(f"  α (χ² 1σ) = {alpha_best:.4f} +{alpha_err_high:.4f} / -{alpha_err_low:.4f}")
    print(f"  α (MCMC)  = {alpha_median:.4f} +{alpha_high_68:.4f} / -{alpha_low_68:.4f}")
    
    print(f"\nModel comparison:")
    print(f"  ΛCDM χ²       = {chi2_lcdm:.2f} (dof={len(z_data)-2})")
    print(f"  bH^δ χ²       = {chi2_bHdelta:.2f} (dof={len(z_data)-3})")
    print(f"  bH^δ + α χ²   = {chi2_alpha:.2f} (dof={len(z_data)-4})")
    print(f"  Δχ² (α vs bH^δ) = {chi2_alpha - chi2_bHdelta:.2f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    n_sigma_final = abs(alpha_median) / max(alpha_low_68, alpha_high_68)
    
    if n_sigma_final > 3:
        print("  ⚠️ α > 3σ from 0: STRONG evidence for H_dot term!")
        print("  The data significantly prefer an evolving dark energy model.")
    elif n_sigma_final > 2:
        print("  ⚠️ α > 2σ from 0: Moderate evidence for H_dot term.")
        print("  More data would be needed for a definitive conclusion.")
    else:
        print("  ✓ α is consistent with 0 within uncertainties.")
        print("  The data do not require the H_dot extension.")
    
    print("\nPlots saved:")
    print("  - chi2_alpha.png: χ² profile vs α")
    print("  - alpha_posterior.png: Posterior distribution of α")
    print("  - H_fit_comparison.png: H(z) fits for different α values")
    print("\nDone!")


if __name__ == "__main__":
    main()