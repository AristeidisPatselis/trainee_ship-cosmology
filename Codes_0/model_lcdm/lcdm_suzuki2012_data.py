#!/usr/bin/env python3
"""
Cosmological Analysis of Suzuki et al. 2012 HST Cluster Supernova Survey
This script extracts the 14 SNe Ia from the 2012 paper and performs
a full LCDM fit (optimization, contours, MCMC) on just this sample.
"""

import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.integrate import cumulative_trapezoid
from functools import lru_cache
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import emcee
import corner
from matplotlib import rc

# =============================================================================
# CONFIGURATION
# =============================================================================
C_LIGHT = 299792.458  # km/s
OUTPUT_DIR = './results_suzuki2012'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# MATPLOTLIB SETUP
# =============================================================================

def setup_matplotlib():
    """Attempts to enable LaTeX formatting for professional plots."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
    except Exception as e:
        print(f"Warning: LaTeX rendering not enabled. Error: {e}")

# =============================================================================
# 1. DATA LOADING (Suzuki et al. 2012)
# =============================================================================

def load_suzuki2012_data():
    """
    Loads the 14 SNe Ia from Suzuki et al. 2012 (HST Cluster Supernova Survey)
    that pass the quality cuts and are used for cosmology.
    
    Returns: z_vals, mu_vals, mu_err, and metadata
    """
    # Data extracted from Table 3 of the paper
    # These are the 14 SNe Ia that pass the quality cuts
    data = {
        'SN_Name': ['SCP06A4', 'SCP06C0', 'SCP06C1', 'SCP06F12', 'SCP06G4',
                    'SCP06H3', 'SCP06H5', 'SCP06K0', 'SCP06N33', 'SCP05D0',
                    'SCP05D6', 'SCP05P9', 'SCP06R12', 'SCP06Z5'],
        'z': [1.192, 1.092, 0.980, 1.110, 1.350,
              0.850, 1.231, 1.415, 1.188, 1.014,
              1.315, 0.821, 1.212, 0.623],
        'mB': [25.497, 25.636, 24.613, 25.253, 25.424,
               24.345, 25.389, 25.811, 25.407, 25.201,
               25.660, 24.367, 25.789, 23.482],
        'mB_err': [0.048, 0.066, 0.028, 0.068, 0.052,
                   0.038, 0.111, 0.087, 0.132, 0.066,
                   0.046, 0.049, 0.114, 0.144],
        'x1': [-1.45, -2.66, -0.35, -2.09, 0.15,
               0.58, -3.12, 0.30, -2.15, -0.61,
               -1.26, 0.25, -2.06, -0.76],
        'x1_err': [0.68, 0.65, 0.33, 1.29, 0.64,
                   0.31, 1.10, 0.97, 1.32, 0.65,
                   0.56, 0.50, 1.50, 0.88],
        'c': [0.065, 0.257, 0.014, -0.133, -0.029,
              0.089, -0.103, 0.147, -0.038, 0.061,
              -0.058, 0.022, -0.158, 0.070],
        'c_err': [0.084, 0.083, 0.053, 0.142, 0.052,
                  0.067, 0.187, 0.081, 0.175, 0.085,
                  0.061, 0.075, 0.198, 0.120],
        'Host_Mass': [0.44, 1.97, 0.00, 0.00, 1.72,
                      0.00, 3.66, 2.30, 0.00, 0.40,
                      2.61, 0.00, 0.23, 0.00],
        'Lens_Factor': [1.000, 1.030, 1.000, 1.000, 1.015,
                        1.000, 1.000, 1.000, 1.066, 1.000,
                        1.021, 1.000, 1.000, 1.000]
    }
    
    df = pd.DataFrame(data)
    
    # Cosmological parameters used in the paper for calibration
    # These are the global best-fit values from Table 6
    alpha = 0.121   # Light-curve shape correction coefficient
    beta = 2.47     # Color correction coefficient
    delta = -0.032  # Host-mass correction coefficient
    MB = -19.321    # Absolute B-band magnitude (from Table 6)
    
    # Calculate distance moduli using the SALT2 standardization
    z_vals = []
    mu_vals = []
    mu_err_vals = []
    
    for idx, row in df.iterrows():
        # Host mass correction probability
        # For cluster SNe, the paper uses precise mass measurements
        host_mass = row['Host_Mass']
        if host_mass > 0.5:
            P_mass = 0.13  # Massive hosts (typical for cluster ellipticals)
        else:
            P_mass = 0.55  # Unknown or low-mass hosts
        
        # Correct for gravitational lensing if applicable
        lens_factor = row['Lens_Factor']
        if lens_factor > 1.0:
            lens_correction = -2.5 * np.log10(lens_factor)
        else:
            lens_correction = 0.0
        
        # Calculate distance modulus
        mu = (row['mB'] + 
              alpha * row['x1'] - 
              beta * row['c'] + 
              delta * P_mass - 
              MB + 
              lens_correction)
        
        # Estimate distance modulus error
        sigma_mB = row['mB_err']
        sigma_x1 = row['x1_err']
        sigma_c = row['c_err']
        
        # Propagation of errors (simplified, ignoring correlations)
        sigma_mu = np.sqrt(sigma_mB**2 + (alpha * sigma_x1)**2 + (beta * sigma_c)**2)
        
        # Add a small systematic term (typical ~0.02 mag)
        sigma_mu = np.sqrt(sigma_mu**2 + 0.02**2)
        
        z_vals.append(row['z'])
        mu_vals.append(mu)
        mu_err_vals.append(sigma_mu)
    
    z_vals = np.array(z_vals)
    mu_vals = np.array(mu_vals)
    mu_err_vals = np.array(mu_err_vals)
    
    print(f"Loaded {len(z_vals)} SNe from Suzuki et al. 2012")
    print(f"Redshift range: {z_vals.min():.4f} to {z_vals.max():.4f}")
    
    return z_vals, mu_vals, mu_err_vals, df

# =============================================================================
# 2. COSMOLOGICAL MODEL & STATISTICS
# =============================================================================

# Cache redshift grid and precompute (1+z)^3 once
_z_cache = {}

def _get_z_cache(z, z_grid_points):
    """Return cached z_grid, (1+z_grid)^3, and (1+z_obs)."""
    z = np.atleast_1d(z)
    key = (float(z.max()), int(z_grid_points), z.shape)
    if key not in _z_cache:
        z_grid = np.linspace(1e-8, z.max(), z_grid_points)
        zp1_cubed = (1.0 + z_grid) ** 3
        one_plus_z = 1.0 + z
        _z_cache[key] = (z_grid, zp1_cubed, one_plus_z)
    return _z_cache[key]


def mu_model(z, Om_m0, H_0, z_grid_points=2000):
    """
    Theoretical distance modulus for a flat Lambda-CDM model.
    """
    z = np.atleast_1d(z)
    z_grid, zp1_cubed, one_plus_z = _get_z_cache(z, z_grid_points)
    
    integrand = C_LIGHT / (np.sqrt(Om_m0 * zp1_cubed + (1.0 - Om_m0)) * H_0)
    cum_integral = np.concatenate(([0.0], cumulative_trapezoid(integrand, z_grid)))
    Dc = np.interp(z, z_grid, cum_integral)   # comoving distance [Mpc]
    dL = one_plus_z * Dc                       # luminosity distance [Mpc]
    
    return 5.0 * np.log10(dL) + 25.0


def mu_model_batch(z, Om_arr, H0_arr, z_grid_points=2000, max_chunk=5000):
    """
    Vectorized distance modulus for MANY (Om, H0) pairs at once.
    """
    Om_arr = np.atleast_1d(np.asarray(Om_arr, dtype=float))
    H0_arr = np.atleast_1d(np.asarray(H0_arr, dtype=float))
    z = np.atleast_1d(z)
    
    K = Om_arr.shape[0]
    if K == 0:
        return np.empty((0, z.shape[0]))
    
    z_grid, zp1_cubed, one_plus_z = _get_z_cache(z, z_grid_points)
    N = z.shape[0]
    result = np.empty((K, N))
    
    # Precompute interpolation indices & weights
    idx = np.clip(np.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 2)
    x0, x1 = z_grid[idx], z_grid[idx + 1]
    frac = (z - x0) / (x1 - x0)
    
    # Process the full batch in memory-safe chunks
    for start in range(0, K, max_chunk):
        end = min(start + max_chunk, K)
        Om_c = Om_arr[start:end]
        H0_c = H0_arr[start:end]
        Kc = end - start
        
        # E(z) for every (Om, H0) pair at once
        Ez = np.sqrt(Om_c[:, None] * zp1_cubed[None, :] + (1.0 - Om_c[:, None]))
        Hz = Ez * H0_c[:, None]
        integrand = C_LIGHT / Hz
        
        # Cumulative trapezoidal integral along the grid axis
        cum = np.concatenate(
            [np.zeros((Kc, 1)),
             cumulative_trapezoid(integrand, z_grid, axis=1)],
            axis=1
        )
        
        # Batched linear interpolation
        y0, y1 = cum[:, idx], cum[:, idx + 1]
        Dc = y0 + frac[None, :] * (y1 - y0)
        result[start:end] = one_plus_z[None, :] * Dc
    
    return 5.0 * np.log10(result) + 25.0


def calc_chisq_diag(pars, z_vals, mu_vals, inv_var):
    """Chi-squared using diagonal errors only."""
    Om_m0, H_0 = pars
    Om_m0 = np.atleast_1d(Om_m0)
    H_0 = np.atleast_1d(H_0)
    model = mu_model_batch(z_vals, Om_m0, H_0)
    dmu = mu_vals[None, :] - model
    return np.sum(dmu ** 2 * inv_var[None, :], axis=1)


def calc_chisq_cov(theta, z_vals, mu_vals, cov_inv):
    """Chi-squared using diagonal errors (no covariance matrix for small sample)."""
    Om_m0, H_0 = theta
    model = mu_model(z_vals, Om_m0, H_0)
    dmu = mu_vals - model
    return float(dmu @ cov_inv @ dmu)

# =============================================================================
# 3. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    setup_matplotlib()
    print(f"Results will be saved to: {OUTPUT_DIR}\n")
    
    # --- Step 1: Load Suzuki et al. 2012 SN Data ---
    print("--- Loading Suzuki et al. 2012 SN Data ---")
    z_vals, mu_vals, mu_err, df = load_suzuki2012_data()
    
    # Use diagonal errors (no covariance matrix for this small sample)
    cov_inv = np.diag(1.0 / mu_err ** 2)
    inv_var = 1.0 / mu_err ** 2
    
    # --- Step 2: Frequentist Optimization (Curve Fitting) ---
    print("\n--- Optimization Results ---")
    p0 = [0.3, 70.0]
    bounds = ([0.0, 50.0], [1.0, 100.0])
    
    popt, pcov = opt.curve_fit(
        mu_model, z_vals, mu_vals,
        p0=p0, sigma=mu_err, absolute_sigma=True, bounds=bounds
    )
    
    best_Om, best_H0 = popt
    Om_err, H0_err = np.sqrt(np.diag(pcov))
    min_chisq = calc_chisq_cov(popt, z_vals, mu_vals, cov_inv)
    dof = len(z_vals) - len(popt)
    
    print(f"Omega_m       = {best_Om:.4f} +/- {Om_err:.4f}")
    print(f"Omega_Lambda  = {1 - best_Om:.4f} +/- {Om_err:.4f}")
    print(f"H_0           = {best_H0:.4f} +/- {H0_err:.4f}")
    print(f"chi^2_reduced = {min_chisq:.2f}/{dof} = {min_chisq/dof:.3f}\n")
    
    # --- Step 3: Chi-Squared Grid & Contour Plotting ---
    print("--- Generating Chi-Squared Maps ---")
    sample_rate = 100
    Om_space = np.linspace(0.0, 1.0, sample_rate)
    H0_space = np.linspace(50, 90, sample_rate)
    xx, yy = np.meshgrid(Om_space, H0_space)
    
    # Fast diagonal path
    Z = calc_chisq_diag([xx.ravel(), yy.ravel()], z_vals, mu_vals, inv_var).reshape(xx.shape)
    
    # Convert absolute chi-squared values into Delta-chi^2 values
    delta_chisq = Z - Z.min()
    confidence_levels = [2.30, 6.18, 11.83]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx, yy, delta_chisq, levels=[0] + confidence_levels,
                     cmap='viridis_r', extend='max')
    cs_lines = ax.contour(xx, yy, delta_chisq, levels=confidence_levels,
                          colors='white', linewidths=1)
    
    ax.clabel(cs_lines, inline=True, fontsize=10,
              fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'})
    ax.plot(best_Om, best_H0, 'r*', markersize=15, label='Best fit')
    
    ax.set_xlabel(r'$\Omega_{m,0}$')
    ax.set_ylabel(r'$H_0$')
    ax.set_title(r'$\Delta\chi^2$ Confidence Contours (Suzuki et al. 2012)')
    ax.legend()
    fig.colorbar(cf, ax=ax, label=r'$\Delta\chi^2$')
    
    plt.savefig(os.path.join(OUTPUT_DIR, "DeltaChi2_Contour_Suzuki2012.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    # --- Step 4: Bayesian MCMC Sampling ---
    print("\n--- Running MCMC Sampling ---")
    
    def log_prob_vec(theta_batch, z_vals, mu_vals, cov_inv):
        """Vectorized log-posterior for the whole ensemble at once."""
        Om = theta_batch[:, 0]
        H0 = theta_batch[:, 1]
        
        # Flat prior over a sensible box
        in_prior = (Om > 0.0) & (Om < 1.0) & (H0 > 40.0) & (H0 < 100.0)
        lp = np.where(in_prior, 0.0, -np.inf)
        
        chi2 = np.full(theta_batch.shape[0], np.inf)
        if np.any(in_prior):
            model = mu_model_batch(z_vals, Om[in_prior], H0[in_prior])
            dmu = mu_vals[None, :] - model
            dmu_cov = dmu @ cov_inv
            chi2[in_prior] = np.sum(dmu_cov * dmu, axis=1)
        
        return lp - 0.5 * chi2
    
    ndim, nwalkers, nsteps = 2, 32, 5000
    pos = popt + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([1, 10])
    
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_vec,
        args=(z_vals, mu_vals, cov_inv),
        vectorize=True
    )
    sampler.run_mcmc(pos, nsteps, progress=True)
    
    # Use autocorrelation time to choose discard/thin
    try:
        tau = sampler.get_autocorr_time()
        discard = int(2 * np.max(tau))
        thin = max(1, int(0.5 * np.min(tau)))
        print(f"  Autocorrelation time: {tau}")
        print(f"  Using discard={discard}, thin={thin}")
    except emcee.autocorr.AutocorrError as e:
        print(f"  Autocorr warning: {e}")
        discard, thin = 500, 15
    
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    
    Om_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    H0_mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])
    
    print(f"\nMCMC Omega_m = {Om_mcmc[1]:.4f} "
          f"(+{Om_mcmc[2]-Om_mcmc[1]:.4f} / -{Om_mcmc[1]-Om_mcmc[0]:.4f})")
    print(f"MCMC H_0     = {H0_mcmc[1]:.2f} "
          f"(+{H0_mcmc[2]-H0_mcmc[1]:.2f} / -{H0_mcmc[1]-H0_mcmc[0]:.2f})\n")
    
    fig_corner = corner.corner(
        flat_samples,
        labels=[r"$\Omega_{m,0}$", r"$H_0$"],
        truths=[best_Om, best_H0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    plt.savefig(os.path.join(OUTPUT_DIR, "MCMC_H0_vs_Omega_m_Suzuki2012.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_corner)
    
    # --- Step 5: Hubble Diagram ---
    print("\n--- Generating Hubble Diagram ---")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the data points with error bars
    ax.errorbar(z_vals, mu_vals, yerr=mu_err, fmt='o', 
                color='navy', capsize=3, markersize=6, label='Suzuki et al. 2012')
    
    # Plot the best-fit model
    z_model = np.linspace(0.1, 1.5, 100)
    mu_model_best = mu_model(z_model, best_Om, best_H0)
    ax.plot(z_model, mu_model_best, 'r-', linewidth=2, label='Best-fit LCDM')
    
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Distance Modulus $\mu$')
    ax.set_title(r'Hubble Diagram: Suzuki et al. 2012 HST Cluster SNe Ia')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, "Hubble_Diagram_Suzuki2012.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    # --- Step 6: Export Results ---
    print("\n--- Exporting Results ---")
    with open(os.path.join(OUTPUT_DIR, "lcdm_fit_results_Suzuki2012.txt"), "w") as f:
        f.write("# LCDM Fit Results (Suzuki et al. 2012 HST Cluster SNe Ia)\n")
        f.write("# =========================================================\n")
        f.write(f"N_SNe used    = {len(z_vals)}\n")
        f.write(f"Redshift range = {z_vals.min():.4f} to {z_vals.max():.4f}\n")
        f.write(f"Omega_m       = {best_Om:.6f} +/- {Om_err:.6f}\n")
        f.write(f"Omega_Lambda  = {1 - best_Om:.6f} +/- {Om_err:.6f}\n")
        f.write(f"H_0           = {best_H0:.6f} +/- {H0_err:.6f}\n")
        f.write(f"chi^2         = {min_chisq:.6f}\n")
        f.write(f"dof           = {dof}\n")
        f.write(f"chi^2_reduced = {min_chisq/dof:.6f}\n")
        f.write(f"MCMC Omega_m  = {Om_mcmc[1]:.6f} "
                f"(+{Om_mcmc[2]-Om_mcmc[1]:.6f} / -{Om_mcmc[1]-Om_mcmc[0]:.6f})\n")
        f.write(f"MCMC H_0      = {H0_mcmc[1]:.6f} "
                f"(+{H0_mcmc[2]-H0_mcmc[1]:.6f} / -{H0_mcmc[1]-H0_mcmc[0]:.6f})\n")
    
    print(f"  Results exported to: {os.path.join(OUTPUT_DIR, 'lcdm_fit_results_Suzuki2012.txt')}")
    
    # Save the extracted data
    data_df = pd.DataFrame({
        'SN_Name': df['SN_Name'],
        'z': z_vals,
        'mu': mu_vals,
        'mu_err': mu_err
    })
    data_df.to_csv(os.path.join(OUTPUT_DIR, "suzuki2012_extracted_data.csv"), index=False)
    print(f"  Extracted data saved to: {os.path.join(OUTPUT_DIR, 'suzuki2012_extracted_data.csv')}")
    
    # Save contour data
    contour_data = {'X': xx, 'Y': yy, 'delta_chi2': delta_chisq}
    np.save(os.path.join(OUTPUT_DIR, 'contour_H0_Om_Suzuki2012.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(OUTPUT_DIR, 'contour_H0_Om_Suzuki2012.npy')}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"""
    Sample: Suzuki et al. 2012 HST Cluster Supernova Survey
    Number of SNe: {len(z_vals)}
    Redshift range: {z_vals.min():.3f} to {z_vals.max():.3f}
    
    Best-fit LCDM parameters:
    Omega_m       = {best_Om:.4f} +/- {Om_err:.4f}
    Omega_Lambda  = {1 - best_Om:.4f} +/- {Om_err:.4f}
    H_0           = {best_H0:.2f} +/- {H0_err:.2f} km/s/Mpc
    chi^2/dof     = {min_chisq:.2f}/{dof} = {min_chisq/dof:.3f}
    
    MCMC constraints (68% credible interval):
    Omega_m       = {Om_mcmc[1]:.4f} (+{Om_mcmc[2]-Om_mcmc[1]:.4f}/-{Om_mcmc[1]-Om_mcmc[0]:.4f})
    H_0           = {H0_mcmc[1]:.2f} (+{H0_mcmc[2]-H0_mcmc[1]:.2f}/-{H0_mcmc[1]-H0_mcmc[0]:.2f})
    """)

if __name__ == "__main__":
    main()