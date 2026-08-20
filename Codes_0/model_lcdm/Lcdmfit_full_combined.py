"""
lcdm_joint_fit.py
=================
Joint flat Lambda-CDM fitting pipeline combining:
  1. Cosmic Chronometers / H(z) expansion rate data
  2. Pantheon+ Type Ia Supernovae distance modulus data

Outputs:
  - Frequentist joint optimization & Delta-chi^2 contour map
  - MCMC parameter posterior sampling (emcee) with Gelman-Rubin diagnostic (R-hat)
  - MCMC corner plot
  - Joint data fit visualization (H(z) and Pantheon+ mu(z))
  - Hubble tension comparison plot (Planck vs SH0ES vs Joint Fit)
  - Text summary export of fit results
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import emcee
import corner
from matplotlib import rc
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# =============================================================================
# 0. CONFIGURATION & DATA LOADING
# =============================================================================
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/'

# H(z) data files
Z_H_FILE = 'co_z_vals.txt'
H_FILE = 'co_H_vals.txt'
SIGMA_H_FILE = 'co_sigma_vals.txt'

# Pantheon+ SN data files
SN_DATA_FILES = ['Pantheon+SH0ES.dat', 'pantheon_shoes.dat', 'Pantheon+.dat', 'Pantheon+SH0ES.txt']
SN_COV_FILE = 'Pantheon+SH0ES_STAT+SYS.cov'

C_LIGHT = 299792.458  # km/s

def setup_matplotlib():
    """Enables LaTeX rendering if available."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
    except Exception as e:
        print(f"Warning: LaTeX rendering disabled. Error: {e}")

def find_file_recursively(filename, data_dir):
    """Recursively searches for a file in data_dir and subdirectories."""
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath
    for root, dirs, files in os.walk(data_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not locate '{filename}' in '{data_dir}'.")

def load_hz_data(data_dir):
    """Loads z, H(z), and sigma_H datasets."""
    z_path = find_file_recursively(Z_H_FILE, data_dir)
    h_path = find_file_recursively(H_FILE, data_dir)
    sig_path = find_file_recursively(SIGMA_H_FILE, data_dir)

    z_vals = np.loadtxt(z_path)
    H_vals = np.loadtxt(h_path)
    sigma_vals = np.loadtxt(sig_path)

    print(f"Loaded {len(z_vals)} H(z) data points.")
    return z_vals, H_vals, sigma_vals

def load_sn_data(data_dir):
    """Loads Pantheon+ Supernova dataset and optional covariance matrix."""
    sn_path = None
    for pfile in SN_DATA_FILES:
        try:
            sn_path = find_file_recursively(pfile, data_dir)
            break
        except FileNotFoundError:
            pass

    if sn_path is None:
        raise FileNotFoundError("Could not locate Pantheon+ data file.")

    df = pd.read_csv(sn_path, sep=r'\s+', comment="#", engine="python")
    z_col = "zHD" if "zHD" in df.columns else ("z" if "z" in df.columns else df.columns[0])
    err_col = "m_b_corr_err_DIAG" if "m_b_corr_err_DIAG" in df.columns else ("MU_ERR" if "MU_ERR" in df.columns else df.columns[2])

    mask = df[z_col].values > 0
    df = df[mask].copy()

    if "MU_SH0ES" in df.columns:
        mu_vals = df["MU_SH0ES"].values
    elif "MU" in df.columns:
        mu_vals = df["MU"].values
    else:
        mu_vals = df["m_b_corr"].values - (-19.253)

    z_vals = df[z_col].values
    mu_err = df[err_col].values

    # Attempt to load full covariance matrix
    cov_inv = None
    try:
        cov_path = find_file_recursively(SN_COV_FILE, data_dir)
        with open(cov_path, "r") as f:
            n = int(f.readline().strip())
            vals = np.loadtxt(f, dtype=float)
        cov_full = vals.reshape(n, n)
        cov_sub = cov_full[np.ix_(mask, mask)]
        cov_inv = np.linalg.inv(cov_sub)
        print(f"Loaded full Pantheon+ covariance matrix ({len(z_vals)}x{len(z_vals)}).")
    except Exception:
        print("Pantheon+ covariance file not found or load failed. Using diagonal errors.")
        cov_inv = np.diag(1.0 / mu_err**2)

    print(f"Loaded {len(z_vals)} Pantheon+ SNe Ia data points.")
    return z_vals, mu_vals, mu_err, cov_inv

# =============================================================================
# 1. COSMOLOGICAL MODEL FUNCTIONS
# =============================================================================

def H_model(z, Om_m0, H_0):
    """Hubble parameter H(z) for flat Lambda-CDM cosmology."""
    return H_0 * np.sqrt(Om_m0 * (1.0 + z)**3 + (1.0 - Om_m0))

def mu_model(z, Om_m0, H_0, z_eval_grid):
    """Distance modulus mu(z) evaluated via line-of-sight integration."""
    H_grid = H_model(z_eval_grid, Om_m0, H_0)
    integ = cumulative_trapezoid(1.0 / H_grid, z_eval_grid, initial=0.0)
    dL = (1.0 + z_eval_grid) * C_LIGHT * integ
    
    mu_grid = np.zeros_like(z_eval_grid)
    mu_grid[1:] = 5.0 * np.log10(dL[1:]) + 25.0
    mu_grid[0] = mu_grid[1]
    return np.interp(z, z_eval_grid, mu_grid)

# =============================================================================
# 2. STATISTICAL LIKELIHOODS & GELMAN-RUBIN DIAGNOSTIC
# =============================================================================

def calc_chisq_H(theta, z_H, H_data, sigma_H):
    """Chi-squared for H(z) dataset."""
    Om_m0, H_0 = theta
    Hm = H_model(z_H, Om_m0, H_0)
    return np.sum(((H_data - Hm) / sigma_H)**2)

def calc_chisq_sn(theta, z_sn, mu_data, cov_inv, z_eval_grid):
    """Chi-squared for Pantheon+ SNe Ia dataset."""
    Om_m0, H_0 = theta
    mum = mu_model(z_sn, Om_m0, H_0, z_eval_grid)
    dmu = mu_data - mum
    if cov_inv.ndim == 2:
        return float(dmu @ cov_inv @ dmu)
    return float(np.sum(dmu**2 * cov_inv))

def joint_chisq(theta, z_H, H_data, sigma_H, z_sn, mu_data, cov_inv, z_eval_grid):
    """Total joint Chi-squared: chi2_H + chi2_SN."""
    return calc_chisq_H(theta, z_H, H_data, sigma_H) + calc_chisq_sn(theta, z_sn, mu_data, cov_inv, z_eval_grid)

def log_prior(theta):
    """Flat prior on Om_m0 and H0."""
    Om_m0, H_0 = theta
    if 0.0 < Om_m0 < 1.0 and 40.0 < H_0 < 100.0:
        return 0.0
    return -np.inf

def log_prob(theta, z_H, H_data, sigma_H, z_sn, mu_data, cov_inv, z_eval_grid):
    """Joint log-posterior probability."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    c2 = joint_chisq(theta, z_H, H_data, sigma_H, z_sn, mu_data, cov_inv, z_eval_grid)
    return lp - 0.5 * c2

def compute_gelman_rubin(chain_3d):
    """Calculates Gelman-Rubin R-hat statistic for 3D array (N_steps, N_walkers, N_dim)."""
    N, M, D = chain_3d.shape
    if M < 2:
        return np.full(D, np.nan)
    walker_means = np.mean(chain_3d, axis=0)
    grand_mean = np.mean(walker_means, axis=0)
    B = (N / (M - 1.0)) * np.sum((walker_means - grand_mean)**2, axis=0)
    W = np.mean(np.var(chain_3d, axis=0, ddof=1), axis=0)
    var_plus = ((N - 1.0) / N) * W + (1.0 / N) * B
    return np.sqrt(var_plus / W)

# =============================================================================
# 3. MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    setup_matplotlib()

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_lcdm_joint")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}\n")

    # --- Step 1: Load Both Datasets ---
    print("--- Loading Datasets ---")
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = DATA_DIR if os.path.isabs(DATA_DIR) else os.path.join(script_dir, DATA_DIR)

    z_H, H_data, sigma_H = load_hz_data(data_dir)
    z_sn, mu_data, sigma_sn, cov_inv = load_sn_data(data_dir)

    z_max_eval = max(z_H.max(), z_sn.max()) * 1.05
    z_eval_grid = np.concatenate(([0.0], np.geomspace(1e-5, z_max_eval, 350)))

    # --- Step 2: Joint Frequentist Optimization ---
    print("\n--- Joint Frequentist Optimization ---")
    p0 = [0.3, 70.0]
    res = opt.minimize(
        lambda t: joint_chisq(t, z_H, H_data, sigma_H, z_sn, mu_data, cov_inv, z_eval_grid),
        p0,
        bounds=[(0.01, 0.99), (40.0, 100.0)],
        method='Nelder-Mead'
    )

    best_Om, best_H0 = res.x
    min_chisq = res.fun
    dof = (len(z_H) + len(z_sn)) - len(p0)

    print(f"Joint Best Fit: Omega_m = {best_Om:.4f}, H_0 = {best_H0:.4f}")
    print(f"Total Chi^2 = {min_chisq:.2f} / dof ({dof}) = {min_chisq/dof:.3f}")

    # --- Step 3: Delta Chi-Squared Contour Map ---
    print("\n--- Generating Joint Delta-Chi^2 Map ---")
    sample_rate = 80
    Om_space = np.linspace(0.1, 0.5, sample_rate)
    H0_space = np.linspace(60, 80, sample_rate)
    xx, yy = np.meshgrid(Om_space, H0_space)

    Z = np.zeros_like(xx)
    for i in range(sample_rate):
        for j in range(sample_rate):
            theta = [xx[i, j], yy[i, j]]
            Z[i, j] = joint_chisq(theta, z_H, H_data, sigma_H, z_sn, mu_data, cov_inv, z_eval_grid)

    delta_chisq = Z - min_chisq
    confidence_levels = [2.30, 6.18, 11.83]

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx, yy, delta_chisq, levels=[0] + confidence_levels, cmap='viridis_r', extend='max')
    cs_lines = ax.contour(xx, yy, delta_chisq, levels=confidence_levels, colors='white', linewidths=1)
    ax.clabel(cs_lines, inline=True, fontsize=10, fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'})
    ax.plot(best_Om, best_H0, 'r*', markersize=15, label='Joint Best Fit')
    ax.set_xlabel(r'$\Omega_{m,0}$')
    ax.set_ylabel(r'$H_0 \ [\mathrm{km/s/Mpc}]$')
    ax.set_title(r'Joint $\Delta\chi^2$ Confidence Contours [$H(z)$ + Pantheon+]')
    ax.legend()
    fig.colorbar(cf, ax=ax, label=r'$\Delta\chi^2$')
    plt.savefig(os.path.join(output_dir, "Joint_DeltaChi2_Contour.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # --- Step 4: Bayesian MCMC Sampling ---
    print("\n--- Running Joint MCMC Sampling ---")
    ndim, nwalkers, nsteps, burnin = 2, 32, 4000, 1000
    ncpu = max(1, cpu_count() - 1)
    rng = np.random.default_rng(42)

    pos = np.array([best_Om, best_H0]) + 1e-3 * rng.standard_normal((nwalkers, ndim)) * np.array([0.01, 0.5])

    with Pool(processes=ncpu) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob,
            args=(z_H, H_data, sigma_H, z_sn, mu_data, cov_inv, z_eval_grid),
            pool=pool
        )
        sampler.run_mcmc(pos, nsteps, progress=True)

    chain_3d = sampler.get_chain(discard=burnin, flat=False)
    flat_samples = sampler.get_chain(discard=burnin, flat=True)

    r_hat = compute_gelman_rubin(chain_3d)
    Om_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    H0_mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])

    print("\n--- MCMC RESULTS & DIAGNOSTICS ---")
    print(f"Gelman-Rubin R-hat (Om_m0, H0): {r_hat[0]:.4f}, {r_hat[1]:.4f}")
    print(f"MCMC Omega_m = {Om_mcmc[1]:.4f} (+{Om_mcmc[2]-Om_mcmc[1]:.4f} / -{Om_mcmc[1]-Om_mcmc[0]:.4f})")
    print(f"MCMC H_0     = {H0_mcmc[1]:.2f} (+{H0_mcmc[2]-H0_mcmc[1]:.2f} / -{H0_mcmc[1]-H0_mcmc[0]:.2f})\n")

    fig_corner = corner.corner(
        flat_samples,
        labels=[r"$\Omega_{m,0}$", r"$H_0$"],
        truths=[best_Om, best_H0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    fig_corner.suptitle(r"Joint Flat $\Lambda$CDM Posterior Distributions", y=1.02)
    plt.savefig(os.path.join(output_dir, "Joint_MCMC_Corner.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    # --- Step 5: Joint Data Fit Visualization ---
    print("--- Plotting Model Fits vs. Data ---")
    z_plot_grid = np.linspace(1e-3, z_max_eval, 300)

    fig_fits, (ax_h, ax_mu) = plt.subplots(1, 2, figsize=(14, 5))

    # H(z) Panel
    ax_h.errorbar(z_H, H_data, yerr=sigma_H, fmt="o", color="black", ms=3, ecolor="lightgray", capsize=2, label=r"$H(z)$ Data")
    ax_h.plot(z_plot_grid, H_model(z_plot_grid, best_Om, best_H0), color="crimson", lw=2, label=rf"Joint Fit ($H_0={best_H0:.1f}$)")
    ax_h.set_xlabel(r"Redshift $z$")
    ax_h.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax_h.set_title(r"Hubble Parameter $H(z)$ Fit")
    ax_h.legend(frameon=False)
    ax_h.grid(True, alpha=0.3)

    # Pantheon+ Panel
    ax_mu.errorbar(z_sn, mu_data, yerr=sigma_sn, fmt="o", color="gray", ms=1, alpha=0.25, label="Pantheon+ SNe Ia")
    ax_mu.plot(z_plot_grid, mu_model(z_plot_grid, best_Om, best_H0, z_eval_grid), color="crimson", lw=2, label="Joint Fit")
    ax_mu.set_xlabel(r"Redshift $z$")
    ax_mu.set_ylabel(r"Distance Modulus $\mu(z)$")
    ax_mu.set_title(r"Pantheon+ Distance Modulus Fit")
    ax_mu.legend(frameon=False)
    ax_mu.grid(True, alpha=0.3)

    fig_fits.tight_layout()
    plt.savefig(os.path.join(output_dir, "Joint_Data_Fits.png"), dpi=250, bbox_inches='tight')
    plt.close(fig_fits)

    # --- Step 6: Hubble Tension Visualization ---
    literature = {
        "This Work (Joint Fit)": (best_H0, (H0_mcmc[2] - H0_mcmc[0]) / 2.0, 'crimson'),
        "Planck 2018 (CMB)":     (67.4, 0.5, 'steelblue'),
        "SH0ES 2022 (Local)":    (73.04, 1.04, 'darkorange'),
    }

    fig_tens, ax_t = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(literature.items()):
        ax_t.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)

    ax_t.set_yticks(range(len(literature)))
    ax_t.set_yticklabels(literature.keys())
    ax_t.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax_t.set_title(r"Hubble Parameter: Joint Fit vs. $H_0$ Tension")
    ax_t.axvspan(67.4 - 0.5, 67.4 + 0.5, color='steelblue', alpha=0.15)
    ax_t.axvspan(73.04 - 1.04, 73.04 + 1.04, color='darkorange', alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Hubble_Tension_Comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_tens)

    # --- Step 7: Export Numerical Results ---
    print("--- Exporting Summary ---")
    summary_path = os.path.join(output_dir, "lcdm_joint_fit_results.txt")
    with open(summary_path, "w") as f:
        f.write("# Joint Lambda-CDM Fit Results [H(z) + Pantheon+]\n")
        f.write("# ================================================\n")
        f.write(f"N_H(z) points   = {len(z_H)}\n")
        f.write(f"N_Pantheon+     = {len(z_sn)}\n")
        f.write(f"Best-fit Om_m0  = {best_Om:.6f}\n")
        f.write(f"Best-fit H0     = {best_H0:.6f}\n")
        f.write(f"Min Chi^2       = {min_chisq:.6f}\n")
        f.write(f"dof             = {dof}\n")
        f.write(f"Reduced Chi^2   = {min_chisq/dof:.6f}\n")
        f.write(f"R-hat (Om, H0)  = {r_hat[0]:.4f}, {r_hat[1]:.4f}\n")
        f.write(f"MCMC Om_m0      = {Om_mcmc[1]:.6f} (+{Om_mcmc[2]-Om_mcmc[1]:.6f} / -{Om_mcmc[1]-Om_mcmc[0]:.6f})\n")
        f.write(f"MCMC H0         = {H0_mcmc[1]:.6f} (+{H0_mcmc[2]-H0_mcmc[1]:.6f} / -{H0_mcmc[1]-H0_mcmc[0]:.6f})\n")

    print(f"Results exported to: {summary_path}\nAll operations completed successfully.")

if __name__ == "__main__":
    main()