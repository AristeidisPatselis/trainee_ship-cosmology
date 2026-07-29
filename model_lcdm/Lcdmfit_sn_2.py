import os
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import emcee
import corner
from matplotlib import rc

# Import data loader
import data_loader

# =============================================================================
# CONFIG
# =============================================================================
# Point this at the folder where you keep the Pantheon+SH0ES data release
# (from https://github.com/PantheonPlusSH0ES/DataRelease -> Pantheon+_Data/)
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes/Data_Sets/'

# Pantheon+ data release file names - change these if your local copies differ
SN_DATA_FILE = 'Pantheon+SH0ES.dat'                 # main table: z, mu, mu_err, calibrator flag, ...
SN_COV_FILE  = 'Pantheon+SH0ES_STAT+SYS.cov'         # full stat+syst covariance matrix (optional)

# Column names as they appear in the official Pantheon+SH0ES.dat header
Z_COL      = 'zHD'                 # Hubble-diagram redshift (peculiar-velocity corrected)
MU_COL     = 'MU_SH0ES'            # calibrated distance modulus
MU_ERR_COL = 'MU_SH0ES_ERR_DIAG'   # diagonal error on mu (used only if no covariance file is found)
CALIB_COL  = 'IS_CALIBRATOR'       # 1 if SN is a Cepheid-host calibrator, 0 if Hubble-flow SN

# Quality cuts
Z_MIN = 0.01          # drop very-low-z SNe dominated by peculiar velocities
EXCLUDE_CALIBRATORS = True   # exclude Cepheid-calibrator SNe -> pure Hubble-flow cosmology fit

# Speed/accuracy tradeoff: the full 1701x1701 covariance is used for the
# best-fit optimization and the MCMC (where accuracy matters most). For the
# quick-look Delta-chi^2 contour map (10,000 grid evaluations) we fall back
# to the diagonal errors only, since a full covariance solve at every grid
# point would be very slow. Set to True if you don't mind the wait.
USE_FULL_COV_FOR_CONTOUR_MAP = False

C_LIGHT = 299792.458  # km/s

# =============================================================================
# SETUP
# =============================================================================

def setup_matplotlib():
    """Attempts to enable LaTeX formatting for professional plots."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
    except Exception as e:
        print(f"Warning: LaTeX rendering not enabled. Error: {e}")


def find_file_recursively(filename, data_dir):
    """
    Search for a file recursively in data_dir and its subdirectories.
    Returns the full path if found, raises FileNotFoundError otherwise.
    """
    # First check if data_loader has the path
    if filename == 'Pantheon+SH0ES.dat':
        try:
            from data_loader import PANTHEON_PATH
            if os.path.exists(PANTHEON_PATH):
                return PANTHEON_PATH
        except (ImportError, AttributeError):
            pass
    
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath

    for root, dirs, files in os.walk(data_dir):
        if filename in files:
            return os.path.join(root, filename)

    available_files = list_available_files(data_dir)
    raise FileNotFoundError(
        f"Could not find '{filename}' in '{data_dir}' or its subdirectories.\n"
        f"Available files in {data_dir} and subdirectories:\n{available_files}"
    )


def list_available_files(data_dir, max_files=30):
    """List available data files in data_dir and subdirectories."""
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.dat') or f.endswith('.txt') or f.endswith('.cov'):
                rel_path = os.path.relpath(os.path.join(root, f), data_dir)
                files.append(rel_path)
                if len(files) >= max_files:
                    files.append("... and more")
                    return "\n".join(files)
    return "\n".join(files) if files else "No data files found"


# =============================================================================
# 1. DATA LOADING (Pantheon+ Type Ia SNe) - Enhanced with data_loader
# =============================================================================

def load_sn_data(data_dir):
    """
    Loads the Pantheon+SH0ES Hubble diagram: redshift, calibrated distance
    modulus, and diagonal mu error, applying the configured quality cuts.
    Returns z, mu, mu_err (diagonal only) and the boolean mask used, so the
    caller can apply the identical mask to the full covariance matrix.
    
    Enhanced: Now uses data_loader's path management when available.
    """
    # Try to use data_loader's path first
    try:
        from data_loader import PANTHEON_PATH
        if os.path.exists(PANTHEON_PATH):
            filepath = PANTHEON_PATH
            print(f"  Using data_loader path: {filepath}")
        else:
            filepath = find_file_recursively(SN_DATA_FILE, data_dir)
    except (ImportError, AttributeError):
        filepath = find_file_recursively(SN_DATA_FILE, data_dir)
    
    print(f"  Found SN data table: {filepath}")

    df = pd.read_csv(filepath, sep=r"\s+", engine="python")
    
    mask = df[Z_COL].values > Z_MIN
    if EXCLUDE_CALIBRATORS and CALIB_COL in df.columns:
        mask &= (df[CALIB_COL].values == 0)

    z_vals = df[Z_COL].values[mask]
    mu_vals = df[MU_COL].values[mask]
    mu_err = df[MU_ERR_COL].values[mask]

    print(f"  {mask.sum()} / {len(df)} SNe pass cuts "
          f"(z > {Z_MIN}, exclude_calibrators={EXCLUDE_CALIBRATORS})")

    return z_vals, mu_vals, mu_err, mask


def load_sn_covariance(data_dir, mask):
    """
    Loads the full Pantheon+ stat+syst covariance matrix (mag^2) and returns
    the sub-block corresponding to the SNe that survived the quality cuts.
    Format: first line = N (matrix dimension), followed by N*N numbers in
    row-major order, in the SAME row order as the main data table (before
    any cuts).
    Returns None if the covariance file cannot be found, in which case the
    caller should fall back to the diagonal mu errors.
    
    Enhanced: Now uses data_loader's path and Cholesky decomposition.
    """
    # Try to use data_loader's path first
    try:
        from data_loader import COV_MATRIX_PATH
        if os.path.exists(COV_MATRIX_PATH):
            filepath = COV_MATRIX_PATH
            print(f"  Using data_loader path: {filepath}")
        else:
            filepath = find_file_recursively(SN_COV_FILE, data_dir)
    except (ImportError, AttributeError):
        try:
            filepath = find_file_recursively(SN_COV_FILE, data_dir)
        except FileNotFoundError:
            print(f"  Covariance file '{SN_COV_FILE}' not found - "
                  f"falling back to diagonal errors only.")
            return None

    print(f"  Found SN covariance matrix: {filepath}")
    with open(filepath, "r") as f:
        n = int(f.readline().strip())
        vals = np.array([float(x) for x in f.read().split()])

    if vals.size != n * n:
        raise ValueError(
            f"Covariance file size mismatch: expected {n*n} values, got {vals.size}"
        )

    cov_full = vals.reshape(n, n)

    if mask.size != n:
        raise ValueError(
            f"Covariance matrix dimension ({n}) does not match the number of "
            f"rows in the data table ({mask.size}). Make sure the .dat and "
            f".cov files come from the same Pantheon+ release."
        )

    cov_cut = cov_full[np.ix_(mask, mask)]
    return cov_cut


def load_pantheon_data_enhanced(data_dir):
    """
    Enhanced data loader that tries to use data_loader first, then falls back
    to the original implementation. Returns the same format as load_sn_data
    plus covariance matrix.
    """
    try:
        # Try to use data_loader's efficient loading
        z_cmb, mu_obs, C, C_fac, Cinv_ones, scalar_denom = data_loader.load_pantheon_data()
        
        # Apply quality cuts
        mask = z_cmb > Z_MIN
        
        # Note: data_loader doesn't load the calibrator flag, so we can't
        # apply EXCLUDE_CALIBRATORS here. We fall back to original method.
        # This is a limitation we'll need to address in data_loader.py
        print("  Using data_loader for efficient loading (without calibrator cuts)")
        
        # Load calibrator info from original method
        try:
            filepath = find_file_recursively(SN_DATA_FILE, data_dir)
            df = pd.read_csv(filepath, sep=r"\s+", engine="python")
            if EXCLUDE_CALIBRATORS and CALIB_COL in df.columns:
                mask &= (df[CALIB_COL].values == 0)
        except:
            print("  Warning: Could not apply calibrator cuts, using redshift cut only")
        
        z_vals = z_cmb[mask]
        mu_vals = mu_obs[mask]
        mu_err = np.sqrt(np.diag(C))[mask]
        cov = C[np.ix_(mask, mask)]
        
        print(f"  {mask.sum()} / {len(z_cmb)} SNe pass cuts "
              f"(z > {Z_MIN})")
        
        return z_vals, mu_vals, mu_err, mask, cov
        
    except (ImportError, FileNotFoundError, AttributeError) as e:
        print(f"  data_loader not available or failed: {e}")
        print("  Falling back to original loading method...")
        # Fall back to original method
        z_vals, mu_vals, mu_err, mask = load_sn_data(data_dir)
        cov = load_sn_covariance(data_dir, mask)
        return z_vals, mu_vals, mu_err, mask, cov


# =============================================================================
# 2. COSMOLOGICAL MODEL & STATISTICS
# =============================================================================

def H_model(z, Om_m0, H_0):
    """
    Theoretical Hubble parameter for a flat Lambda-CDM model.
    Formula: H(z) = H0 * sqrt(Omega_m * (1+z)^3 + (1 - Omega_m))
    """
    E_z = np.sqrt(Om_m0 * (1 + z) ** 3 + (1 - Om_m0))
    return E_z * H_0


def mu_model(z, Om_m0, H_0, z_grid_points=2000):
    """
    Theoretical distance modulus for a flat Lambda-CDM model.
    mu(z) = 5*log10(d_L(z) / 10 pc)
    d_L(z) = (1+z) * c * Integral_0^z dz' / H(z')

    The line-of-sight comoving distance is obtained once on a fine redshift
    grid via cumulative trapezoidal integration and then interpolated at the
    observed SN redshifts - the same "solve once on a grid, interpolate"
    pattern used in the ODE-based H(z) ROMS solvers elsewhere in this thesis
    codebase, just applied to the distance integral instead.
    """
    z = np.atleast_1d(z)
    z_grid = np.linspace(1e-8, z.max(), z_grid_points)

    integrand = C_LIGHT / H_model(z_grid, Om_m0, H_0)
    cum_integral = np.concatenate(([0.0], cumulative_trapezoid(integrand, z_grid)))

    Dc = np.interp(z, z_grid, cum_integral)   # comoving distance [Mpc]
    dL = (1.0 + z) * Dc                       # luminosity distance [Mpc]

    return 5.0 * np.log10(dL) + 25.0          # +25 converts Mpc -> 10 pc units


def calc_chisq_diag(pars, z_vals, mu_vals, mu_err):
    """
    Chi-squared using diagonal errors only. Vectorized over a 1D array of
    (Om, H0) pairs (used for the fast contour map).
    """
    Om_m0, H_0 = pars
    Om_m0 = np.atleast_1d(Om_m0)
    H_0 = np.atleast_1d(H_0)

    chi2 = np.empty(Om_m0.shape)
    for i in range(Om_m0.size):
        model = mu_model(z_vals, Om_m0[i], H_0[i])
        chi2[i] = np.sum((mu_vals - model) ** 2 / mu_err ** 2)
    return chi2


def calc_chisq_cov(theta, z_vals, mu_vals, cov_inv):
    """
    Chi-squared using the full covariance matrix: chi2 = dmu^T Cinv dmu.
    Used for the best-fit optimization and the MCMC likelihood, where the
    off-diagonal SN systematics genuinely matter.
    """
    Om_m0, H_0 = theta
    model = mu_model(z_vals, Om_m0, H_0)
    dmu = mu_vals - model
    return dmu @ cov_inv @ dmu


# =============================================================================
# 3. MAIN EXECUTION BLOCK
# =============================================================================

# -----------------------------------------------------------------------------
# Main program: orchestrates the complete cosmological analysis pipeline.
# It loads the data, performs the fit, explores the posterior with MCMC,
# generates figures, and exports the numerical results.
# -----------------------------------------------------------------------------
def main():
    setup_matplotlib()

    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_sn")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}\n")

    # --- Step 1: Load Pantheon+ Type Ia SN data ---
    print("--- Loading Pantheon+ Type Ia SN Data ---")

    if not os.path.isabs(DATA_DIR):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        data_dir = os.path.join(script_dir, DATA_DIR)
    else:
        data_dir = DATA_DIR

    if not os.path.exists(data_dir):
        print(f"Critical Error: Data directory not found: {data_dir}")
        return

    try:
        # Try enhanced loading with data_loader
        z_vals, mu_vals, mu_err, mask, cov = load_pantheon_data_enhanced(data_dir)
        print(f"\nSuccessfully loaded {len(z_vals)} SNe.")
        print(f"Redshift range: {z_vals.min():.4f} to {z_vals.max():.4f}\n")
    except FileNotFoundError as e:
        print(f"Critical Error: Data files missing. {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return

    # Decide whether to use the full covariance matrix or only diagonal errors.
    # The full covariance accounts for correlated systematic uncertainties.
    use_full_cov = cov is not None
    if use_full_cov:
        cov_inv = np.linalg.inv(cov)
        print(f"  Using full covariance matrix ({cov.shape[0]}x{cov.shape[1]})")
    else:
        cov_inv = np.diag(1.0 / mu_err ** 2)
        print("  Using diagonal errors only")

    # --- Step 2: Frequentist Optimization (Curve Fitting) ---
    print("--- Optimization Results ---")

    # Initial parameter guess supplied to the optimizer.
    # These values are close to the concordance LCDM cosmology.
    p0 = [0.3, 70.0]
    bounds = ([0.0, 50.0], [1.0, 100.0])

    if use_full_cov:
        popt, pcov = opt.curve_fit(
            mu_model, z_vals, mu_vals,
            p0=p0, sigma=cov, absolute_sigma=True, bounds=bounds
        )
    else:
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
    H0_space = np.linspace(55, 80, sample_rate)

    xx, yy = np.meshgrid(Om_space, H0_space)

    if USE_FULL_COV_FOR_CONTOUR_MAP and use_full_cov:
        print("  Using full covariance for contour map (may be slow)...")
        Z = np.empty(xx.shape)
        for idx in np.ndindex(xx.shape):
            Z[idx] = calc_chisq_cov((xx[idx], yy[idx]), z_vals, mu_vals, cov_inv)
    else:
        print("  Using diagonal errors for contour map (faster)...")
        result = calc_chisq_diag([xx.ravel(), yy.ravel()], z_vals, mu_vals, mu_err)
        Z = result.reshape(xx.shape)

    # Convert absolute chi-squared values into Δχ² values.
    # Confidence regions depend only on the difference from the minimum.
    delta_chisq = Z - Z.min()

    confidence_levels = [2.30, 6.18, 11.83]

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx, yy, delta_chisq, levels=[0] + confidence_levels, cmap='viridis_r', extend='max')
    cs_lines = ax.contour(xx, yy, delta_chisq, levels=confidence_levels, colors='white', linewidths=1)

    ax.clabel(cs_lines, inline=True, fontsize=10, fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'})
    ax.plot(best_Om, best_H0, 'r*', markersize=15, label='Best fit')
    ax.set_xlim(0.2,0.5)
    ax.set_ylim(70,75)
    ax.set_xlabel(r'$\Omega_{m,0}$')
    ax.set_ylabel(r'$H_0$')
    ax.set_title(r'$\Delta\chi^2$ Confidence Contours (Pantheon+ SNe Ia)')
    ax.legend()
    fig.colorbar(cf, ax=ax, label=r'$\Delta\chi^2$')

    plt.savefig(os.path.join(output_dir, "DeltaChi2_Contour_SN.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # --- Step 4: Bayesian MCMC Sampling ---
    print("\n--- Running MCMC Sampling ---")

    def log_prior(theta):
        Om, H0 = theta
        if 0.0 < Om < 1.0 and 40.0 < H0 < 100.0:
            return 0.0
        return -np.inf

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        chi2 = calc_chisq_cov(theta, z_vals, mu_vals, cov_inv)
        return lp - 0.5 * chi2

    ndim, nwalkers, nsteps = 2, 32, 3000

    pos = popt + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([1, 10])

    # Initialize the affine-invariant ensemble sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Remove burn-in, thin the chain, and flatten all walkers into one sample.
    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)

    Om_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    H0_mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])

    print(f"\nMCMC Omega_m = {Om_mcmc[1]:.4f} (+{Om_mcmc[2]-Om_mcmc[1]:.4f} / -{Om_mcmc[1]-Om_mcmc[0]:.4f})")
    print(f"MCMC H_0     = {H0_mcmc[1]:.2f} (+{H0_mcmc[2]-H0_mcmc[1]:.2f} / -{H0_mcmc[1]-H0_mcmc[0]:.2f})\n")

    fig_corner = corner.corner(
        flat_samples,
        labels=[r"$\Omega_{m,0}$", r"$H_0$"],
        truths=[best_Om, best_H0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )

    plt.savefig(os.path.join(output_dir, "MCMC_H0_vs_Omega_m_SN.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_corner)

    # --- Step 5: Hubble Tension Visualization ---
    # Reference H0 measurements used for comparison with this work.
    literature = {
        "This Work (Pantheon+ SNe Ia)": (best_H0, H0_err, 'crimson'),
        "Planck 2018 (CMB)":            (67.4,  0.5,  'steelblue'),
        "SH0ES 2022 (Local)":           (73.04, 1.04, 'darkorange'),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(literature.items()):
        ax.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)

    ax.set_yticks(range(len(literature)))
    ax.set_yticklabels(literature.keys())
    ax.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax.set_title(r"Hubble Parameter: Current Fit vs. The $H_0$ Tension")

    ax.axvspan(67.4 - 0.5, 67.4 + 0.5, color='steelblue', alpha=0.15)
    ax.axvspan(73.04 - 1.04, 73.04 + 1.04, color='darkorange', alpha=0.15)

    # Calculate tension with Planck
    tension_sigma = abs(best_H0 - 67.4) / np.sqrt(H0_err**2 + 0.5**2)
    ax.text(0.02, 0.98, f'Tension with Planck: {tension_sigma:.1f}σ',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Hubble_Parameter_SN.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # --- Step 6: Export Results ---
    print("\n--- Exporting Results ---")

    with open(os.path.join(output_dir, "lcdm_fit_results_SN.txt"), "w") as f:
        f.write("# LCDM Fit Results (Pantheon+ Type Ia SNe)\n")
        f.write("# =========================================\n")
        f.write(f"# Data source: {'data_loader' if use_full_cov and 'data_loader' in str(globals()) else 'local'}\n")
        f.write(f"N_SNe used    = {len(z_vals)}\n")
        f.write(f"Used full covariance matrix = {use_full_cov}\n")
        f.write(f"Omega_m       = {best_Om:.6f} +/- {Om_err:.6f}\n")
        f.write(f"Omega_Lambda  = {1 - best_Om:.6f} +/- {Om_err:.6f}\n")
        f.write(f"H_0           = {best_H0:.6f} +/- {H0_err:.6f}\n")
        f.write(f"chi^2         = {min_chisq:.6f}\n")
        f.write(f"dof           = {dof}\n")
        f.write(f"chi^2_reduced = {min_chisq/dof:.6f}\n")
        f.write(f"MCMC Omega_m  = {Om_mcmc[1]:.6f} (+{Om_mcmc[2]-Om_mcmc[1]:.6f} / -{Om_mcmc[1]-Om_mcmc[0]:.6f})\n")
        f.write(f"MCMC H_0      = {H0_mcmc[1]:.6f} (+{H0_mcmc[2]-H0_mcmc[1]:.6f} / -{H0_mcmc[1]-H0_mcmc[0]:.6f})\n")

    print(f"  Results exported to: {os.path.join(output_dir, 'lcdm_fit_results_SN.txt')}")
    print(f"  Plots saved in: {output_dir}")

    contour_data = {'X': xx, 'Y': yy, 'delta_chi2': delta_chisq}
    np.save(os.path.join(output_dir, 'contour_H0_Om_lcdm_SN.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(output_dir, 'contour_H0_Om_lcdm_SN.npy')}")


if __name__ == "__main__":
    main()