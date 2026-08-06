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
# Point this at the folder where you keep the Pantheon+SH0ES data release
# (from https://github.com/PantheonPlusSH0ES/DataRelease -> Pantheon+_Data/)
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/'

# Pantheon+ data release file names - change these if your local copies differ
SN_DATA_FILE = 'Pantheon+SH0ES.dat'                 # main table: z, mu, mu_err, calibrator flag, ...
SN_COV_FILE  = 'Pantheon+SH0ES_STAT+SYS.cov'         # full stat+syst covariance matrix (optional)

# Column names as they appear in the official Pantheon+SH0ES.dat header
Z_COL      = 'zHD'                 # Hubble-diagram redshift (peculiar-velocity corrected)
MU_COL     = 'MU_SH0ES'            # calibrated distance modulus
MU_ERR_COL = 'MU_SH0ES_ERR_DIAG'   # diagonal error on mu (used only if no covariance file is found)
CALIB_COL  = 'IS_CALIBRATOR'       # 1 if SN is a Cepheid-host calibrator, 0 if Hubble-flow SN

# --- NO DATA POINTS ARE EXCLUDED ---
# All SNe are kept in the fit, including low-z objects and Cepheid calibrators.
# This means the sample will contain both Hubble-flow SNe and the Cepheid-
# anchored calibrators used by SH0ES.  The fit therefore solves for the
# absolute H0 scale simultaneously with the cosmological parameters.
Z_MIN = -1.0                       # negative -> every redshift passes
EXCLUDE_CALIBRATORS = False        # keep Cepheid-host calibrators in the sample

# Speed/accuracy tradeoff: the full 1701x1701 covariance is used for the
# best-fit optimization and the MCMC (where accuracy matters most). For the
# quick-look Delta-chi^2 contour map (10,000 grid evaluations) we fall back
# to the diagonal errors only, since a full covariance solve at every grid
# point would be very slow. Set to True if you don't mind the wait.
USE_FULL_COV_FOR_CONTOUR_MAP = False

C_LIGHT = 299792.458  # km/s

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


def find_file_recursively(filename, data_dir):
    """
    Search for a file recursively in data_dir and its subdirectories.
    Returns the full path if found, raises FileNotFoundError otherwise.
    """
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath
    for root, dirs, files in os.walk(data_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not find '{filename}' in '{data_dir}'")


# =============================================================================
# 1. DATA LOADING (Pantheon+ Type Ia SNe)
# =============================================================================

def load_sn_data(data_dir):
    """
    Loads the Pantheon+SH0ES Hubble diagram: redshift, calibrated distance
    modulus, and diagonal mu error, applying the configured quality cuts.
    Returns z, mu, mu_err (diagonal only) and the boolean mask used, so the
    caller can apply the identical mask to the full covariance matrix.
    """
    filepath = find_file_recursively(SN_DATA_FILE, data_dir)
    print(f"  Found SN data table: {filepath}")
    df = pd.read_csv(filepath, sep=r"\s+", engine="python")
    
    # Build the quality mask: redshift cut + optional calibrator exclusion
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

    OPTIMIZATION: np.loadtxt parses the remaining numbers in C, which is
    faster than a Python list comprehension over f.read().split().
    """
    try:
        filepath = find_file_recursively(SN_COV_FILE, data_dir)
    except FileNotFoundError:
        print(f"  Covariance file not found - falling back to diagonal errors.")
        return None

    print(f"  Found SN covariance matrix: {filepath}")
    with open(filepath, "r") as f:
        n = int(f.readline().strip())
        vals = np.loadtxt(f, dtype=float)

    if vals.size != n * n:
        raise ValueError(f"Covariance file size mismatch: expected {n*n}, got {vals.size}")

    cov_full = vals.reshape(n, n)
    if mask.size != n:
        raise ValueError(f"Covariance dimension ({n}) does not match data rows ({mask.size})")
    
    # Slice the full covariance to keep only the SNe that passed the cuts.
    # np.ix_(mask, mask) builds the 2D index array needed for fancy indexing.
    return cov_full[np.ix_(mask, mask)]


# =============================================================================
# 2. COSMOLOGICAL MODEL & STATISTICS
# =============================================================================

# OPTIMIZATION: Cache redshift grid and precompute (1+z)^3 once.
# The grid depends only on max(z) and resolution, not on (Om, H0).
# This removes redundant np.linspace and power operations inside hot loops.
_z_cache = {}

def _get_z_cache(z, z_grid_points):
    """
    Return cached z_grid, (1+z_grid)^3, and (1+z_obs).
    z_grid and (1+z_grid)^3 are parameter-independent, so they only need to
    be built once per script run. (1+z_obs) is also reused in every dL call.
    """
    z = np.atleast_1d(z)
    # Cache key must be hashable: cast arrays to their max and shape tuple.
    key = (float(z.max()), int(z_grid_points), z.shape)
    if key not in _z_cache:
        z_grid = np.linspace(1e-8, z.max(), z_grid_points)
        zp1_cubed = (1.0 + z_grid) ** 3
        one_plus_z = 1.0 + z
        _z_cache[key] = (z_grid, zp1_cubed, one_plus_z)
    return _z_cache[key]


def H_model(z, Om_m0, H_0):
    """
    Theoretical Hubble parameter for a flat Lambda-CDM model.
    Formula: H(z) = H0 * sqrt(Omega_m * (1+z)^3 + (1 - Omega_m))
    Kept for reference; hot paths inline this formula to avoid call overhead.
    """
    E_z = np.sqrt(Om_m0 * (1 + z) ** 3 + (1 - Om_m0))
    return E_z * H_0


def mu_model(z, Om_m0, H_0, z_grid_points=2000):
    """
    Theoretical distance modulus for a flat Lambda-CDM model (scalar
    Om_m0, H_0). Used by curve_fit and calc_chisq_cov, where parameters
    are evaluated one point at a time.

    mu(z) = 5*log10(d_L(z) / 10 pc)
    d_L(z) = (1+z) * c * Integral_0^z dz' / H(z')

    The line-of-sight comoving distance is obtained once on a fine redshift
    grid via cumulative trapezoidal integration and then interpolated at the
    observed SN redshifts.

    OPTIMIZATION: Reuses cached (1+z_grid)^3 and (1+z_obs) from _get_z_cache.
    """
    z = np.atleast_1d(z)
    z_grid, zp1_cubed, one_plus_z = _get_z_cache(z, z_grid_points)

    # Inline H(z) on the grid: avoid a function call inside the hot loop.
    integrand = C_LIGHT / (np.sqrt(Om_m0 * zp1_cubed + (1.0 - Om_m0)) * H_0)
    cum_integral = np.concatenate(([0.0], cumulative_trapezoid(integrand, z_grid)))
    Dc = np.interp(z, z_grid, cum_integral)   # comoving distance [Mpc]
    dL = one_plus_z * Dc                       # luminosity distance [Mpc]

    return 5.0 * np.log10(dL) + 25.0          # +25 converts Mpc -> 10 pc units


def mu_model_batch(z, Om_arr, H0_arr, z_grid_points=2000, max_chunk=5000):
    """
    Vectorized distance modulus for MANY (Om, H0) pairs at once.

    This is the main optimization in this script. The original
    calc_chisq_diag looped in pure Python over every (Om, H0) grid point
    (100 x 100 = 10,000 points) and called mu_model individually for each
    one, re-doing a full 2000-point integration every time. Here all K
    parameter pairs are integrated simultaneously as a single (K, G) numpy
    array operation, then interpolated onto the observed SN redshifts in
    one batched step.

    ADDITIONAL OPTIMIZATIONS:
      1. Cached z_grid quantities (no redundant pow/concat).
      2. Chunked evaluation: limits peak memory when K is huge (e.g. 10k
         contour grid points) by processing at most max_chunk models at once.
      3. Interpolation indices/weights are precomputed outside the chunk loop
         because they depend only on observed redshifts, not on (Om, H0).

    Returns shape (K, N): one model distance-modulus curve per (Om, H0)
    pair, evaluated at all N observed redshifts.
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

    # Precompute interpolation indices & weights (parameter-independent).
    # searchsorted gives the insertion point; clip to valid range for safety.
    idx = np.clip(np.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 2)
    x0, x1 = z_grid[idx], z_grid[idx + 1]
    frac = (z - x0) / (x1 - x0)

    # Process the full batch in memory-safe chunks.
    for start in range(0, K, max_chunk):
        end = min(start + max_chunk, K)
        Om_c = Om_arr[start:end]
        H0_c = H0_arr[start:end]
        Kc = end - start

        # E(z) for every (Om, H0) pair at once -> shape (Kc, G)
        Ez = np.sqrt(Om_c[:, None] * zp1_cubed[None, :] + (1.0 - Om_c[:, None]))
        Hz = Ez * H0_c[:, None]
        integrand = C_LIGHT / Hz

        # Cumulative trapezoidal integral along the grid axis, for all Kc rows
        # at once (axis=1 keeps each parameter pair's integration independent).
        cum = np.concatenate(
            [np.zeros((Kc, 1)),
             cumulative_trapezoid(integrand, z_grid, axis=1)],
            axis=1
        )

        # Batched linear interpolation using the precomputed idx/frac.
        y0, y1 = cum[:, idx], cum[:, idx + 1]
        Dc = y0 + frac[None, :] * (y1 - y0)
        result[start:end] = one_plus_z[None, :] * Dc

    return 5.0 * np.log10(result) + 25.0


def calc_chisq_diag(pars, z_vals, mu_vals, inv_var):
    """
    Chi-squared using diagonal errors only, fully vectorized over a whole
    array of (Om, H0) pairs at once (used for the fast contour map).

    OPTIMIZATION: Accepts precomputed 1/sigma^2 (inv_var) instead of raw
    mu_err, saving a division per SN per evaluation.
    """
    Om_m0, H_0 = pars
    Om_m0 = np.atleast_1d(Om_m0)
    H_0 = np.atleast_1d(H_0)
    model = mu_model_batch(z_vals, Om_m0, H_0)
    dmu = mu_vals[None, :] - model
    return np.sum(dmu ** 2 * inv_var[None, :], axis=1)


def calc_chisq_cov(theta, z_vals, mu_vals, cov_inv):
    """
    Chi-squared using the full covariance matrix: chi2 = dmu^T Cinv dmu.
    Used for the best-fit optimization and the MCMC likelihood, where the
    off-diagonal SN systematics genuinely matter.
    """
    Om_m0, H_0 = theta
    model = mu_model(z_vals, Om_m0, H_0)
    dmu = mu_vals - model
    return float(dmu @ cov_inv @ dmu)


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
        z_vals, mu_vals, mu_err, mask = load_sn_data(data_dir)
        cov = load_sn_covariance(data_dir, mask)
        print(f"\nSuccessfully loaded {len(z_vals)} SNe.")
        print(f"Redshift range: {z_vals.min():.4f} to {z_vals.max():.4f}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Decide whether to use the full covariance matrix or only diagonal errors.
    # The full covariance accounts for correlated systematic uncertainties.
    use_full_cov = cov is not None
    if use_full_cov:
        cov_inv = np.linalg.inv(cov)
    else:
        cov_inv = np.diag(1.0 / mu_err ** 2)

    # Precompute inverse variance for the fast diagonal path.
    inv_var = 1.0 / mu_err ** 2

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
        # OPTIMIZATION: Chunked evaluation + explicit matmul instead of einsum.
        # einsum('ki,ij,kj->k') is elegant but matmul uses optimized BLAS (GEMM).
        K = xx.size
        Z = np.empty(K)
        x_flat, y_flat = xx.ravel(), yy.ravel()
        chunk_size = 2000  # tune based on available RAM
        for start in range(0, K, chunk_size):
            end = min(start + chunk_size, K)
            model = mu_model_batch(z_vals, x_flat[start:end], y_flat[start:end])
            dmu = mu_vals[None, :] - model
            # dmu @ cov_inv  -> shape (Kc, N); elementwise multiply with dmu and sum over N
            dmu_cov = dmu @ cov_inv
            Z[start:end] = np.sum(dmu_cov * dmu, axis=1)
        Z = Z.reshape(xx.shape)
    else:
        # Fast diagonal path: 10,000 chi^2 values in a single vectorized call.
        Z = calc_chisq_diag([xx.ravel(), yy.ravel()], z_vals, mu_vals, inv_var).reshape(xx.shape)

    # Convert absolute chi-squared values into Delta-chi^2 values.
    # Confidence regions depend only on the difference from the minimum.
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
    ax.set_title(r'$\Delta\chi^2$ Confidence Contours (Pantheon+ SNe Ia)')
    ax.legend()
    fig.colorbar(cf, ax=ax, label=r'$\Delta\chi^2$')

    plt.savefig(os.path.join(output_dir, "DeltaChi2_Contour_SN.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # --- Step 4: Bayesian MCMC Sampling ---
    print("\n--- Running MCMC Sampling ---")

    def log_prob_vec(theta_batch, z_vals, mu_vals, cov_inv):
        """
        Vectorized log-posterior for the whole ensemble at once.

        The original log_prob/log_prior pair was called once per walker
        per step (32 walkers x 3000 steps = 96,000 separate Python calls),
        each doing its own 2000-point mu_model integration and its own
        1701x1701 covariance matrix-vector product. Here every walker's
        position is stacked into a single (nwalkers, 2) array; mu_model_batch
        integrates all of them in one shot, and a batched matmul evaluates
        every walker's chi^2 simultaneously. Passing vectorize=True to emcee
        tells it to call this function once per step with the whole ensemble.

        OPTIMIZATION: matmul+sum instead of einsum for the quadratic form,
        routing through optimized BLAS.
        """
        Om = theta_batch[:, 0]
        H0 = theta_batch[:, 1]

        # Flat prior over a sensible box; -inf outside.
        in_prior = (Om > 0.0) & (Om < 1.0) & (H0 > 40.0) & (H0 < 100.0)
        lp = np.where(in_prior, 0.0, -np.inf)

        chi2 = np.full(theta_batch.shape[0], np.inf)
        if np.any(in_prior):
            model = mu_model_batch(z_vals, Om[in_prior], H0[in_prior])
            dmu = mu_vals[None, :] - model
            dmu_cov = dmu @ cov_inv
            chi2[in_prior] = np.sum(dmu_cov * dmu, axis=1)

        return lp - 0.5 * chi2

    ndim, nwalkers, nsteps = 2, 32, 3000
    # Initialize walkers in a tiny Gaussian ball around the frequentist best fit.
    pos = popt + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([1, 10])

    # Optional: HDF5 backend so chains survive interruptions and can be resumed.
    backend_kwargs = {}
    try:
        backend = emcee.backends.HDFBackend(os.path.join(output_dir, "mcmc.h5"))
        backend.reset(nwalkers, ndim)
        backend_kwargs["backend"] = backend
    except (ImportError, AttributeError):
        pass

    # Initialize the affine-invariant ensemble sampler in vectorized mode,
    # so log_prob_vec is called once per step with the full walker
    # ensemble rather than once per individual walker.
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_vec,
        args=(z_vals, mu_vals, cov_inv),
        vectorize=True,
        **backend_kwargs
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    # OPTIMIZATION: Use autocorrelation time to choose discard/thin
    # instead of hard-coded values. This maximizes effective sample size.
    try:
        tau = sampler.get_autocorr_time()
        discard = int(2 * np.max(tau))
        thin = max(1, int(0.5 * np.min(tau)))
        print(f"  Autocorrelation time: {tau}")
        print(f"  Using discard={discard}, thin={thin}")
    except emcee.autocorr.AutocorrError as e:
        print(f"  Autocorr warning: {e}")
        discard, thin = 500, 15

    # Remove burn-in, thin the chain, and flatten all walkers into one sample.
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)

    # Report 16th, 50th, 84th percentiles = median +/- 1-sigma credible interval.
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
    plt.savefig(os.path.join(output_dir, "MCMC_H0_vs_Omega_m_SN.png"),
                dpi=300, bbox_inches='tight')
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

    # Shaded bands for the two reference measurements.
    ax.axvspan(67.4 - 0.5, 67.4 + 0.5, color='steelblue', alpha=0.15)
    ax.axvspan(73.04 - 1.04, 73.04 + 1.04, color='darkorange', alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Hubble_Parameter_SN.png"),
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # --- Step 6: Export Results ---
    print("\n--- Exporting Results ---")
    with open(os.path.join(output_dir, "lcdm_fit_results_SN.txt"), "w") as f:
        f.write("# LCDM Fit Results (Pantheon+ Type Ia SNe)\n")
        f.write("# =========================================\n")
        f.write(f"N_SNe used    = {len(z_vals)}\n")
        f.write(f"Used full covariance matrix = {use_full_cov}\n")
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

    print(f"  Results exported to: {os.path.join(output_dir, 'lcdm_fit_results_SN.txt')}")
    contour_data = {'X': xx, 'Y': yy, 'delta_chi2': delta_chisq}
    np.save(os.path.join(output_dir, 'contour_H0_Om_lcdm_SN.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(output_dir, 'contour_H0_Om_lcdm_SN.npy')}")


if __name__ == "__main__":
    main()