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
    """
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


@lru_cache(maxsize=8)
def _get_z_grid(z_max, z_grid_points):
    """
    Cached fine redshift integration grid used by mu_model/mu_model_batch.

    z_max (and z_grid_points) is essentially constant for an entire script
    run - it's fixed by the observed SN sample, not by (Om_m0, H_0). The
    original code called np.linspace(1e-8, z.max(), 2000) fresh on *every*
    single mu_model call: once per curve_fit iteration, once per contour
    grid point (10,000 calls), and once per MCMC likelihood evaluation
    (nwalkers * nsteps ~ 10^5 calls). None of those calls needed a new
    grid - only the model parameters changed. Caching it here removes that
    redundant work entirely (lru_cache is keyed on the exact float/int
    arguments, so it's safe as long as z_max is passed consistently).
    """
    return np.linspace(1e-8, z_max, z_grid_points)


def mu_model(z, Om_m0, H_0, z_grid_points=2000):
    """
    Theoretical distance modulus for a flat Lambda-CDM model (scalar
    Om_m0, H_0). Used by curve_fit and calc_chisq_cov, where parameters
    are evaluated one point at a time.

    mu(z) = 5*log10(d_L(z) / 10 pc)
    d_L(z) = (1+z) * c * Integral_0^z dz' / H(z')

    The line-of-sight comoving distance is obtained once on a fine redshift
    grid via cumulative trapezoidal integration and then interpolated at the
    observed SN redshifts - the same "solve once on a grid, interpolate"
    pattern used in the ODE-based H(z) ROMS solvers elsewhere in this thesis
    codebase, just applied to the distance integral instead.
    """
    z = np.atleast_1d(z)
    z_grid = _get_z_grid(float(z.max()), z_grid_points)

    integrand = C_LIGHT / H_model(z_grid, Om_m0, H_0)
    cum_integral = np.concatenate(([0.0], cumulative_trapezoid(integrand, z_grid)))

    Dc = np.interp(z, z_grid, cum_integral)   # comoving distance [Mpc]
    dL = (1.0 + z) * Dc                       # luminosity distance [Mpc]

    return 5.0 * np.log10(dL) + 25.0          # +25 converts Mpc -> 10 pc units


def mu_model_batch(z, Om_arr, H0_arr, z_grid_points=2000):
    """
    Vectorized distance modulus for MANY (Om, H0) pairs at once.

    This is the main optimization in this script. The original
    calc_chisq_diag looped in pure Python over every (Om, H0) grid point
    (100 x 100 = 10,000 points) and called mu_model individually for each
    one, re-doing a full 2000-point integration every time. Here all K
    parameter pairs are integrated simultaneously as a single (K, G) numpy
    array operation, then interpolated onto the observed SN redshifts in
    one batched step - turning a 10,000-iteration Python loop into a
    handful of numpy array ops.

    Returns shape (K, N): one model distance-modulus curve per (Om, H0)
    pair, evaluated at all N observed redshifts.
    """
    Om_arr = np.atleast_1d(np.asarray(Om_arr, dtype=float))
    H0_arr = np.atleast_1d(np.asarray(H0_arr, dtype=float))
    z = np.atleast_1d(z)

    z_grid = _get_z_grid(float(z.max()), z_grid_points)   # (G,)

    # H(z) for every (Om, H0) pair at once -> shape (K, G)
    Ez = np.sqrt(Om_arr[:, None] * (1.0 + z_grid)[None, :] ** 3
                 + (1.0 - Om_arr[:, None]))
    Hz = Ez * H0_arr[:, None]
    integrand = C_LIGHT / Hz                                    # (K, G)

    # Cumulative trapezoidal integral along the grid axis, for all K rows
    # at once (axis=1 keeps each parameter pair's integration independent).
    cum = np.concatenate(
        [np.zeros((integrand.shape[0], 1)),
         cumulative_trapezoid(integrand, z_grid, axis=1)],
        axis=1
    )   # (K, G)

    # np.interp only interpolates one function at a time, so the batched
    # interpolation is done manually here. Since z_grid is shared across
    # all K rows, a single searchsorted call locates every observed
    # redshift's bracketing grid indices, and the linear interpolation
    # weights are then applied to all K rows simultaneously.
    idx = np.clip(np.searchsorted(z_grid, z) - 1, 0, len(z_grid) - 2)
    x0, x1 = z_grid[idx], z_grid[idx + 1]
    frac = (z - x0) / (x1 - x0)
    y0, y1 = cum[:, idx], cum[:, idx + 1]
    Dc = y0 + frac[None, :] * (y1 - y0)                         # (K, N)

    dL = (1.0 + z)[None, :] * Dc
    return 5.0 * np.log10(dL) + 25.0                            # (K, N)


def calc_chisq_diag(pars, z_vals, mu_vals, mu_err):
    """
    Chi-squared using diagonal errors only, vectorized over a whole array
    of (Om, H0) pairs at once (used for the fast contour map). Replaces
    the original per-point Python loop with a single call to
    mu_model_batch plus a vectorized reduction over the SN axis.
    """
    Om_m0, H_0 = pars
    Om_m0 = np.atleast_1d(Om_m0)
    H_0 = np.atleast_1d(H_0)

    model = mu_model_batch(z_vals, Om_m0, H_0)                  # (K, N)
    chi2 = np.sum((mu_vals[None, :] - model) ** 2 / mu_err[None, :] ** 2, axis=1)
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
        z_vals, mu_vals, mu_err, mask = load_sn_data(data_dir)
        cov = load_sn_covariance(data_dir, mask)
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
    else:
        cov_inv = np.diag(1.0 / mu_err ** 2)

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
        # Vectorized equivalent of the original per-point np.ndindex loop:
        # batch-evaluate the model for every (Om, H0) grid point at once via
        # mu_model_batch, then compute all 10,000 chi^2 = dmu^T Cinv dmu
        # values in one einsum call instead of 10,000 separate Python calls
        # (each of which previously re-ran a full 2000-point integration).
        model_grid = mu_model_batch(z_vals, xx.ravel(), yy.ravel())     # (K, N)
        dmu_grid = mu_vals[None, :] - model_grid                        # (K, N)
        Z = np.einsum('ki,ij,kj->k', dmu_grid, cov_inv, dmu_grid).reshape(xx.shape)
    else:
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

    def log_prob_vec(theta_batch, z_vals, mu_vals, cov_inv):
        """
        Vectorized log-posterior for ALL walkers at once.

        The original log_prob/log_prior pair was called once per walker
        per step (32 walkers x 3000 steps = 96,000 separate Python calls),
        each doing its own 2000-point mu_model integration and its own
        1701x1701 covariance matrix-vector product. Here every walker's
        position is stacked into a single (nwalkers, 2) array; mu_model_batch
        integrates all of them in one shot, and np.einsum evaluates every
        walker's chi^2 = dmu^T Cinv dmu simultaneously. Passing
        vectorize=True to emcee below tells it to call this function once
        per step with the whole ensemble, instead of once per walker.
        """
        Om = theta_batch[:, 0]
        H0 = theta_batch[:, 1]

        in_prior = (Om > 0.0) & (Om < 1.0) & (H0 > 40.0) & (H0 < 100.0)
        lp = np.where(in_prior, 0.0, -np.inf)

        chi2 = np.full(theta_batch.shape[0], np.inf)
        if np.any(in_prior):
            model = mu_model_batch(z_vals, Om[in_prior], H0[in_prior])   # (Kfit, N)
            dmu = mu_vals[None, :] - model
            chi2[in_prior] = np.einsum('ki,ij,kj->k', dmu, cov_inv, dmu)

        return lp - 0.5 * chi2

    ndim, nwalkers, nsteps = 2, 32, 3000

    pos = popt + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([1, 10])

    # Initialize the affine-invariant ensemble sampler in vectorized mode,
    # so log_prob_vec is called once per step with the full walker
    # ensemble rather than once per individual walker.
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_vec,
        args=(z_vals, mu_vals, cov_inv),
        vectorize=True
    )
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

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Hubble_Parameter_SN.png"), dpi=300, bbox_inches='tight')
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
        f.write(f"MCMC Omega_m  = {Om_mcmc[1]:.6f} (+{Om_mcmc[2]-Om_mcmc[1]:.6f} / -{Om_mcmc[1]-Om_mcmc[0]:.6f})\n")
        f.write(f"MCMC H_0      = {H0_mcmc[1]:.6f} (+{H0_mcmc[2]-H0_mcmc[1]:.6f} / -{H0_mcmc[1]-H0_mcmc[0]:.6f})\n")

    print(f"  Results exported to: {os.path.join(output_dir, 'lcdm_fit_results_SN.txt')}")
    print(f"  Plots saved in: {output_dir}")

    contour_data = {'X': xx, 'Y': yy, 'delta_chi2': delta_chisq}
    np.save(os.path.join(output_dir, 'contour_H0_Om_lcdm_SN.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(output_dir, 'contour_H0_Om_lcdm_SN.npy')}")


if __name__ == "__main__":
    main()