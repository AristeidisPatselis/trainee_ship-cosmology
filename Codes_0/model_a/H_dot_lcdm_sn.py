"""
H_dot_lcdm_sn.py
================
Modified Friedmann equation fit to Pantheon+SH0ES Type Ia supernovae:

    H(z)^2 = H0^2 * Om * (1+z)^3  -  alpha * (1+z) * H(z) * dH/dz

Free parameters: (H0, Om, alpha). alpha=0 collapses the correction term, but
note this is NOT the standard LambdaCDM limit -- there is no explicit
(1-Om) dark-energy term anywhere in this equation. The entire late-time
behaviour has to come from the alpha*(1+z)*H*dH/dz piece, which is exactly
what makes alpha the interesting parameter to pin down here.

This is the SN analogue of H_dot_lcdm_fit.py (which fits cosmic-chronometer
H(z) data directly). The underlying algebraic model is identical -- same
closed-form solution for H(z), same (H0, Om, alpha) parameters -- but the
observable is now the Pantheon+SH0ES distance modulus

    mu(z) = 5 log10(d_L(z)/10 pc),
    d_L(z) = (1+z) * c * Integral_0^z dz'/H(z'),

so every likelihood evaluation integrates 1/H(z) over a redshift grid
instead of comparing H(z) directly to chronometer data. Structurally this
script mirrors delta_lcdm_sn.py (the SN analogue of delta_lcdm_fit.py), so
all three "chronometer + SN" pairs share the same pipeline shape and can be
compared line-by-line.

--------------------------------------------------------------------------
Why this is fast (no ODE solver in the hot loop)
--------------------------------------------------------------------------
The u = H^2 substitution turns the modified Friedmann equation into a
linear first-order ODE in u(z), which has an exact closed-form solution
(see model_H_analytical / model_H_batch below):

    u(z) = H0^2 * x^(-2/alpha)
           + [2*H0^2*Om / (3*alpha + 2)] * (x^3 - x^(-2/alpha)),   x = 1+z

This is the same algebraic solution used in H_dot_lcdm_fit.py. Because it
is closed-form, there is no solve_ivp / lru_cache machinery needed here at
all: model_H_batch below evaluates H(z) for an entire batch of (H0, Om,
alpha) parameter sets simultaneously via plain numpy broadcasting, exactly
the way delta_lcdm_sn.py's _solve_H_newton batches its implicit solve. That
batched H(z) is what makes the SN contour/grid scans and the vectorized
MCMC likelihood cheap, even though every likelihood evaluation also
requires a numerical 1/H(z) integration to get mu(z).
--------------------------------------------------------------------------
"""

# --- Standard library --------------------------------------------------------
import os
import warnings
from tqdm import tqdm

# --- Numerics / optimization --------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.stats import chi2 as chi2_dist

# --- Plotting ------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec

# --- Bayesian inference & stats -------------------------------------------------
import emcee
import corner

np.random.seed(42)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIG
# =============================================================================

# Point this at the folder containing the Pantheon+SH0ES data release
# (https://github.com/PantheonPlusSH0ES/DataRelease -> Pantheon+_Data/)
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/'

SN_DATA_FILE = 'Pantheon+SH0ES.dat'
SN_COV_FILE  = 'Pantheon+SH0ES_STAT+SYS.cov'

Z_COL      = 'zHD'
MU_COL     = 'MU_SH0ES'
MU_ERR_COL = 'MU_SH0ES_ERR_DIAG'
CALIB_COL  = 'IS_CALIBRATOR'

# Keep every SN (including Cepheid calibrators) so the absolute H0 scale
# is constrained simultaneously with cosmology.
Z_MIN = -1.0
EXCLUDE_CALIBRATORS = False

# Full covariance is used for best-fit + MCMC. Contour maps fall back to
# diagonal errors for speed unless this flag is True.
USE_FULL_COV_FOR_CONTOUR_MAP = False

C_LIGHT = 299792.458  # km/s

# --- FIT CONFIGURATION ---
BOUNDS = [(50.0, 100.0), (0.01, 3.0), (0.01, 6.0)]   # H0, Om, alpha
PARAM_NAMES = ['H0', 'Om', 'alpha']
PARAM_LABELS = {'H0': r'$H_0$', 'Om': r'$\Omega_{m,0}$', 'alpha': r'$\alpha$'}

CONTOUR_GRID = 40
PROFILE_POINTS = 40
Z_GRID_POINTS = 800        # redshift resolution for the 1/H integral

# emcee sampler settings
NWALKERS = 32
NSTEPS = 3000
DISCARD = 500
THIN = 15

CONF_LEVELS_2D = [2.30, 6.18, 11.83]
CONF_LEVELS_1D = [1.0, 4.0, 9.0]

N_MULTISTART = 8


# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

def setup_matplotlib(use_latex=False):
    """
    use_latex=True spawns a separate latex+dvipng subprocess for every
    distinct piece of plot text (including every tick label). On plots with
    many numeric ticks (corner plots especially) this can mean dozens to
    hundreds of subprocess launches per figure. Leave this False unless you
    specifically need LaTeX-rendered labels; mathtext covers everything
    used in this script (\\alpha, \\Omega_{m,0}, etc.).
    """
    if use_latex:
        try:
            rc('text', usetex=True)
            rc('font', family='serif')
            fig_test = plt.figure()
            plt.text(0.5, 0.5, r"$\alpha$")
            fig_test.canvas.draw()
            plt.close(fig_test)
            return
        except Exception as e:
            print(f"Note: LaTeX rendering unavailable, using mathtext instead. ({e})")
    rc('text', usetex=False)
    rc('font', family='DejaVu Sans')


def find_file_recursively(filename, data_dir, max_depth=4):
    """
    Look for `filename` directly in data_dir first (fast path). Only if
    that fails, fall back to a recursive walk - capped at max_depth so a
    misconfigured DATA_DIR can't turn into an unbounded whole-disk scan.
    """
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath

    print(f"  '{filename}' not found directly in {data_dir}; "
          f"searching subdirectories (max depth {max_depth})...")
    base_depth = data_dir.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(data_dir):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirs[:] = []  # don't descend further from here
            continue
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(
        f"Could not find '{filename}' in '{data_dir}' or its subdirectories "
        f"(searched up to depth {max_depth}). Check DATA_DIR is set correctly."
    )


def load_sn_data(data_dir):
    """Load Pantheon+SH0ES Hubble diagram with the configured quality cuts."""
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
    """Load full stat+syst covariance and slice to the surviving SNe."""
    try:
        filepath = find_file_recursively(SN_COV_FILE, data_dir)
    except FileNotFoundError:
        print("  Covariance file not found - falling back to diagonal errors.")
        return None

    print(f"  Found SN covariance matrix: {filepath}")
    with open(filepath, "r") as f:
        n = int(f.readline().strip())
        vals = np.loadtxt(f, dtype=float)

    if vals.size != n * n:
        raise ValueError(f"Covariance size mismatch: expected {n*n}, got {vals.size}")

    cov_full = vals.reshape(n, n)
    if mask.size != n:
        raise ValueError(f"Covariance dimension ({n}) does not match data rows ({mask.size})")
    return cov_full[np.ix_(mask, mask)]


# =============================================================================
# 2. MODEL: closed-form H(z) from the u = H^2 substitution, then distance modulus
# =============================================================================

def model_H_analytical(z_eval, H0, Om, alpha):
    """
    Exact closed-form solution of

        H^2 = H0^2 * Om * (1+z)^3 - alpha * (1+z) * H * dH/dz

    for a single scalar parameter set (H0, Om, alpha), evaluated at every
    redshift in z_eval. Identical to H_dot_lcdm_fit.py's model_H_analytical.
    """
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    if alpha <= 0 or H0 <= 0 or Om <= 0:
        return np.full_like(z_eval, np.nan)

    x = 1.0 + z_eval

    # Clip extreme exponent boundaries to avoid potential float overflow/underflow
    exponent = np.clip(-2.0 / alpha, -100.0, 100.0)
    pow_term = x ** exponent

    u_of_z = (H0 ** 2) * pow_term + (2.0 * H0 ** 2 * Om / (3.0 * alpha + 2.0)) * (x ** 3 - pow_term)
    u_of_z = np.clip(u_of_z, 1e-10, None)

    return np.sqrt(u_of_z)


def model_H_batch(z_grid, H0_arr, Om_arr, alpha_arr):
    """
    Same closed-form solution as model_H_analytical, but for a whole BATCH
    of (H0, Om, alpha) parameter sets at once, broadcast against a shared
    redshift grid.

    z_grid   : shape (nz,)
    H0/Om/alpha_arr : shape (n,)
    Returns  : H(z) of shape (n, nz), with rows that hit invalid parameters
               (alpha/H0/Om <= 0) set to NaN.

    This is what makes the SN contour/grid scans and the vectorized MCMC
    likelihood cheap: instead of looping in Python over every (H0, Om,
    alpha) point, all parameter sets are solved simultaneously via
    broadcasting, exactly analogous to delta_lcdm_sn.py's _solve_H_newton.
    """
    z_grid = np.asarray(z_grid, dtype=float)
    H0_arr = np.atleast_1d(np.asarray(H0_arr, dtype=float))
    Om_arr = np.atleast_1d(np.asarray(Om_arr, dtype=float))
    alpha_arr = np.atleast_1d(np.asarray(alpha_arr, dtype=float))

    invalid = (alpha_arr <= 0) | (H0_arr <= 0) | (Om_arr <= 0)

    H0c = np.where(invalid, 1.0, H0_arr)[:, None]
    Omc = np.where(invalid, 1.0, Om_arr)[:, None]
    alphac = np.where(invalid, 1.0, alpha_arr)[:, None]

    x = (1.0 + z_grid)[None, :]

    exponent = np.clip(-2.0 / alphac, -100.0, 100.0)
    pow_term = x ** exponent

    u_of_z = (H0c ** 2) * pow_term + (2.0 * H0c ** 2 * Omc / (3.0 * alphac + 2.0)) * (x ** 3 - pow_term)
    u_of_z = np.clip(u_of_z, 1e-10, None)
    H = np.sqrt(u_of_z)
    H[invalid, :] = np.nan
    return H


def H_lcdm(z, H0, Om):
    """Standard flat LambdaCDM, used only as the baseline for AIC/BIC."""
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


# Cache for the fixed redshift grid used by the distance integral
_z_grid_cache = {}


def _get_integration_grid(z_obs, n_points=Z_GRID_POINTS):
    """Return a fixed redshift grid from ~0 to max(z_obs)."""
    z_max = float(np.max(z_obs))
    key = (z_max, n_points)
    if key not in _z_grid_cache:
        z_grid = np.linspace(1e-8, z_max, n_points)
        _z_grid_cache[key] = z_grid
    return _z_grid_cache[key]


def mu_model(z, H0, Om, alpha, z_grid_points=Z_GRID_POINTS):
    """Distance modulus for the H_dot-alpha model at a single (H0, Om, alpha)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z_grid = _get_integration_grid(z, z_grid_points)

    H_grid = model_H_analytical(z_grid, H0, Om, alpha)
    if np.any(~np.isfinite(H_grid)) or np.any(H_grid <= 0):
        return np.full_like(z, 1e6, dtype=float)

    integrand = C_LIGHT / H_grid
    cum = cumulative_trapezoid(integrand, z_grid, initial=0.0)
    Dc = np.interp(z, z_grid, cum)
    dL = (1.0 + z) * Dc
    dL = np.where(dL > 0, dL, np.nan)
    mu = 5.0 * np.log10(dL) + 25.0
    if np.any(~np.isfinite(mu)):
        return np.full_like(z, 1e6, dtype=float)
    return mu


def mu_model_batch(z_obs, H0_arr, Om_arr, alpha_arr, z_grid_points=Z_GRID_POINTS):
    """
    Distance modulus for a whole BATCH of (H0, Om, alpha) parameter sets at
    once, evaluated at the same observed redshifts z_obs.

    Returns an array of shape (n_params, len(z_obs)), with rows that failed
    (non-finite / non-positive H or dL anywhere) set to 1e6, matching the
    failure convention of mu_model / chi2_diag.
    """
    z_obs = np.atleast_1d(np.asarray(z_obs, dtype=float))
    H0_arr = np.atleast_1d(np.asarray(H0_arr, dtype=float))
    Om_arr = np.atleast_1d(np.asarray(Om_arr, dtype=float))
    alpha_arr = np.atleast_1d(np.asarray(alpha_arr, dtype=float))
    n_params = H0_arr.shape[0]

    z_grid = _get_integration_grid(z_obs, z_grid_points)  # (nz,)

    H_grid = model_H_batch(z_grid, H0_arr, Om_arr, alpha_arr)  # (n_params, nz)

    bad_H = ~np.isfinite(H_grid) | (H_grid <= 0)
    safe_H = np.where(bad_H, 1.0, H_grid)
    integrand = C_LIGHT / safe_H
    cum = cumulative_trapezoid(integrand, z_grid, axis=1, initial=0.0)  # (n_params, nz)

    Dc = np.empty((n_params, z_obs.shape[0]))
    for i in range(n_params):
        Dc[i] = np.interp(z_obs, z_grid, cum[i])

    dL = (1.0 + z_obs)[None, :] * Dc
    with np.errstate(invalid='ignore', divide='ignore'):
        mu = np.where(dL > 0, 5.0 * np.log10(np.clip(dL, 1e-300, None)) + 25.0, np.nan)

    row_bad = np.any(bad_H, axis=1) | np.any(~np.isfinite(mu), axis=1)
    mu[row_bad, :] = 1e6
    return mu


def mu_lcdm(z, H0, Om, z_grid_points=Z_GRID_POINTS):
    """Distance modulus for pure flat LambdaCDM (analytic H), for comparison."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z_grid = _get_integration_grid(z, z_grid_points)
    H_grid = H_lcdm(z_grid, H0, Om)
    integrand = C_LIGHT / H_grid
    cum = cumulative_trapezoid(integrand, z_grid, initial=0.0)
    Dc = np.interp(z, z_grid, cum)
    dL = (1.0 + z) * Dc
    dL = np.where(dL > 0, dL, np.nan)
    return 5.0 * np.log10(dL) + 25.0


# =============================================================================
# 3. CHI-SQUARED
# =============================================================================

def _within_bounds(params):
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, BOUNDS))


def chi2_diag(params, z_vals, mu_vals, inv_var):
    """Diagonal-error chi-squared (fast path for contours)."""
    H0, Om, alpha = params
    if not _within_bounds(params):
        return 1e12
    mu_th = mu_model(z_vals, H0, Om, alpha)
    if np.any(~np.isfinite(mu_th)):
        return 1e12
    return float(np.sum((mu_vals - mu_th) ** 2 * inv_var))


def chi2_cov(params, z_vals, mu_vals, cov_inv):
    """Full-covariance chi-squared: dmu^T C^{-1} dmu."""
    H0, Om, alpha = params
    if not _within_bounds(params):
        return 1e12
    mu_th = mu_model(z_vals, H0, Om, alpha)
    if np.any(~np.isfinite(mu_th)):
        return 1e12
    dmu = mu_vals - mu_th
    return float(dmu @ cov_inv @ dmu)


def chi2_lcdm_cov(params, z_vals, mu_vals, cov_inv):
    """LambdaCDM chi-squared with full covariance."""
    H0, Om = params
    if H0 <= 0 or not (0 < Om < 1):
        return 1e12
    mu_th = mu_lcdm(z_vals, H0, Om)
    dmu = mu_vals - mu_th
    return float(dmu @ cov_inv @ dmu)


def chi2_grid_diag(params_grid, z_vals, mu_vals, inv_var):
    """
    Chi-squared over a grid of (H0, Om, alpha) using diagonal errors.

    Vectorized: all valid grid points are solved and evaluated together via
    mu_model_batch instead of being looped over one at a time.
    """
    chi2_vals = np.full(len(params_grid), 1e12)
    valid_mask = np.array([_within_bounds(p) for p in params_grid])
    if not np.any(valid_mask):
        return chi2_vals

    valid_params = params_grid[valid_mask]
    H0_arr = valid_params[:, 0]
    Om_arr = valid_params[:, 1]
    alpha_arr = valid_params[:, 2]

    mu_th = mu_model_batch(z_vals, H0_arr, Om_arr, alpha_arr)  # (n_valid, n_z)
    bad_rows = np.any(~np.isfinite(mu_th), axis=1) | np.any(mu_th >= 1e6 - 1.0, axis=1)

    resid2 = (mu_vals[None, :] - mu_th) ** 2 * inv_var[None, :]
    chi2_valid = np.sum(resid2, axis=1)
    chi2_valid[bad_rows] = 1e12

    chi2_vals[valid_mask] = chi2_valid
    return chi2_vals


# =============================================================================
# 4. BEST FIT
# =============================================================================

def best_fit(z_vals, mu_vals, cov_inv, use_full_cov, inv_var,
             n_starts=N_MULTISTART, verbose=True):
    """Global differential-evolution + multi-start Nelder-Mead polish."""
    chi2_fn = (lambda p, *a: chi2_cov(p, *a)) if use_full_cov else \
              (lambda p, *a: chi2_diag(p, *a))

    args = (z_vals, mu_vals, cov_inv) if use_full_cov else (z_vals, mu_vals, inv_var)

    print("  Running differential evolution...")
    de_result = differential_evolution(
        chi2_fn, bounds=BOUNDS, args=args,
        seed=42, maxiter=150, tol=1e-7, polish=True, popsize=15,
    )
    best_x, best_chi2 = de_result.x, de_result.fun

    print(f"  Running {n_starts} multi-start local optimizations...")
    rng = np.random.default_rng(42)
    starts = [best_x] + [
        [rng.uniform(lo, hi) for (lo, hi) in BOUNDS] for _ in range(n_starts)
    ]

    local_results = []
    for x0 in tqdm(starts, desc="  Local optimizations", disable=not verbose):
        res = minimize(chi2_fn, x0, args=args, method='Nelder-Mead',
                       bounds=BOUNDS,
                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 4000})
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        print(f"  Multi-start scan: {len(spread)}/{len(starts)} runs converged "
              f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
        if spread.size and spread.max() - spread.min() > 1.0:
            print("  -> spread across starts suggests a degenerate/multi-modal "
                  "chi^2 surface")

    return best_x, best_chi2, de_result.success


# =============================================================================
# 5. UNCERTAINTIES: curve_fit + MCMC
# =============================================================================

def model_mu_curvefit(z_array, H0, Om, alpha):
    """curve_fit-friendly signature."""
    mu = mu_model(z_array, H0, Om, alpha)
    if np.any(~np.isfinite(mu)):
        return np.full_like(np.atleast_1d(z_array), 1e6, dtype=float)
    return mu


def fit_uncertainties_curvefit(z_vals, mu_vals, sigma, p0, use_full_cov, cov):
    lo = [b[0] for b in BOUNDS]
    hi = [b[1] for b in BOUNDS]
    if use_full_cov and cov is not None:
        popt, pcov = curve_fit(
            model_mu_curvefit, z_vals, mu_vals, p0=p0,
            sigma=cov, absolute_sigma=True, bounds=(lo, hi), maxfev=15000,
        )
    else:
        popt, pcov = curve_fit(
            model_mu_curvefit, z_vals, mu_vals, p0=p0,
            sigma=sigma, absolute_sigma=True, bounds=(lo, hi), maxfev=15000,
        )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov


def log_prior(theta):
    for val, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_likelihood(theta, z_vals, mu_vals, cov_inv, use_full_cov, inv_var):
    if use_full_cov:
        c = chi2_cov(theta, z_vals, mu_vals, cov_inv)
    else:
        c = chi2_diag(theta, z_vals, mu_vals, inv_var)
    if c >= 1e11:
        return -np.inf
    return -0.5 * c


def log_prob(theta, z_vals, mu_vals, cov_inv, use_full_cov, inv_var):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_vals, mu_vals, cov_inv, use_full_cov, inv_var)


def run_mcmc(best_x, z_vals, mu_vals, cov_inv, use_full_cov, inv_var,
             nwalkers=NWALKERS, nsteps=NSTEPS, discard=DISCARD, thin=THIN):
    ndim = 3
    spread = np.array([2.0, 0.05, 0.2])
    pos = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        pos[i] = best_x + spread * np.random.randn(ndim)
        for j, (lo, hi) in enumerate(BOUNDS):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-6, hi - 1e-6)

    # The closed-form model_H_analytical is evaluated in microseconds, so
    # (as in H_dot_lcdm_fit.py) there's no benefit to multiprocessing here:
    # process-pool IPC overhead would dominate over the actual likelihood
    # evaluation. Run the sampler sequentially.
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=(z_vals, mu_vals, cov_inv, use_full_cov, inv_var)
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    return sampler, flat_samples


def plot_walkers(sampler, outdir="."):
    chain = sampler.get_chain()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        for walker in range(chain.shape[1]):
            axes[i].plot(chain[:, walker, i], alpha=0.3, lw=0.5)
        axes[i].set_ylabel(PARAM_LABELS[PARAM_NAMES[i]])
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "walker_chains_sn.png"), dpi=300)
    plt.close()


# =============================================================================
# 6. PROFILE LIKELIHOOD & CONFIDENCE CONTOURS
# =============================================================================

def plot_chi2_profile_alpha(best_x, chi2_best, z_vals, mu_vals, inv_var,
                             n_points=PROFILE_POINTS, outdir='.'):
    """1D profile chi^2(alpha): H0 and Om re-fit at every alpha (diagonal)."""
    H0_fit, Om_fit, alpha_fit = best_x
    alpha_lo = max(BOUNDS[2][0], alpha_fit * 0.3)
    alpha_hi = min(BOUNDS[2][1], alpha_fit * 2.5)
    alphas = np.linspace(alpha_lo, alpha_hi, n_points)

    chi2_vals = np.empty(n_points)
    print("  Computing profile likelihood (diagonal errors)...")
    for i, a in enumerate(tqdm(alphas, desc="  Alpha profile")):
        def chi2_reduced(p2):
            return chi2_diag([p2[0], p2[1], a], z_vals, mu_vals, inv_var)
        res = minimize(chi2_reduced, [H0_fit, Om_fit], method='Nelder-Mead',
                       bounds=[BOUNDS[0], BOUNDS[1]])
        chi2_vals[i] = res.fun

    delta_chi2 = chi2_vals - chi2_best

    alpha_lo68 = alpha_hi68 = None
    below = delta_chi2 <= 1.0
    idx_below = np.where(below)[0]
    if idx_below.size:
        i_first, i_last = idx_below[0], idx_below[-1]
        if i_first > 0:
            alpha_lo68 = np.interp(1.0, [delta_chi2[i_first - 1], delta_chi2[i_first]],
                                   [alphas[i_first - 1], alphas[i_first]])
        else:
            alpha_lo68 = alphas[i_first]
        if i_last < n_points - 1:
            alpha_hi68 = np.interp(1.0, [delta_chi2[i_last + 1], delta_chi2[i_last]],
                                   [alphas[i_last + 1], alphas[i_last]])
        else:
            alpha_hi68 = alphas[i_last]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, delta_chi2, color='navy', lw=2)
    ax.axvline(alpha_fit, color='gray', ls=':', lw=1, label=f'best fit ({alpha_fit:.3f})')
    if alpha_lo68 is not None and alpha_hi68 is not None:
        ax.axvspan(alpha_lo68, alpha_hi68, color='navy', alpha=0.12,
                   label=r'1$\sigma$ interval')
    for level, label in zip(CONF_LEVELS_1D, [r'1$\sigma$', r'2$\sigma$', r'3$\sigma$']):
        ax.axhline(level, color='gray', ls='--', lw=0.8)
        ax.text(alphas[-1], level, label, va='bottom', ha='right', fontsize=9, color='gray')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\Delta\chi^2(\alpha)$')
    ax.set_title(r'Profile likelihood: $\Delta\chi^2$ vs $\alpha$ (Pantheon+ SNe)')
    ax.set_ylim(0, 12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chi2_profile_alpha_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if alpha_lo68 is not None and alpha_hi68 is not None:
        print(f"  alpha 1sigma profile interval: [{alpha_lo68:.4f}, {alpha_hi68:.4f}]")
    return alphas, chi2_vals


def _param_range(center, bounds, rel_span=0.6, min_abs_span=0.3):
    """Sign-safe symmetric range around `center`, clipped to `bounds`."""
    span = max(abs(center) * rel_span, min_abs_span)
    lo = max(bounds[0], center - span)
    hi = min(bounds[1], center + span)
    if lo >= hi:
        lo, hi = bounds
    return lo, hi


def plot_contour_2d(best_x, chi2_best, z_vals, mu_vals, inv_var,
                    vary=('alpha', 'Om'), n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour on a 2-D parameter plane (diagonal errors)."""
    idx = {'H0': 0, 'Om': 1, 'alpha': 2}
    ix, iy = idx[vary[0]], idx[vary[1]]
    iz = ({0, 1, 2} - {ix, iy}).pop()

    center = best_x[ix], best_x[iy]
    x_lo, x_hi = _param_range(center[0], BOUNDS[ix])
    y_lo, y_hi = _param_range(center[1], BOUNDS[iy])

    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)

    params_flat = np.zeros((n_grid * n_grid, 3))
    params_flat[:, ix] = X.ravel()
    params_flat[:, iy] = Y.ravel()
    params_flat[:, iz] = best_x[iz]

    print(f"  Computing {n_grid}x{n_grid} grid for {vary[0]}-{vary[1]} contour...")
    chi2_flat = chi2_grid_diag(params_flat, z_vals, mu_vals, inv_var)
    CHI2 = chi2_flat.reshape(n_grid, n_grid)
    delta_chi2 = CHI2 - chi2_best

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(X, Y, delta_chi2, levels=CONF_LEVELS_2D,
                    colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={CONF_LEVELS_2D[0]: r'1$\sigma$',
                       CONF_LEVELS_2D[1]: r'2$\sigma$',
                       CONF_LEVELS_2D[2]: r'3$\sigma$'})
    ax.contourf(X, Y, delta_chi2,
                levels=[0, *CONF_LEVELS_2D, max(delta_chi2.max(), CONF_LEVELS_2D[-1] + 1)],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    ax.plot(center[0], center[1], 'k*', ms=14, label='best fit')
    ax.set_xlabel(PARAM_LABELS[vary[0]])
    ax.set_ylabel(PARAM_LABELS[vary[1]])
    ax.set_title(rf'$\Delta\chi^2$ contours: {PARAM_LABELS[vary[0]]} vs {PARAM_LABELS[vary[1]]} '
                 rf'(Pantheon+ SNe, {PARAM_LABELS[PARAM_NAMES[iz]]} fixed)')
    ax.legend()
    fig.tight_layout()
    fname = f'contour_{vary[0]}_{vary[1]}_sn.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return X, Y, delta_chi2


def plot_confidence_ellipses_H0_Om(best_x, chi2_best, z_vals, mu_vals, inv_var,
                                   n_grid=CONTOUR_GRID, outdir='.'):
    """H0-Om confidence contours at the best-fit alpha."""
    H0_fit, Om_fit, alpha_fit = best_x
    H0_lo = max(BOUNDS[0][0], H0_fit * 0.85)
    H0_hi = min(BOUNDS[0][1], H0_fit * 1.15)
    Om_lo = max(BOUNDS[1][0], Om_fit * 0.5)
    Om_hi = min(BOUNDS[1][1], Om_fit * 2.0)

    H0_grid = np.linspace(H0_lo, H0_hi, n_grid)
    Om_grid = np.linspace(Om_lo, Om_hi, n_grid)
    H0_mesh, Om_mesh = np.meshgrid(H0_grid, Om_grid)

    params_flat = np.zeros((n_grid * n_grid, 3))
    params_flat[:, 0] = H0_mesh.ravel()
    params_flat[:, 1] = Om_mesh.ravel()
    params_flat[:, 2] = alpha_fit

    print("  Computing H0-Om confidence-ellipse grid...")
    chi2_flat = chi2_grid_diag(params_flat, z_vals, mu_vals, inv_var)
    CHI2_mesh = chi2_flat.reshape(n_grid, n_grid)
    delta_chi2 = CHI2_mesh - chi2_best

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(H0_mesh, Om_mesh, delta_chi2, levels=CONF_LEVELS_2D,
                    colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={CONF_LEVELS_2D[0]: r'1$\sigma$',
                       CONF_LEVELS_2D[1]: r'2$\sigma$',
                       CONF_LEVELS_2D[2]: r'3$\sigma$'})
    ax.contourf(H0_mesh, Om_mesh, delta_chi2,
                levels=[0, *CONF_LEVELS_2D, max(delta_chi2.max(), CONF_LEVELS_2D[-1] + 1)],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    ax.plot(H0_fit, Om_fit, 'k*', ms=14, label='best fit')
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_ylabel(r'$\Omega_{m,0}$')
    ax.set_title(rf'Confidence contours: $H_0$ vs $\Omega_{{m,0}}$ at $\alpha={alpha_fit:.3f}$ (SNe)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'confidence_ellipses_H0_Om_sn.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    return H0_mesh, Om_mesh, delta_chi2


# =============================================================================
# 7. HUBBLE DIAGRAM (distance-modulus version)
# =============================================================================

def plot_hubble_diagram_clean(best_x, z_vals, mu_vals, mu_err, outdir='.'):
    """Distance-modulus Hubble diagram with residuals."""
    H0_fit, Om_fit, alpha_fit = best_x
    z_smooth = np.linspace(max(z_vals.min() * 0.5, 1e-4), z_vals.max() * 1.05, 300)
    mu_smooth = mu_model(z_smooth, H0_fit, Om_fit, alpha_fit)
    mu_at_data = mu_model(z_vals, H0_fit, Om_fit, alpha_fit)
    residuals = mu_vals - mu_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, mu_vals, yerr=mu_err, fmt='o', color='crimson',
                 ms=2, alpha=0.35, capsize=0, label='Pantheon+SH0ES')
    ax1.plot(z_smooth, mu_smooth, color='navy', lw=2,
             label=rf'model fit ($\alpha={alpha_fit:.3f}$)')
    mu_lcdm_s = mu_lcdm(z_smooth, H0_fit, Om_fit)
    ax1.plot(z_smooth, mu_lcdm_s, color='green', lw=1.5, ls='--',
             label=r'$\Lambda$CDM (same $H_0,\Omega_{m,0}$, for reference)')
    ax1.set_ylabel(r'$\mu(z)$ [mag]')
    ax1.set_title(r'Hubble diagram (distance modulus): best-fit $\dot{H}$-$\alpha$ model')
    ax1.legend(fontsize=9)

    ax2.errorbar(z_vals, residuals, yerr=mu_err, fmt='o', color='crimson',
                 ms=2, alpha=0.35, capsize=0)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$\mu_{\rm obs}-\mu_{\rm model}$')
    ax2.set_ylim(-0.6, 0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MODEL COMPARISON TABLE
# =============================================================================

def create_model_comparison_table(best_x, chi2_best, z_vals, mu_vals, cov_inv,
                                  use_full_cov, inv_var, outdir='.'):
    """AIC / AICc / BIC comparison of the H_dot-alpha model vs pure LambdaCDM."""
    print("\n  Fitting LambdaCDM for comparison...")
    bounds_lcdm = [BOUNDS[0], (0.01, 0.99)]
    chi2_lcdm_fn = (lambda p, *a: chi2_lcdm_cov(p, *a)) if use_full_cov else \
                   (lambda p: chi2_diag([p[0], p[1], BOUNDS[2][0]], z_vals, mu_vals, inv_var))
    args_lcdm = (z_vals, mu_vals, cov_inv) if use_full_cov else ()

    try:
        de_lcdm = differential_evolution(
            chi2_lcdm_fn, bounds=bounds_lcdm, args=args_lcdm,
            seed=42, maxiter=150, tol=1e-7, polish=True, popsize=15,
        )
        H0_l, Om_l = de_lcdm.x
        chi2_lcdm_best = de_lcdm.fun
        res = minimize(chi2_lcdm_fn, [H0_l, Om_l], args=args_lcdm, method='Nelder-Mead',
                       bounds=bounds_lcdm,
                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 4000})
        if res.fun < chi2_lcdm_best:
            H0_l, Om_l = res.x
            chi2_lcdm_best = res.fun
        print(f"  LambdaCDM best fit: H0={H0_l:.2f}, Om={Om_l:.3f}, chi2={chi2_lcdm_best:.2f}")
    except Exception as e:
        print(f"  Warning: LambdaCDM fit failed: {e}")
        H0_l, Om_l = best_x[0], best_x[1]
        if use_full_cov:
            chi2_lcdm_best = chi2_lcdm_cov([H0_l, Om_l], z_vals, mu_vals, cov_inv)
        else:
            chi2_lcdm_best = chi2_diag([H0_l, Om_l, BOUNDS[2][0]], z_vals, mu_vals, inv_var)

    n = len(z_vals)
    dof_model = n - 3
    dof_lcdm = n - 2

    def stats(chi2_val, k, dof):
        aic = chi2_val + 2 * k
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else aic
        bic = chi2_val + k * np.log(n)
        chi2_dof = chi2_val / dof if dof > 0 else np.inf
        return aic, aicc, bic, chi2_dof

    aic_m, aicc_m, bic_m, chi2dof_m = stats(chi2_best, 3, dof_model)
    aic_l, aicc_l, bic_l, chi2dof_l = stats(chi2_lcdm_best, 2, dof_lcdm)

    table_data = [
        ['Model', 'H_dot-alpha', 'LambdaCDM'],
        ['H0', f'{best_x[0]:.2f}', f'{H0_l:.2f}'],
        ['Om,0', f'{best_x[1]:.3f}', f'{Om_l:.3f}'],
        ['alpha', f'{best_x[2]:.3f}', '-'],
        ['chi2', f'{chi2_best:.2f}', f'{chi2_lcdm_best:.2f}'],
        ['k', '3', '2'],
        ['dof', f'{dof_model}', f'{dof_lcdm}'],
        ['chi2/dof', f'{chi2dof_m:.3f}', f'{chi2dof_l:.3f}'],
        ['AIC', f'{aic_m:.2f}', f'{aic_l:.2f}'],
        ['dAIC', f'{aic_m - aic_l:+.2f}', '0 (ref)'],
        ['AICc', f'{aicc_m:.2f}', f'{aicc_l:.2f}'],
        ['dAICc', f'{aicc_m - aicc_l:+.2f}', '0 (ref)'],
        ['BIC', f'{bic_m:.2f}', f'{bic_l:.2f}'],
        ['dBIC', f'{bic_m - bic_l:+.2f}', '0 (ref)'],
    ]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE (Pantheon+SH0ES)")
    print("=" * 70)
    col_w = [max(len(r[i]) for r in table_data) + 2 for i in range(3)]
    for row in table_data:
        print("│" + "│".join(f"{row[i]:^{col_w[i]}}" for i in range(3)) + "│")
    print("=" * 70)

    fname = os.path.join(outdir, 'model_comparison_table_sn.txt')
    with open(fname, 'w') as f:
        f.write("MODEL COMPARISON TABLE (Pantheon+SH0ES)\n")
        f.write("=" * 70 + "\n")
        for row in table_data:
            f.write(f"{row[0]:<12} {row[1]:<18} {row[2]:<18}\n")
        f.write("=" * 70 + "\n")
        daic = aic_m - aic_l
        if daic < -2:
            f.write("H_dot-alpha model preferred by AIC\n")
        elif daic < 2:
            f.write("Models essentially equivalent by AIC\n")
        else:
            f.write("LambdaCDM preferred by AIC\n")
    print(f"\nModel comparison table saved to: {fname}")
    return table_data


# =============================================================================
# 9. EXPORTS & SUMMARY
# =============================================================================

def consistency_check_alpha_small(best_x, z_vals, mu_vals, inv_var):
    """
    As alpha shrinks, the correction term sits in a 1/alpha coefficient.
    NOTE: unlike delta_lcdm_sn.py's delta->0 check, alpha->0 does NOT
    recover LambdaCDM here (there is no explicit (1-Om) term in this model),
    so this is purely diagnostic: it shows how sensitive chi^2 is to alpha
    near the small-alpha edge of the prior, not a consistency cross-check.
    """
    H0_fit, Om_fit, _ = best_x
    print("\nBehaviour of chi^2 as alpha shrinks (H0, Om fixed at best fit):")
    for a in [1.0, 0.5, 0.2, 0.1, 0.05]:
        if a < BOUNDS[2][0]:
            continue
        c = chi2_diag([H0_fit, Om_fit, a], z_vals, mu_vals, inv_var)
        print(f"  alpha={a:<5} chi^2={c:.3f}")


def export_best_fit_data(z_vals, mu_vals, mu_err, best_x, outdir='.'):
    H0_fit, Om_fit, alpha_fit = best_x
    mu_best = mu_model(z_vals, H0_fit, Om_fit, alpha_fit)
    residuals = mu_vals - mu_best

    z_smooth = np.linspace(max(z_vals.min() * 0.5, 1e-4), z_vals.max() * 1.05, 250)
    mu_smooth = mu_model(z_smooth, H0_fit, Om_fit, alpha_fit)

    data_fname = os.path.join(outdir, 'H_dot_lcdm_sn_fit_results.txt')
    with open(data_fname, 'w') as f:
        f.write("# z, mu_obs, sigma_mu, mu_model, residual\n")
        for zi, mui, si, mm, ri in zip(z_vals, mu_vals, mu_err, mu_best, residuals):
            f.write(f"{zi:.6f} {mui:.6f} {si:.6f} {mm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_fname}")

    curve_fname = os.path.join(outdir, 'H_dot_lcdm_sn_smooth_curve.txt')
    with open(curve_fname, 'w') as f:
        f.write("# z, mu_model(z)\n")
        for zi, mi in zip(z_smooth, mu_smooth):
            f.write(f"{zi:.6f} {mi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_fname}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir="."):
    filename = os.path.join(outdir, "fit_summary_sn.txt")
    with open(filename, "w") as f:
        f.write("===== BEST FIT (Pantheon+SH0ES, H_dot-alpha model) =====\n\n")
        for n, v, e in zip(PARAM_NAMES, best_x, perr):
            f.write(f"{n:8s} = {v:.6f} +/- {e:.6f}\n")
        f.write(f"\nchi2     = {chi2_best:.4f}\n")
        f.write(f"dof      = {dof}\n")
        f.write(f"chi2/dof = {chi2_best/dof:.4f}\n")
        acc = np.mean(sampler.acceptance_fraction)
        f.write(f"\nAcceptance fraction = {acc:.4f}\n")
        try:
            tau = sampler.get_autocorr_time()
            f.write("\nAutocorrelation times\n")
            for n, t in zip(PARAM_NAMES, tau):
                f.write(f"{n:8s} {t:.2f}\n")
        except Exception:
            pass
        p = np.percentile(flat_samples, [16, 50, 84], axis=0)
        f.write("\n===== MCMC =====\n\n")
        for i, n in enumerate(PARAM_NAMES):
            lo, med, hi = p[:, i]
            f.write(f"{n:8s} = {med:.6f} (+{hi-med:.6f}/-{med-lo:.6f})\n")
    print(f"  Fit summary written to: {filename}")


def validate_config():
    assert CONTOUR_GRID >= 15
    assert PROFILE_POINTS >= 15
    assert NWALKERS >= 12
    assert NSTEPS >= 500
    assert BOUNDS[0][0] < BOUNDS[0][1]
    assert BOUNDS[1][0] < BOUNDS[1][1]
    assert BOUNDS[2][0] < BOUNDS[2][1]
    assert BOUNDS[2][0] > 0.0, "alpha must stay strictly positive (division by alpha in the model)"


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    validate_config()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results_hdot_sn")
    os.makedirs(outdir, exist_ok=True)
    print(f"Results will be saved to: {outdir}\n")

    setup_matplotlib()

    # --- Load data ---
    print("--- Loading Pantheon+SH0ES Type Ia SN Data ---")
    if not os.path.isabs(DATA_DIR):
        data_dir = os.path.join(script_dir, DATA_DIR)
    else:
        data_dir = DATA_DIR

    if not os.path.exists(data_dir):
        print(f"Critical Error: Data directory not found: {data_dir}")
        print("Please set DATA_DIR to the folder containing Pantheon+SH0ES.dat")
        return

    try:
        z_vals, mu_vals, mu_err, mask = load_sn_data(data_dir)
        cov = load_sn_covariance(data_dir, mask)
        print(f"\nSuccessfully loaded {len(z_vals)} SNe.")
        print(f"Redshift range: {z_vals.min():.4f} to {z_vals.max():.4f}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    use_full_cov = cov is not None
    if use_full_cov:
        cov_inv = np.linalg.inv(cov)
        print("  Using full stat+syst covariance matrix for likelihood.")
    else:
        cov_inv = np.diag(1.0 / mu_err ** 2)
        print("  Using diagonal mu errors only.")

    inv_var = 1.0 / mu_err ** 2

    # --- Best fit ---
    print("\n--- Best fit (global optimizer + multi-start) ---")
    best_x, chi2_best, converged = best_fit(
        z_vals, mu_vals, cov_inv, use_full_cov, inv_var
    )
    H0_fit, Om_fit, alpha_fit = best_x
    dof = len(z_vals) - 3
    print(f"  converged: {converged}")
    print(f"  H0    = {H0_fit:.4f}")
    print(f"  Om    = {Om_fit:.4f}")
    print(f"  alpha = {alpha_fit:.4f}")
    print(f"  chi^2 = {chi2_best:.4f}  (chi^2/dof = {chi2_best/dof:.4f}, dof={dof})")

    # --- curve_fit uncertainties ---
    print("\n--- curve_fit covariance (Gaussian uncertainties) ---")
    perr = np.array([np.nan, np.nan, np.nan])
    pcov = None
    try:
        popt, perr, pcov = fit_uncertainties_curvefit(
            z_vals, mu_vals, mu_err, best_x, use_full_cov, cov
        )
        for name, val, err in zip(PARAM_NAMES, popt, perr):
            print(f"  {name:6s} = {val:.4f} +/- {err:.4f}")
        if use_full_cov:
            chi2_cf = chi2_cov(popt, z_vals, mu_vals, cov_inv)
        else:
            chi2_cf = chi2_diag(popt, z_vals, mu_vals, inv_var)
        if chi2_cf < chi2_best:
            best_x, chi2_best = popt, chi2_cf
    except Exception as e:
        print(f"  curve_fit uncertainty estimation failed: {e}")

    # --- MCMC ---
    print("\n--- MCMC posterior (emcee) ---")
    sampler, flat_samples = run_mcmc(
        best_x, z_vals, mu_vals, cov_inv, use_full_cov, inv_var
    )
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = percentiles[:, i]
        print(f"  {name:6s} = {med:.4f} (+{hi-med:.4f} / -{med-lo:.4f})")

    plot_walkers(sampler, outdir)

    print("\n--- Corner plot ---")
    fig_corner = corner.corner(
        flat_samples, labels=[PARAM_LABELS[n] for n in PARAM_NAMES],
        truths=list(best_x), show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )
    fig_corner.savefig(os.path.join(outdir, 'corner_H_dot_lcdm_sn.png'),
                       dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    # --- Profile & contours (use diagonal for speed) ---
    print("\n--- Profile likelihood & confidence contours for alpha ---")
    chi2_best_diag = chi2_diag(best_x, z_vals, mu_vals, inv_var)
    plot_chi2_profile_alpha(best_x, chi2_best_diag, z_vals, mu_vals, inv_var, outdir=outdir)

    print("  Generating 2-D contours...")
    plot_contour_2d(best_x, chi2_best_diag, z_vals, mu_vals, inv_var,
                    vary=('alpha', 'Om'), outdir=outdir)
    plot_contour_2d(best_x, chi2_best_diag, z_vals, mu_vals, inv_var,
                    vary=('alpha', 'H0'), outdir=outdir)
    print("  Saved: chi2_profile_alpha_sn.png, contour_alpha_Om_sn.png, contour_alpha_H0_sn.png")

    print("\n--- Confidence ellipses at best alpha ---")
    H0_mesh, Om_mesh, dchi2 = plot_confidence_ellipses_H0_Om(
        best_x, chi2_best_diag, z_vals, mu_vals, inv_var, outdir=outdir
    )
    np.save(os.path.join(outdir, 'contour_H0_Om_hdot_sn.npy'),
            {'X': H0_mesh, 'Y': Om_mesh, 'delta_chi2': dchi2})
    print("  Saved: confidence_ellipses_H0_Om_sn.png")

    # --- Hubble diagram ---
    print("\n--- Hubble diagram ---")
    plot_hubble_diagram_clean(best_x, z_vals, mu_vals, mu_err, outdir=outdir)
    print("  Saved: hubble_diagram_sn.png")

    # --- Exports ---
    print("\n--- Export best-fit data ---")
    export_best_fit_data(z_vals, mu_vals, mu_err, best_x, outdir=outdir)

    consistency_check_alpha_small(best_x, z_vals, mu_vals, inv_var)

    print("\n--- Model Comparison Table ---")
    create_model_comparison_table(
        best_x, chi2_best, z_vals, mu_vals, cov_inv,
        use_full_cov, inv_var, outdir=outdir
    )

    if pcov is not None:
        corr = pcov / np.outer(perr, perr)
        print("\nCorrelation matrix")
        print("--------------------------------")
        for row in corr:
            print(" ".join(f"{x:8.3f}" for x in row))
        np.savetxt(os.path.join(outdir, "correlation_matrix_sn.txt"), corr, fmt="%.6f")

    write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir)

    print(f"\nDone. All figures and results saved to: {outdir}")


if __name__ == "__main__":
    main()