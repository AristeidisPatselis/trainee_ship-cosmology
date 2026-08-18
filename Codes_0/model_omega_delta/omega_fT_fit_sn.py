"""
omega_fT_sn.py
==============
Omega-parametrization f(T) gravity fit to Pantheon+SH0ES Type Ia supernovae,
following Anagnostopoulos, Basilakos & Saridakis 2019 (arXiv:1907.07533).

Three viable one-parameter f(T) models are implemented:

  1. f1CDM (power-law, Eq. 29):        y(z,b) = E(z,b)^(2b)
  2. f2CDM (sqrt-exponential, Eq. 33): y(z,b) = [1-(1+E^b)e^(-E/b)] /
                                                  [1-(1+1/b)e^(-1/b)]
  3. f3CDM (exponential, Eq. 37):      y(z,b) = [1-(1+2E^2/b)e^(-E^2/b)] /
                                                  [1-(1+2/b)e^(-1/b)]

Parameter order is (H0, Om, b) for every f(T) model.

The observable is now the Pantheon+SH0ES distance modulus
    mu(z) = 5 log10(d_L(z)/10 pc),
    d_L(z) = (1+z) * c * Integral_0^z dz'/H(z'),

so every likelihood evaluation requires solving the implicit E(z) equation
on a redshift grid and then numerically integrating 1/H(z).

Pipeline per model: global optimizer + multi-start polish, curve_fit
covariance, emcee MCMC, corner plot, 1D profile likelihood for b, 2D
delta-chi^2 contours, per-model Hubble diagram. Then model comparison table.
"""

# --- Standard library --------------------------------------------------------
import os
import warnings
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# IMPORTANT: Set BLAS threading before numpy import. This matters even more
# now that we run BLAS-light worker processes (multiprocessing Pool) for
# emcee/differential_evolution -- without this, each worker would try to
# spawn its own BLAS thread pool and they'd all fight over the same cores.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

# --- Numerics / optimization --------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.integrate import cumulative_trapezoid

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
H0_BOUNDS = (40.0, 100.0)
OM_BOUNDS = (0.01, 0.99)

# b-bounds are model-specific (same as omega_fT_fit.py)
MODEL_B_BOUNDS = {
    'f1CDM': (-1.5, 0.9),
    'f2CDM': (0.02, 3.0),
    'f3CDM': (0.02, 0.18),
}
MODEL_LABELS = {
    'f1CDM': 'Power-law $f(T)$ (f1CDM)',
    'f2CDM': r'$\sqrt{\ }$-exponential $f(T)$ (f2CDM)',
    'f3CDM': 'Exponential $f(T)$ (f3CDM)',
}
MODEL_COLORS = {
    'f1CDM': '#1f77b4',
    'f2CDM': '#ff7f0e',
    'f3CDM': '#2ca02c',
}
MODEL_LIST = ['f1CDM', 'f2CDM', 'f3CDM']

PARAM_NAMES = ['H0', 'Om', 'b']
PARAM_LABELS = {'H0': r'$H_0$', 'Om': r'$\Omega_{m,0}$', 'b': r'$b$'}

CONTOUR_GRID = 35
PROFILE_POINTS = 35
Z_GRID_POINTS = 500        # redshift resolution for the 1/H integral

# emcee sampler settings
NWALKERS = 32
NSTEPS = 2000
DISCARD = 400
THIN = 10

CONF_LEVELS_2D = [2.30, 6.18, 11.83]
CONF_LEVELS_1D = [1.0, 4.0, 9.0]

# --- OPTIMIZATION CONFIGURATION ---
ADAPTIVE_CONTOURS = True
BATCH_SIZE = 100
N_MULTISTART = 6

# Number of worker processes for emcee / differential_evolution parallelism.
# Leave one core free for the OS/plotting thread.
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Reference H0 values for diagnostic plotting
LITERATURE_H0 = {
    "Planck 2018 (CMB)": (67.4, 0.5),
    "SH0ES 2022 (Local)": (73.04, 1.04),
}


def get_bounds(model_name):
    return [H0_BOUNDS, OM_BOUNDS, MODEL_B_BOUNDS[model_name]]


# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

def setup_matplotlib(use_latex=False):
    """Enable LaTeX only if it actually succeeds on this machine."""
    if use_latex:
        try:
            rc('text', usetex=True)
            rc('font', family='serif')
            fig_test = plt.figure()
            plt.text(0.5, 0.5, r"$b$")
            fig_test.canvas.draw()
            plt.close(fig_test)
            return
        except Exception as e:
            print(f"Note: LaTeX rendering unavailable, using mathtext instead. ({e})")
    rc('text', usetex=False)
    rc('font', family='DejaVu Sans')


def find_file_recursively(filename, data_dir, max_depth=4):
    """Search for a file recursively in data_dir and its subdirectories."""
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
# 2. MODELS: three f(T) y(z,b) forms, each solved implicitly for E(z)
# =============================================================================

def y_vec(E, b, model_name):
    """
    Vectorized y(E,b) for all three f(T) models (Eqs. 29/33/37). `E` may be
    a scalar or an array; invalid inputs (E<=0, b=0, degenerate denominator)
    come back as NaN rather than raising, so callers can just check
    np.isfinite() once on the whole array.
    """
    E = np.asarray(E, dtype=float)
    with np.errstate(all='ignore'):
        valid = E > 0
        if model_name == 'f1CDM':
            y = np.where(valid, E ** (2.0 * b), np.nan)
        elif model_name == 'f2CDM':
            if b == 0:
                return np.full(E.shape, np.nan)
            den = 1.0 - (1.0 + 1.0 / b) * np.exp(-1.0 / b)
            if den == 0 or not np.isfinite(den):
                return np.full(E.shape, np.nan)
            num = 1.0 - (1.0 + E / b) * np.exp(-E / b)
            y = np.where(valid, num / den, np.nan)
        elif model_name == 'f3CDM':
            if b == 0:
                return np.full(E.shape, np.nan)
            den = 1.0 - (1.0 + 2.0 / b) * np.exp(-1.0 / b)
            if den == 0 or not np.isfinite(den):
                return np.full(E.shape, np.nan)
            E2 = E * E
            num = 1.0 - (1.0 + 2.0 * E2 / b) * np.exp(-E2 / b)
            y = np.where(valid, num / den, np.nan)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return y


def _E_curve_vectorized(z_arr, Om, b, model_name, tol=1e-9, max_iter=60):
    """
    Solve E(z) - sqrt(Om*(1+z)^3 + (1-Om)*y(E,b)) = 0 for the ENTIRE
    redshift grid at once, via a damped vectorized Newton iteration
    (numerical derivative).

    This replaces the previous approach of calling scipy.optimize.brentq
    once per redshift point in a Python for-loop. That loop dominated the
    runtime of every likelihood evaluation: a single mu_model() call solved
    ~500 independent 1D root-finding problems, each with several brentq
    iterations, all in pure Python. Since the underlying equation is smooth
    and (for the b-ranges these models allow) close to the analytic LCDM
    solution, a few vectorized Newton steps applied to the whole array
    converge to the same accuracy in a fraction of the time -- turning
    ~500 scalar Python calls into ~10-15 numpy array operations.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    onepz3 = Om * (1.0 + z_arr) ** 3

    def F(E):
        with np.errstate(all='ignore'):
            y = y_vec(E, b, model_name)
            inside = onepz3 + (1.0 - Om) * y
            inside = np.where(np.isfinite(inside) & (inside > 0), inside, np.nan)
            return E - np.sqrt(inside)

    # LambdaCDM E(z) is the exact solution at b=0 and a very good starting
    # guess for the small deviations these models allow.
    with np.errstate(all='ignore'):
        E = np.sqrt(np.maximum(onepz3 + (1.0 - Om), 1e-12))

    eps = 1e-6
    for _ in range(max_iter):
        Fv = F(E)
        Fv_eps = F(E + eps)
        with np.errstate(all='ignore'):
            deriv = (Fv_eps - Fv) / eps
            deriv = np.where(np.abs(deriv) > 1e-12, deriv, np.nan)
            step = np.where(np.isfinite(Fv) & np.isfinite(deriv), Fv / deriv, 0.0)
        E_new = E - step
        # keep the iterate positive/finite; shrink instead of exploding
        E_new = np.where(np.isfinite(E_new) & (E_new > 1e-8), E_new, E * 0.5)
        delta = np.nanmax(np.abs(E_new - E)) if E_new.size else 0.0
        E = E_new
        if np.isfinite(delta) and delta < tol:
            break

    Fv_final = F(E)
    with np.errstate(all='ignore'):
        E = np.where(np.abs(Fv_final) < 1e-6, E, np.nan)
    E = np.where(z_arr <= 0, 1.0, E)  # exact: E(z=0) = 1 for every b, model
    return E


def model_E(z_eval, Om, b, model_name):
    """Vectorized: solves for E=H/H0 at every z in z_eval simultaneously."""
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    return _E_curve_vectorized(z_eval, float(Om), float(b), model_name)


def model_H(z_eval, H0, Om, b, model_name):
    return H0 * model_E(z_eval, Om, b, model_name)


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


def mu_model(z, H0, Om, b, model_name, z_grid_points=Z_GRID_POINTS):
    """
    Distance modulus for the f(T) model at a single (H0, Om, b).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z_grid = _get_integration_grid(z, z_grid_points)

    H_grid = model_H(z_grid, H0, Om, b, model_name)
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


def H_lcdm(z, H0, Om):
    """Standard flat LambdaCDM, used as the baseline for AIC/BIC."""
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def mu_lcdm(z, H0, Om, z_grid_points=Z_GRID_POINTS):
    """Distance modulus for pure flat LambdaCDM (analytic H)."""
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

def _within_bounds(params, model_name):
    bounds = get_bounds(model_name)
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, bounds))


def chi2_diag(params, z_vals, mu_vals, inv_var, model_name):
    """Diagonal-error chi-squared (fast path for contours)."""
    H0, Om, b = params
    if not _within_bounds(params, model_name):
        return 1e12
    mu_th = mu_model(z_vals, H0, Om, b, model_name)
    if np.any(~np.isfinite(mu_th)):
        return 1e12
    return float(np.sum((mu_vals - mu_th) ** 2 * inv_var))


def chi2_cov(params, z_vals, mu_vals, cov_inv, model_name):
    """Full-covariance chi-squared: dmu^T C^{-1} dmu."""
    H0, Om, b = params
    if not _within_bounds(params, model_name):
        return 1e12
    mu_th = mu_model(z_vals, H0, Om, b, model_name)
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
    if np.any(~np.isfinite(mu_th)):
        return 1e12
    dmu = mu_vals - mu_th
    return float(dmu @ cov_inv @ dmu)


def chi2_dispatch(params, z_vals, mu_vals, cov_or_invvar, use_full_cov, model_name):
    """
    Picklable stand-in for the old `lambda p, *a: chi2_cov(p, *a, model_name)`
    closures. differential_evolution(workers=...) and emcee's Pool both need
    to pickle the objective/log-prob function to ship it to worker processes;
    lambdas and closures over local variables can't be pickled, so this has
    to be a plain module-level function with everything passed explicitly
    (via functools.partial for the fixed keyword arguments).
    """
    if use_full_cov:
        return chi2_cov(params, z_vals, mu_vals, cov_or_invvar, model_name)
    return chi2_diag(params, z_vals, mu_vals, cov_or_invvar, model_name)


def chi2_grid_diag(params_grid, z_vals, mu_vals, inv_var, model_name):
    """
    Chi-squared over a grid of (H0, Om, b) using diagonal errors.
    No vectorized batch model exists for the implicit f(T) case, so this loops.
    """
    n_grid = len(params_grid)
    chi2_vals = np.full(n_grid, 1e12)

    valid = np.array([_within_bounds(p, model_name) for p in params_grid])
    if not np.any(valid):
        return chi2_vals

    valid_params = params_grid[valid]
    chi2_vals_valid = np.zeros(len(valid_params))

    for i in tqdm(range(len(valid_params)), desc="  chi2 grid", leave=False):
        H0, Om, b = valid_params[i]
        mu_th = mu_model(z_vals, H0, Om, b, model_name)
        if np.all(np.isfinite(mu_th)) and not np.any(mu_th >= 1e6 - 1.0):
            chi2_vals_valid[i] = np.sum((mu_vals - mu_th) ** 2 * inv_var)
        else:
            chi2_vals_valid[i] = 1e12

    chi2_vals[valid] = chi2_vals_valid
    return chi2_vals


# =============================================================================
# 4. BEST FIT: global optimizer + multi-start cross-check
# =============================================================================

def best_fit(z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name,
             n_starts=N_MULTISTART, verbose=True):
    """Global fit (differential_evolution) followed by a Nelder-Mead polish."""
    # A picklable partial (not a lambda) so differential_evolution can
    # actually use multiple worker processes below.
    chi2_fn = partial(chi2_dispatch, use_full_cov=use_full_cov, model_name=model_name)
    args = (z_vals, mu_vals, cov_inv) if use_full_cov else (z_vals, mu_vals, inv_var)

    bounds = get_bounds(model_name)
    print(f"  [{model_name}] Running differential evolution "
          f"(workers={N_WORKERS})...")
    de_result = differential_evolution(
        chi2_fn, bounds=bounds, args=args,
        seed=42, maxiter=150, tol=1e-7, polish=True, popsize=15,
        workers=N_WORKERS, updating='deferred',
    )
    best_x, best_chi2 = de_result.x, de_result.fun

    print(f"  [{model_name}] Running {n_starts} multi-start local optimizations...")
    rng = np.random.default_rng(42)
    starts = [best_x] + [
        [rng.uniform(lo, hi) for (lo, hi) in bounds] for _ in range(n_starts)
    ]

    local_results = []
    for x0 in tqdm(starts, desc=f"  [{model_name}] Local optimizations", disable=not verbose):
        res = minimize(chi2_fn, x0, args=args, method='Nelder-Mead',
                       bounds=bounds,
                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 4000})
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        if spread.size:
            print(f"  [{model_name}] Multi-start scan: {len(spread)}/{len(starts)} runs converged "
                  f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
            if spread.max() - spread.min() > 1.0:
                print(f"  [{model_name}] -> spread across starts suggests a degenerate/"
                      "multi-modal chi^2 surface")

    return best_x, best_chi2, de_result.success


# =============================================================================
# 5. UNCERTAINTIES: curve_fit covariance + MCMC
# =============================================================================

def model_mu_curvefit(z_array, H0, Om, b, model_name):
    """curve_fit-friendly signature."""
    mu = mu_model(z_array, H0, Om, b, model_name)
    if np.any(~np.isfinite(mu)):
        return np.full_like(np.atleast_1d(z_array), 1e6, dtype=float)
    return mu


def fit_uncertainties_curvefit(z_vals, mu_vals, sigma, p0, use_full_cov, cov, model_name):
    lo = [b[0] for b in get_bounds(model_name)]
    hi = [b[1] for b in get_bounds(model_name)]
    
    def model_wrapper(z_array, H0, Om, b):
        return model_mu_curvefit(z_array, H0, Om, b, model_name)
    
    if use_full_cov and cov is not None:
        popt, pcov = curve_fit(
            model_wrapper, z_vals, mu_vals, p0=p0,
            sigma=cov, absolute_sigma=True, bounds=(lo, hi), maxfev=15000,
        )
    else:
        popt, pcov = curve_fit(
            model_wrapper, z_vals, mu_vals, p0=p0,
            sigma=sigma, absolute_sigma=True, bounds=(lo, hi), maxfev=15000,
        )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov


def log_prior(theta, model_name):
    bounds = get_bounds(model_name)
    for val, (lo, hi) in zip(theta, bounds):
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_likelihood(theta, z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name):
    if use_full_cov:
        c = chi2_cov(theta, z_vals, mu_vals, cov_inv, model_name)
    else:
        c = chi2_diag(theta, z_vals, mu_vals, inv_var, model_name)
    if c >= 1e11:
        return -np.inf
    return -0.5 * c


def log_prob(theta, z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name):
    lp = log_prior(theta, model_name)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name)


# --- LambdaCDM baseline prior/likelihood (module-level so a multiprocessing
# Pool can pickle them for the emcee sampler in run_lcdm_baseline) ---
LCDM_BOUNDS = [H0_BOUNDS, OM_BOUNDS]


def log_prior_lcdm(theta):
    H0, Om = theta
    if not (LCDM_BOUNDS[0][0] < H0 < LCDM_BOUNDS[0][1] and
            LCDM_BOUNDS[1][0] < Om < LCDM_BOUNDS[1][1]):
        return -np.inf
    return 0.0


def log_prob_lcdm(theta, z_vals, mu_vals, cov_inv):
    lp = log_prior_lcdm(theta)
    if not np.isfinite(lp):
        return -np.inf
    c = chi2_lcdm_cov(theta, z_vals, mu_vals, cov_inv)
    if c >= 1e11:
        return -np.inf
    return lp - 0.5 * c


def run_mcmc(best_x, z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name,
             nwalkers=NWALKERS, nsteps=NSTEPS, discard=DISCARD, thin=THIN):
    ndim = 3
    bounds = get_bounds(model_name)
    b_lo, b_hi = bounds[2]
    spread = np.array([2.0, 0.04, 0.1 * (b_hi - b_lo)])
    
    pos = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        pos[i] = best_x + spread * np.random.randn(ndim)
        for j, (lo, hi) in enumerate(bounds):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-6, hi - 1e-6)

    print(f"  [{model_name}] Running emcee ({nwalkers} walkers x {nsteps} steps, "
          f"{N_WORKERS} worker processes)...")
    if N_WORKERS > 1:
        with mp.Pool(processes=N_WORKERS) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_prob,
                args=(z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name),
                pool=pool,
            )
            sampler.run_mcmc(pos, nsteps, progress=True)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_prob,
            args=(z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name)
        )
        sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    return sampler, flat_samples


def plot_walkers(sampler, model_name, outdir="."):
    chain = sampler.get_chain()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        for walker in range(chain.shape[1]):
            axes[i].plot(chain[:, walker, i], alpha=0.3, lw=0.5)
        axes[i].set_ylabel(PARAM_LABELS[PARAM_NAMES[i]])
    axes[-1].set_xlabel("Step")
    axes[0].set_title(f"Walker chains: {MODEL_LABELS[model_name]}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"walker_chains_{model_name}.png"), dpi=300)
    plt.close(fig)


# =============================================================================
# 6. PROFILE LIKELIHOOD & CONFIDENCE CONTOURS
# =============================================================================

def plot_chi2_profile_b(best_x, chi2_best, z_vals, mu_vals, inv_var, model_name,
                         n_points=PROFILE_POINTS, outdir='.'):
    """1D profile chi^2(b): H0 and Om are re-fit at every b."""
    H0_fit, Om_fit, b_fit = best_x
    bounds = get_bounds(model_name)
    b_lo_bound, b_hi_bound = bounds[2]
    span = max(abs(b_fit) * 1.5, 0.3 * (b_hi_bound - b_lo_bound))
    b_lo = max(b_lo_bound, b_fit - span)
    b_hi = min(b_hi_bound, b_fit + span)
    bs = np.linspace(b_lo, b_hi, n_points)

    chi2_vals = np.empty(n_points)
    print(f"  [{model_name}] Computing profile likelihood for b...")
    for i, bb in enumerate(tqdm(bs, desc=f"  [{model_name}] b profile")):
        def chi2_reduced(p2):
            return chi2_diag([p2[0], p2[1], bb], z_vals, mu_vals, inv_var, model_name)
        res = minimize(chi2_reduced, [H0_fit, Om_fit], method='Nelder-Mead',
                       bounds=[bounds[0], bounds[1]])
        chi2_vals[i] = res.fun

    delta_chi2 = chi2_vals - chi2_best

    b_lo68 = b_hi68 = None
    below = delta_chi2 <= 1.0
    idx_below = np.where(below)[0]
    if idx_below.size:
        i_first, i_last = idx_below[0], idx_below[-1]
        if i_first > 0:
            b_lo68 = np.interp(1.0, [delta_chi2[i_first - 1], delta_chi2[i_first]],
                                [bs[i_first - 1], bs[i_first]])
        else:
            b_lo68 = bs[i_first]
        if i_last < n_points - 1:
            b_hi68 = np.interp(1.0, [delta_chi2[i_last + 1], delta_chi2[i_last]],
                                [bs[i_last + 1], bs[i_last]])
        else:
            b_hi68 = bs[i_last]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bs, delta_chi2, color='navy', lw=2)
    ax.axvline(b_fit, color='gray', ls=':', lw=1, label=f'best fit ({b_fit:.3f})')
    ax.axvline(0.0, color='crimson', ls='--', lw=1.2, label=r'$\Lambda$CDM ($b=0$)')
    if b_lo68 is not None and b_hi68 is not None:
        ax.axvspan(b_lo68, b_hi68, color='navy', alpha=0.12, label=r'1$\sigma$ interval')
    for level, label in zip(CONF_LEVELS_1D, [r'1$\sigma$', r'2$\sigma$', r'3$\sigma$']):
        ax.axhline(level, color='gray', ls='--', lw=0.8)
        ax.text(bs[-1], level, label, va='bottom', ha='right', fontsize=9, color='gray')
    ax.set_xlabel(r'$b$')
    ax.set_ylabel(r'$\Delta\chi^2(b)$')
    ax.set_title(rf'Profile likelihood: $\Delta\chi^2$ vs $b$ -- {MODEL_LABELS[model_name]} (SNe)')
    ax.set_ylim(0, 12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'chi2_profile_b_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if b_lo68 is not None and b_hi68 is not None:
        print(f"  [{model_name}] b 1sigma profile interval: [{b_lo68:.4f}, {b_hi68:.4f}]")

    return bs, chi2_vals


def _param_range(center, bounds, rel_span=0.6, min_abs_span=0.3):
    """Sign-safe symmetric range around `center`, clipped to `bounds`."""
    span = max(abs(center) * rel_span, min_abs_span)
    lo = max(bounds[0], center - span)
    hi = min(bounds[1], center + span)
    if lo >= hi:
        lo, hi = bounds
    return lo, hi


def plot_contour_2d(best_x, chi2_best, z_vals, mu_vals, inv_var, model_name,
                     vary=('b', 'Om'), n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour via a grid evaluation."""
    idx = {'H0': 0, 'Om': 1, 'b': 2}
    ix, iy = idx[vary[0]], idx[vary[1]]
    iz = ({0, 1, 2} - {ix, iy}).pop()
    bounds = get_bounds(model_name)

    center = best_x[ix], best_x[iy]
    x_lo, x_hi = _param_range(center[0], bounds[ix])
    y_lo, y_hi = _param_range(center[1], bounds[iy])

    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)

    params_flat = np.zeros((n_grid * n_grid, 3))
    params_flat[:, ix] = X.ravel()
    params_flat[:, iy] = Y.ravel()
    params_flat[:, iz] = best_x[iz]

    print(f"  [{model_name}] Computing {n_grid}x{n_grid} grid for {vary[0]}-{vary[1]} contour...")
    chi2_flat = chi2_grid_diag(params_flat, z_vals, mu_vals, inv_var, model_name)
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
    if vary[0] == 'b' or vary[1] == 'b':
        if vary[1] == 'b':
            ax.axhline(0, color='crimson', ls='--', lw=1.2, label=r'$\Lambda$CDM ($b=0$)')
        else:
            ax.axvline(0, color='crimson', ls='--', lw=1.2, label=r'$\Lambda$CDM ($b=0$)')
    ax.plot(center[0], center[1], 'k*', ms=14, label='best fit')
    ax.set_xlabel(PARAM_LABELS[vary[0]])
    ax.set_ylabel(PARAM_LABELS[vary[1]])
    ax.set_title(rf'$\Delta\chi^2$ contours: {PARAM_LABELS[vary[0]]} vs {PARAM_LABELS[vary[1]]} '
                 rf'-- {MODEL_LABELS[model_name]} (SNe)')
    ax.legend()
    fig.tight_layout()

    fname = f'contour_{vary[0]}_{vary[1]}_{model_name}.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return X, Y, delta_chi2


def plot_confidence_ellipses_H0_Om(best_x, chi2_best, z_vals, mu_vals, inv_var, model_name,
                                   n_grid=CONTOUR_GRID, outdir='.'):
    """H0-Om confidence contours at the best-fit b."""
    H0_fit, Om_fit, b_fit = best_x
    H0_lo = max(H0_BOUNDS[0], H0_fit * 0.85)
    H0_hi = min(H0_BOUNDS[1], H0_fit * 1.15)
    Om_lo = max(OM_BOUNDS[0], Om_fit * 0.5)
    Om_hi = min(OM_BOUNDS[1], Om_fit * 2.0)

    H0_grid = np.linspace(H0_lo, H0_hi, n_grid)
    Om_grid = np.linspace(Om_lo, Om_hi, n_grid)
    H0_mesh, Om_mesh = np.meshgrid(H0_grid, Om_grid)

    params_flat = np.zeros((n_grid * n_grid, 3))
    params_flat[:, 0] = H0_mesh.ravel()
    params_flat[:, 1] = Om_mesh.ravel()
    params_flat[:, 2] = b_fit

    print(f"  [{model_name}] Computing H0-Om confidence-ellipse grid...")
    chi2_flat = chi2_grid_diag(params_flat, z_vals, mu_vals, inv_var, model_name)
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
    
    # Add literature H0 values for reference
    for name, (h_val, h_err) in LITERATURE_H0.items():
        ax.axvline(h_val, color='gray', ls=':', alpha=0.5, lw=1)
        ax.text(h_val, ax.get_ylim()[1], name, rotation=90,
                va='top', ha='right', fontsize=7, alpha=0.6)
    
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_ylabel(r'$\Omega_{m,0}$')
    ax.set_title(rf'Confidence contours: $H_0$ vs $\Omega_{{m,0}}$ at $b={b_fit:.3f}$ '
                 rf'-- {MODEL_LABELS[model_name]} (SNe)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'confidence_ellipses_H0_Om_{model_name}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    return H0_mesh, Om_mesh, delta_chi2


def adaptive_contour_if_needed(best_x, chi2_best, z_vals, mu_vals, inv_var, model_name,
                                vary=('b', 'Om'), n_grid_min=25, n_grid_max=50, outdir='.'):
    if not ADAPTIVE_CONTOURS:
        return plot_contour_2d(best_x, chi2_best, z_vals, mu_vals, inv_var, model_name,
                                vary=vary, n_grid=CONTOUR_GRID, outdir=outdir)

    n_grid = n_grid_min
    X, Y, delta = plot_contour_2d(best_x, chi2_best, z_vals, mu_vals, inv_var, model_name,
                                   vary=vary, n_grid=n_grid, outdir=outdir)

    grad_x = np.gradient(delta, axis=0)
    grad_y = np.gradient(delta, axis=1)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    if np.std(grad_mag) > 0.5 * np.mean(grad_mag) and n_grid < n_grid_max:
        n_grid = min(n_grid * 2, n_grid_max)
        print(f"  [{model_name}] Refining contour grid to {n_grid}x{n_grid}...")
        X, Y, delta = plot_contour_2d(best_x, chi2_best, z_vals, mu_vals, inv_var, model_name,
                                       vary=vary, n_grid=n_grid, outdir=outdir)

    return X, Y, delta


# =============================================================================
# 7. HUBBLE DIAGRAM (distance-modulus version)
# =============================================================================

def plot_hubble_diagram_clean(best_x, z_vals, mu_vals, mu_err, model_name, outdir='.'):
    """Distance-modulus Hubble diagram with residuals."""
    H0_fit, Om_fit, b_fit = best_x
    z_smooth = np.linspace(max(z_vals.min() * 0.5, 1e-4), z_vals.max() * 1.05, 300)
    mu_smooth = mu_model(z_smooth, H0_fit, Om_fit, b_fit, model_name)
    mu_at_data = mu_model(z_vals, H0_fit, Om_fit, b_fit, model_name)
    residuals = mu_vals - mu_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, mu_vals, yerr=mu_err, fmt='o', color='crimson',
                 ms=2, alpha=0.35, capsize=0, label='Pantheon+SH0ES')
    ax1.plot(z_smooth, mu_smooth, color='navy', lw=2,
             label=rf'model fit ($b={b_fit:.3f}$)')
    mu_lcdm_s = mu_lcdm(z_smooth, H0_fit, Om_fit)
    ax1.plot(z_smooth, mu_lcdm_s, color='green', lw=1.5, ls='--',
             label=r'$\Lambda$CDM ($b=0$, same $H_0,\Omega_{m,0}$)')
    ax1.set_ylabel(r'$\mu(z)$ [mag]')
    ax1.set_title(f'Hubble diagram (distance modulus): best fit -- {MODEL_LABELS[model_name]}')
    ax1.legend(fontsize=9)

    ax2.errorbar(z_vals, residuals, yerr=mu_err, fmt='o', color='crimson',
                 ms=2, alpha=0.35, capsize=0)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$\mu_{\rm obs}-\mu_{\rm model}$')
    ax2.set_ylim(-0.6, 0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'hubble_diagram_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_hubble_diagram_all_models(results, z_vals, mu_vals, mu_err, lcdm_fit, outdir='.'):
    """Overlay all three f(T) models + LambdaCDM on one Hubble diagram."""
    z_smooth = np.linspace(max(z_vals.min() * 0.5, 1e-4), z_vals.max() * 1.05, 300)
    H0_l, Om_l = lcdm_fit

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, mu_vals, yerr=mu_err, fmt='o', color='k',
                 ms=2, alpha=0.25, capsize=0, label='Pantheon+SH0ES', zorder=5)

    mu_lcdm_s = mu_lcdm(z_smooth, H0_l, Om_l)
    ax1.plot(z_smooth, mu_lcdm_s, color='gray', lw=2, ls='--', label=r'$\Lambda$CDM')
    ax2.axhline(0, color='gray', lw=1.5, ls='--')

    for model_name in MODEL_LIST:
        best_x = results[model_name]['best_x']
        H0_fit, Om_fit, b_fit = best_x
        mu_smooth = mu_model(z_smooth, H0_fit, Om_fit, b_fit, model_name)
        mu_at_data = mu_model(z_vals, H0_fit, Om_fit, b_fit, model_name)
        ax1.plot(z_smooth, mu_smooth, color=MODEL_COLORS[model_name], lw=2,
                  label=f'{model_name} ($b={b_fit:.3f}$)')
        ax2.plot(z_vals, mu_vals - mu_at_data, 'o', color=MODEL_COLORS[model_name], ms=3, alpha=0.5)

    ax1.set_ylabel(r'$\mu(z)$ [mag]')
    ax1.set_title('Hubble diagram: all f(T) models vs $\\Lambda$CDM (Pantheon+SH0ES)')
    ax1.legend(fontsize=9)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$\mu_{\rm obs}-\mu_{\rm model}$')
    ax2.set_ylim(-0.6, 0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram_all_models_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_H0_tension_comparison(model_name, best_x, perr, outdir='.'):
    """Compare this fit's H0 against literature reference values."""
    H0_fit = best_x[0]
    H0_err = perr[0] if perr is not None and np.isfinite(perr[0]) else 0.0

    all_vals = {f"This work ({model_name}, SNe)": (H0_fit, H0_err, "crimson")}
    for name, (val, err) in LITERATURE_H0.items():
        all_vals[name] = (val, err, "steelblue" if "Planck" in name else "darkorange")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(all_vals.items()):
        ax.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)
        ax.axvspan(val - err, val + err, color=color, alpha=0.1)
    ax.set_yticks(range(len(all_vals)))
    ax.set_yticklabels(all_vals.keys())
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_title(f'$H_0$: {model_name} fit (SNe) vs. literature')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'H0_tension_comparison_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MODEL COMPARISON TABLE
# =============================================================================

def compute_dic(flat_samples, chi2_func, chi2_args):
    """DIC = D(theta_bar) + 2*pD, pD = D_bar - D(theta_bar), D = chi^2"""
    D_samples = np.array([chi2_func(theta, *chi2_args) for theta in flat_samples])
    D_bar = np.mean(D_samples)
    theta_bar = np.mean(flat_samples, axis=0)
    D_hat = chi2_func(theta_bar, *chi2_args)
    pD = D_bar - D_hat
    dic = D_hat + 2 * pD
    return dic, pD


def create_model_comparison_table(results, lcdm_result, n, outdir='.'):
    """results: dict model_name -> {'best_x','chi2_best','flat_samples'}
    lcdm_result: {'best_x','chi2_best','flat_samples'} for flat LambdaCDM."""
    
    rows = []
    all_names = MODEL_LIST + ['LambdaCDM']
    stats = {}

    for name in all_names:
        if name == 'LambdaCDM':
            k = 2
            chi2_best = lcdm_result['chi2_best']
            dic, pD = compute_dic(lcdm_result['flat_samples'], chi2_lcdm_cov,
                                   (lcdm_result['z_vals'], lcdm_result['mu_vals'], 
                                    lcdm_result['cov_inv']))
            dof = n - k
        else:
            k = 3
            chi2_best = results[name]['chi2_best']
            z_vals = results[name]['z_vals']
            mu_vals = results[name]['mu_vals']
            cov_inv = results[name]['cov_inv']
            def chi2_wrapper(theta, zv, mv, ci):
                return chi2_cov(theta, zv, mv, ci, name)
            dic, pD = compute_dic(results[name]['flat_samples'], chi2_wrapper,
                                   (z_vals, mu_vals, cov_inv))
            dof = n - k
        
        aic = chi2_best + 2 * k
        aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else aic
        bic = chi2_best + k * np.log(n)
        stats[name] = dict(k=k, dof=dof, chi2=chi2_best, chi2_dof=chi2_best/dof,
                            aic=aic, aicc=aicc, bic=bic, dic=dic, pD=pD)

    aic_min = min(s['aic'] for s in stats.values())
    bic_min = min(s['bic'] for s in stats.values())
    dic_min = min(s['dic'] for s in stats.values())

    print("\n" + "=" * 110)
    print("MODEL COMPARISON TABLE (Omega-parametrization f(T) models vs LambdaCDM, Pantheon+SH0ES)")
    print("=" * 110)
    header = (f"{'Model':<12}{'k':>4}{'dof':>6}{'chi2':>10}{'chi2/dof':>11}"
              f"{'AIC':>10}{'dAIC':>9}{'AICc':>10}{'dAICc':>9}{'BIC':>10}"
              f"{'dBIC':>9}{'DIC':>10}{'dDIC':>9}")
    print(header)
    print("-" * 110)
    lines = [header, "-" * 110]
    for name in all_names:
        s = stats[name]
        d_aic = s['aic'] - aic_min
        d_bic = s['bic'] - bic_min
        d_dic = s['dic'] - dic_min
        d_aicc = s['aicc'] - stats[all_names[0]]['aicc']
        row = (f"{name:<12}{s['k']:>4}{s['dof']:>6}{s['chi2']:>10.3f}{s['chi2_dof']:>11.3f}"
               f"{s['aic']:>10.3f}{d_aic:>9.3f}{s['aicc']:>10.3f}{d_aicc:>9.3f}"
               f"{s['bic']:>10.3f}{d_bic:>9.3f}{s['dic']:>10.3f}{d_dic:>9.3f}")
        print(row)
        lines.append(row)
    print("=" * 110)
    lines.append("=" * 110)

    lines.append("\nJeffreys-scale interpretation (paper's convention): "
                 "dIC<=2 statistically indistinguishable from the best model, "
                 "2<dIC<6 mild tension, dIC>=10 strong tension.")

    filename = os.path.join(outdir, 'model_comparison_table_sn.txt')
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nModel comparison table saved to: {filename}")

    return stats


# =============================================================================
# 9. CONSISTENCY CHECK: b -> 0 recovers LambdaCDM
# =============================================================================

def consistency_check_b_zero(best_x, z_vals, mu_vals, inv_var, model_name):
    """Evaluates chi^2 as b shrinks toward the LambdaCDM limit."""
    H0_fit, Om_fit, b_fit = best_x
    print(f"\n[{model_name}] Behaviour of chi^2 as b -> 0 (H0, Om fixed at best fit):")
    if model_name == 'f1CDM':
        fracs = [1.0, 0.5, 0.2, 0.1, 0.0]
    else:
        fracs = [1.0, 0.5, 0.2, 0.1, 0.01]
    for frac in fracs:
        bb = b_fit * frac if frac > 0 else 0.0
        mu_th = mu_model(z_vals, H0_fit, Om_fit, bb, model_name)
        if np.any(~np.isfinite(mu_th)) or np.any(mu_th >= 1e6 - 1.0):
            print(f"  b={bb:<9.4f} chi^2=  (invalid: model broke down at this b)")
            continue
        c = float(np.sum(((mu_vals - mu_th) / np.sqrt(inv_var)) ** 2))
        print(f"  b={bb:<9.4f} chi^2={c:.3f}")
    c_lcdm_direct = float(np.sum(((mu_vals - mu_lcdm(z_vals, H0_fit, Om_fit)) * np.sqrt(inv_var)) ** 2))
    print(f"  [cross-check] direct LambdaCDM chi^2 at same (H0,Om): {c_lcdm_direct:.3f} "
          f"(should be approached as b -> 0 above)")


# =============================================================================
# 10. EXPORTS & SUMMARY
# =============================================================================

def export_best_fit_data(z_vals, mu_vals, mu_err, best_x, model_name, outdir='.'):
    H0_fit, Om_fit, b_fit = best_x
    mu_best = mu_model(z_vals, H0_fit, Om_fit, b_fit, model_name)
    residuals = mu_vals - mu_best

    z_smooth = np.linspace(max(z_vals.min() * 0.5, 1e-4), z_vals.max() * 1.05, 250)
    mu_smooth = mu_model(z_smooth, H0_fit, Om_fit, b_fit, model_name)

    data_filename = os.path.join(outdir, f'{model_name}_fit_results_sn.txt')
    with open(data_filename, 'w') as f:
        f.write("# z, mu_obs, sigma_mu, mu_model, residual\n")
        for zi, mui, si, mm, ri in zip(z_vals, mu_vals, mu_err, mu_best, residuals):
            f.write(f"{zi:.6f} {mui:.6f} {si:.6f} {mm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_filename}")

    curve_filename = os.path.join(outdir, f'{model_name}_smooth_curve_sn.txt')
    with open(curve_filename, 'w') as f:
        f.write("# z, mu_model(z)\n")
        for zi, mi in zip(z_smooth, mu_smooth):
            f.write(f"{zi:.6f} {mi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_filename}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, model_name, outdir="."):
    filename = os.path.join(outdir, f"fit_summary_{model_name}_sn.txt")
    with open(filename, "w") as f:
        f.write(f"===== BEST FIT: {model_name} (Pantheon+SH0ES) =====\n\n")
        for n_, v, e in zip(PARAM_NAMES, best_x, perr):
            f.write(f"{n_:8s} = {v:.6f} +/- {e:.6f}\n")
        f.write("\n")
        f.write(f"chi2        = {chi2_best:.4f}\n")
        f.write(f"dof         = {dof}\n")
        f.write(f"chi2/dof    = {chi2_best/dof:.4f}\n")
        f.write("\n")
        acc = np.mean(sampler.acceptance_fraction)
        f.write(f"Acceptance fraction = {acc:.4f}\n")
        try:
            tau = sampler.get_autocorr_time()
            f.write("\nAutocorrelation times\n")
            for n_, t in zip(PARAM_NAMES, tau):
                f.write(f"{n_:8s} {t:.2f}\n")
        except Exception:
            pass
        f.write("\n")
        p = np.percentile(flat_samples, [16, 50, 84], axis=0)
        f.write("===== MCMC =====\n\n")
        for i, n_ in enumerate(PARAM_NAMES):
            lo, med, hi = p[:, i]
            f.write(f"{n_:8s} = {med:.6f} (+{hi-med:.6f}/-{med-lo:.6f})\n")
    print(f"  Fit summary written to: {filename}")


def validate_config():
    assert CONTOUR_GRID >= 15
    assert PROFILE_POINTS >= 15
    assert NWALKERS >= 12
    assert NSTEPS >= 500
    assert H0_BOUNDS[0] < H0_BOUNDS[1], "Invalid H0 bounds"
    assert OM_BOUNDS[0] < OM_BOUNDS[1], "Invalid Om bounds"
    for name, (lo, hi) in MODEL_B_BOUNDS.items():
        assert lo < hi, f"Invalid b bounds for {name}"


# =============================================================================
# 11. PER-MODEL PIPELINE
# =============================================================================

def run_one_model(model_name, z_vals, mu_vals, mu_err, cov_inv, use_full_cov, inv_var, base_outdir):
    outdir = os.path.join(base_outdir, model_name)
    os.makedirs(outdir, exist_ok=True)
    print("\n" + "#" * 78)
    print(f"# MODEL: {MODEL_LABELS[model_name]} (Pantheon+SH0ES)")
    print("#" * 78)

    print(f"\n--- [{model_name}] Best fit (global optimizer + multi-start) ---")
    best_x, chi2_best, converged = best_fit(z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name)
    H0_fit, Om_fit, b_fit = best_x
    dof = len(z_vals) - 3
    print(f"  converged: {converged}")
    print(f"  H0 = {H0_fit:.4f}   Om = {Om_fit:.4f}   b = {b_fit:.4f}")
    print(f"  chi^2 = {chi2_best:.4f}  (chi^2/dof = {chi2_best/dof:.4f}, dof={dof})")

    print(f"\n--- [{model_name}] curve_fit covariance ---")
    perr = None
    try:
        popt, perr, pcov = fit_uncertainties_curvefit(z_vals, mu_vals, mu_err, best_x, use_full_cov, cov_inv, model_name)
        for name, val, err in zip(PARAM_NAMES, popt, perr):
            print(f"  {name:6s} = {val:.4f} +/- {err:.4f}")
        chi2_cf = chi2_cov(popt, z_vals, mu_vals, cov_inv, model_name)
        if chi2_cf < chi2_best:
            best_x, chi2_best = popt, chi2_cf
    except Exception as e:
        print(f"  curve_fit uncertainty estimation failed: {e}")
        perr = np.full(3, np.nan)

    print(f"\n--- [{model_name}] MCMC posterior (emcee) ---")
    sampler, flat_samples = run_mcmc(best_x, z_vals, mu_vals, cov_inv, use_full_cov, inv_var, model_name)
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = percentiles[:, i]
        print(f"  {name:6s} = {med:.4f} (+{hi-med:.4f} / -{med-lo:.4f})")

    plot_walkers(sampler, model_name, outdir)

    print(f"\n--- [{model_name}] Corner plot ---")
    fig_corner = corner.corner(flat_samples, labels=[PARAM_LABELS[n] for n in PARAM_NAMES],
                                truths=list(best_x), show_titles=True,
                                quantiles=[0.16, 0.5, 0.84])
    fig_corner.suptitle(f"{MODEL_LABELS[model_name]} (Pantheon+SH0ES)", y=1.02)
    fig_corner.savefig(os.path.join(outdir, f'corner_{model_name}_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    # Recompute chi2_best with diagonal errors for consistent contour comparison
    chi2_best_diag = chi2_diag(best_x, z_vals, mu_vals, inv_var, model_name)

    print(f"\n--- [{model_name}] Profile likelihood & contours for b ---")
    plot_chi2_profile_b(best_x, chi2_best_diag, z_vals, mu_vals, inv_var, model_name, outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best_diag, z_vals, mu_vals, inv_var, model_name,
                                vary=('b', 'Om'), outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best_diag, z_vals, mu_vals, inv_var, model_name,
                                vary=('b', 'H0'), outdir=outdir)
    
    print(f"\n--- [{model_name}] Confidence ellipses at best b ---")
    H0_mesh, Om_mesh, dchi2 = plot_confidence_ellipses_H0_Om(
        best_x, chi2_best_diag, z_vals, mu_vals, inv_var, model_name, outdir=outdir
    )
    np.save(os.path.join(outdir, f'contour_H0_Om_{model_name}_sn.npy'),
            {'X': H0_mesh, 'Y': Om_mesh, 'delta_chi2': dchi2})

    print(f"\n--- [{model_name}] Hubble diagram ---")
    plot_hubble_diagram_clean(best_x, z_vals, mu_vals, mu_err, model_name, outdir=outdir)
    plot_H0_tension_comparison(model_name, best_x, perr, outdir=outdir)

    print(f"\n--- [{model_name}] Export best-fit data ---")
    export_best_fit_data(z_vals, mu_vals, mu_err, best_x, model_name, outdir=outdir)

    consistency_check_b_zero(best_x, z_vals, mu_vals, inv_var, model_name)

    write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, model_name, outdir=outdir)

    print(f"\n[{model_name}] Done. Outputs in: {outdir}")

    return {
        'best_x': best_x, 'chi2_best': chi2_best, 'flat_samples': flat_samples,
        'z_vals': z_vals, 'mu_vals': mu_vals, 'cov_inv': cov_inv,
    }


# =============================================================================
# 12. LCDM BASELINE
# =============================================================================

def run_lcdm_baseline(z_vals, mu_vals, cov_inv, outdir):
    """Quick LambdaCDM fit + MCMC for the comparison table."""
    print("\n" + "#" * 78)
    print("# BASELINE: flat LambdaCDM (Pantheon+SH0ES)")
    print("#" * 78)

    bounds = LCDM_BOUNDS

    print(f"  Running differential evolution (workers={N_WORKERS})...")
    de_result = differential_evolution(
        chi2_lcdm_cov, bounds=bounds, args=(z_vals, mu_vals, cov_inv),
        seed=42, maxiter=150, tol=1e-7, polish=True, popsize=15,
        workers=N_WORKERS, updating='deferred',
    )
    H0_l, Om_l = de_result.x
    chi2_l = de_result.fun
    print(f"  H0 = {H0_l:.4f}   Om = {Om_l:.4f}   chi^2 = {chi2_l:.4f}")

    ndim, nwalkers = 2, 24
    spread = np.array([2.0, 0.03])
    pos = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        pos[i] = np.array([H0_l, Om_l]) + spread * np.random.randn(ndim)
        pos[i, 0] = np.clip(pos[i, 0], bounds[0][0] + 1e-6, bounds[0][1] - 1e-6)
        pos[i, 1] = np.clip(pos[i, 1], bounds[1][0] + 1e-6, bounds[1][1] - 1e-6)

    print(f"  Running emcee for LambdaCDM baseline ({N_WORKERS} worker processes)...")
    if N_WORKERS > 1:
        with mp.Pool(processes=N_WORKERS) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_prob_lcdm, args=(z_vals, mu_vals, cov_inv), pool=pool,
            )
            sampler.run_mcmc(pos, 2000, progress=True)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_lcdm, args=(z_vals, mu_vals, cov_inv))
        sampler.run_mcmc(pos, 2000, progress=True)
    flat_samples = sampler.get_chain(discard=400, thin=10, flat=True)

    fig_corner = corner.corner(flat_samples, labels=[r'$H_0$', r'$\Omega_{m,0}$'],
                                truths=[H0_l, Om_l], show_titles=True)
    fig_corner.savefig(os.path.join(outdir, 'corner_LambdaCDM_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    return {
        'best_x': np.array([H0_l, Om_l]), 'chi2_best': chi2_l, 'flat_samples': flat_samples,
        'z_vals': z_vals, 'mu_vals': mu_vals, 'cov_inv': cov_inv,
    }


# =============================================================================
# 13. MAIN
# =============================================================================

def main():
    validate_config()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results_omega_fT_sn")
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

    results = {}
    for model_name in MODEL_LIST:
        results[model_name] = run_one_model(
            model_name, z_vals, mu_vals, mu_err, cov_inv, use_full_cov, inv_var, outdir
        )

    lcdm_result = run_lcdm_baseline(z_vals, mu_vals, cov_inv, outdir)

    print("\n--- Combined Hubble diagram (all models) ---")
    plot_hubble_diagram_all_models(results, z_vals, mu_vals, mu_err, lcdm_result['best_x'], outdir=outdir)

    print("\n--- Model comparison table (f1CDM, f2CDM, f3CDM, LambdaCDM) ---")
    stats = create_model_comparison_table(results, lcdm_result, len(z_vals), outdir=outdir)

    print(f"\nDone. All figures and results saved to: {outdir}")
    print("\nSummary of best-fit b (distortion parameter) per model (Pantheon+SH0ES):")
    for model_name in MODEL_LIST:
        b_fit = results[model_name]['best_x'][2]
        print(f"  {model_name:8s}: b = {b_fit:.4f}  (b=0 is the LambdaCDM limit)")


if __name__ == "__main__":
    main()