"""
delta_alpha_lcdm_sn.py
=======================
Combined modified Friedmann equation, delta AND alpha both FREE, fit to
Pantheon+SH0ES Type Ia supernovae:

    H(z)^2 = H0^2 [ Om*(1+z)^3 + (1-Om)*(H(z)/H0)^delta ]  -  alpha*(1+z)*H*dH/dz

Free parameters: (H0, Om, delta, alpha). delta=0 collapses the explicit
dark-energy term to the ordinary constant (1-Om) LambdaCDM term (still
combined with the alpha correction); alpha=0 collapses the correction term
entirely, leaving the plain delta-model of delta_lcdm_sn.py. Neither limit
is enforced -- both are searched freely within their prior box.

This is the SN analogue of delta_alpha_lcdm_fit.py (which fits cosmic-
chronometer H(z) data directly). The underlying algebraic model -- same
u = H^2 substitution, same ODE, same lru_cache'd solve_ivp -- is copied
over unchanged; only the observable changes, from H(z) itself to the
Pantheon+SH0ES distance modulus

    mu(z) = 5 log10(d_L(z)/10 pc),
    d_L(z) = (1+z) * c * Integral_0^z dz'/H(z'),

so every likelihood evaluation now needs an H(z) ODE solve over a fixed
redshift grid *plus* a 1/H(z) integration on top of it. Structurally this
script mirrors delta_lcdm_sn.py / H_dot_lcdm_sn.py so the whole "chronometer
+ SN" family stays comparable line-by-line.

--------------------------------------------------------------------------
Why there is no vectorized batch model here (unlike the other SN scripts)
--------------------------------------------------------------------------
delta_lcdm_sn.py could batch its implicit H(z) solve with a vectorized
Newton iteration, and H_dot_lcdm_sn.py has an exact closed-form H(z). With
BOTH delta and alpha free, the u(z) ODE has no closed form and no simple
algebraic root to Newton-iterate on -- it's a genuine ODE, solved exactly
the way delta_alpha_lcdm_fit.py solves it (solve_ivp + lru_cache, keyed on
the redshift grid and parameters). Grid scans (contours, chi2_grid_diag)
therefore fall back to the same "loop over parameter sets in chunks"
approach the original chronometer script uses -- there just isn't a
vectorized alternative for the fully general case. This makes this script
the slowest of the SN family, exactly as delta_alpha_lcdm_fit.py is the
slowest of the chronometer family. Start with small CONFIG knobs (below)
while debugging; bump them up for the "real" run.
--------------------------------------------------------------------------
"""

# --- Standard library --------------------------------------------------------
import os

# IMPORTANT: must be set before numpy/scipy are imported. Every chi^2 call
# does a (1701x1701) matrix-vector product (dmu @ cov_inv @ dmu) via BLAS.
# BLAS defaults to using every core it can see -- fine in a single process,
# but fatal once combined with USE_PARALLEL's multiprocessing.Pool: each of
# the N worker processes ALSO tries to grab multiple BLAS threads, so N
# processes x many threads each fight over the same physical cores. This
# oversubscription is the main reason the MCMC run can crawl or appear to
# hang -- individual evaluations that take ~0.01-0.05s in isolation can
# balloon to <1s or worse under contention. Each worker process only needs
# 1 BLAS thread since the parallelism already comes from the process pool.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import warnings
from functools import lru_cache
from tqdm import tqdm

# --- Numerics / optimization --------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.integrate import solve_ivp, cumulative_trapezoid
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
# Prior/search box for (H0, Om, delta, alpha). Also used as hard bounds: any
# point outside gets chi^2 = +inf / log-prob = -inf.
BOUNDS = [(40.0, 100.0), (0.01, 0.99), (-2.0, 6.0), (0.01, 6.0)]   # H0, Om, delta, alpha
PARAM_NAMES = ['H0', 'Om', 'delta', 'alpha']
PARAM_LABELS = {'H0': r'$H_0$', 'Om': r'$\Omega_{m,0}$',
                'delta': r'$\delta$', 'alpha': r'$\alpha$'}

# NOTE TO SELF: this is the slowest script in the SN family -- every chi^2
# evaluation does a real ODE solve, not a vectorized/closed-form lookup.
# Keep these knobs small while debugging (CONTOUR_GRID~20-25, NSTEPS~1000),
# bump up for the "real" run.
CONTOUR_GRID = 25         # cost per contour plot is CONTOUR_GRID^2 ODE solves
PROFILE_POINTS = 25
Z_GRID_POINTS = 200        # REDUCED: 400 -> 200 for faster integration

# Hard ceiling on RHS evaluations for a single H(z) ODE solve. As alpha -> 0
# the ODE (dudz ~ 1/alpha) becomes stiff and LSODA can burn through an
# effectively unbounded number of internal steps shrinking its stepsize --
# this is what causes the optimizer/MCMC to appear to hang on certain
# (delta, alpha) points instead of erroring out. Once the budget is
# exceeded we abort that one solve and treat the point like any other
# invalid one (chi^2 = 1e12 / log-prob = -inf), rather than blocking.
MAX_RHS_EVALS = 10000      # REDUCED: 20000 -> 10000

# emcee sampler settings - REDUCED for faster runtime
NWALKERS = 32              # REDUCED: 48 -> 32
NSTEPS = 1500              # REDUCED: 3000 -> 1500
DISCARD = 300              # REDUCED: 600 -> 300
THIN = 10                  # REDUCED: 15 -> 10

CONF_LEVELS_2D = [2.30, 6.18, 11.83]
CONF_LEVELS_1D = [1.0, 4.0, 9.0]

# --- OPTIMIZATION CONFIGURATION ---
USE_CACHING = True
USE_PARALLEL = True       # ODE-based likelihood genuinely benefits from
                          # multiprocessing here (unlike the closed-form
                          # H_dot_lcdm_sn.py MCMC).
BATCH_SIZE = 100
N_MULTISTART = 4           # REDUCED: 6 -> 4

# Reference H0 values from the literature for diagnostic plotting
LITERATURE_H0 = {
    "Planck 2018 (CMB)": (67.4, 0.5),
    "SH0ES 2022 (Local)": (73.04, 1.04),
}


# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

def setup_matplotlib(use_latex=False):
    """
    use_latex=True spawns a separate latex+dvipng subprocess for every
    distinct piece of plot text. Leave this False unless you specifically
    need LaTeX-rendered labels; mathtext covers everything used here
    (\\delta, \\alpha, \\Omega_{m,0}, etc.).
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
# 2. MODEL: implicit H(z) via the u = H^2 substitution (with caching)
# =============================================================================
#
# Starting equation:
#   H^2 = H0^2*[Om*(1+z)^3 + (1-Om)*(H/H0)^delta]  -  alpha*(1+z)*H*dH/dz
#
# Substitute u = H^2  =>  du/dz = 2*H*dH/dz, so H*dH/dz = (1/2) du/dz:
#
#   => du/dz = [2 / (alpha*(1+z))] *
#              ( H0^2*Om*(1+z)^3 + H0^2*(1-Om)*(u/H0^2)^(delta/2)  -  u )
#
# Same ODE as delta_alpha_lcdm_fit.py, unchanged. Boundary condition:
# z=0, H=H0  =>  u(0) = H0^2.

class _IntegrationBudgetExceeded(Exception):
    """Raised when a single H(z) ODE solve exceeds MAX_RHS_EVALS RHS calls.

    This happens for stiff/near-singular (delta, alpha) points -- most
    often as alpha -> 0, where dudz ~ 1/alpha blows up and LSODA keeps
    shrinking its step size without ever giving up on its own. Without
    this guard, solve_ivp can run for minutes on a single parameter
    point during DE/multi-start/MCMC exploration, which looks like a
    hang rather than an error.
    """
    pass


def _rhs_u(z, u, H0, Om, delta, alpha):
    u_safe = min(max(u[0], 1e-8), 1e12)
    x = 1.0 + z
    de_term = H0 ** 2 * (1 - Om) * (u_safe / H0 ** 2) ** (delta / 2.0)
    dudz = (2.0 / (alpha * x)) * (H0 ** 2 * Om * x ** 3 + de_term - u_safe)
    return [np.clip(dudz, -1e12, 1e12)]


def _make_budgeted_rhs(H0, Om, delta, alpha, max_evals=MAX_RHS_EVALS):
    """Wrap _rhs_u with an RHS-call counter; raises _IntegrationBudgetExceeded
    past max_evals so a pathological solve fails fast instead of hanging."""
    n_calls = [0]

    def rhs(z, u):
        n_calls[0] += 1
        if n_calls[0] > max_evals:
            raise _IntegrationBudgetExceeded(
                f"RHS eval budget ({max_evals}) exceeded for "
                f"H0={H0:.4g}, Om={Om:.4g}, delta={delta:.4g}, alpha={alpha:.4g} "
                f"-- treating as an invalid point."
            )
        return _rhs_u(z, u, H0, Om, delta, alpha)

    return rhs


@lru_cache(maxsize=512)
def _model_H_cached(z_tuple, H0, Om, delta, alpha):
    """Cached version of model_H for repeated calls with the same params.

    Called with z_tuple = the fixed integration grid (see
    _get_integration_grid), so within one script run the same grid is
    reused across every call for a given (H0, Om, delta, alpha) -- e.g.
    the same point being touched twice inside an optimizer step.
    """
    z_eval = np.array(z_tuple)
    if alpha <= 0 or H0 <= 0 or not (0 < Om < 1):
        return tuple([np.nan] * len(z_eval))

    z_max = max(z_eval.max(), 1e-6)
    t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))

    try:
        sol = solve_ivp(
            _make_budgeted_rhs(H0, Om, delta, alpha), (0.0, z_max), [H0 ** 2],
            t_eval=t_eval,
            method='LSODA',        # handles the stiffness small alpha causes
            rtol=1e-6, atol=1e-8,  # RELAXED: 1e-8/1e-10 -> 1e-6/1e-8 for speed
            max_step=0.1,          # INCREASED: 0.05 -> 0.1 for fewer steps
        )
    except _IntegrationBudgetExceeded:
        return tuple([np.nan] * len(z_eval))
    except Exception:
        return tuple([np.nan] * len(z_eval))

    if not sol.success:
        return tuple([np.nan] * len(z_eval))

    u_of_z = np.interp(z_eval, sol.t, sol.y[0])
    if np.any(~np.isfinite(u_of_z)) or np.any(u_of_z <= 0):
        return tuple([np.nan] * len(z_eval))
    return tuple(np.sqrt(u_of_z))


def model_H(z_eval, H0, Om, delta, alpha):
    """Solve the ODE for u=H^2 and return H(z) at the requested redshifts."""
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))

    if USE_CACHING:
        z_tuple = tuple(z_eval)
        result = _model_H_cached(z_tuple, float(H0), float(Om), float(delta), float(alpha))
        return np.array(result)

    if alpha <= 0 or H0 <= 0 or not (0 < Om < 1):
        return np.full_like(z_eval, np.nan)

    z_max = max(z_eval.max(), 1e-6)
    t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))

    try:
        sol = solve_ivp(
            _make_budgeted_rhs(H0, Om, delta, alpha), (0.0, z_max), [H0 ** 2],
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-6, atol=1e-8,  # RELAXED for speed
            max_step=0.1,          # INCREASED for speed
        )
    except _IntegrationBudgetExceeded:
        return np.full_like(z_eval, np.nan)
    except Exception:
        return np.full_like(z_eval, np.nan)

    if not sol.success:
        return np.full_like(z_eval, np.nan)

    u_of_z = np.interp(z_eval, sol.t, sol.y[0])
    if np.any(~np.isfinite(u_of_z)) or np.any(u_of_z <= 0):
        return np.full_like(z_eval, np.nan)
    return np.sqrt(u_of_z)


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


def mu_model(z, H0, Om, delta, alpha, z_grid_points=Z_GRID_POINTS):
    """
    Distance modulus for the delta+alpha model at a single (H0, Om, delta,
    alpha). Solves the H(z) ODE once over the fixed integration grid (via
    the lru_cache'd model_H), then integrates c/H(z) to get d_L(z).
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z_grid = _get_integration_grid(z, z_grid_points)

    H_grid = model_H(z_grid, H0, Om, delta, alpha)
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
    """Diagonal-error chi-squared (fast-ish path for contours)."""
    H0, Om, delta, alpha = params
    if not _within_bounds(params):
        return 1e12
    mu_th = mu_model(z_vals, H0, Om, delta, alpha)
    if np.any(~np.isfinite(mu_th)):
        return 1e12
    return float(np.sum((mu_vals - mu_th) ** 2 * inv_var))


def chi2_cov(params, z_vals, mu_vals, cov_inv):
    """Full-covariance chi-squared: dmu^T C^{-1} dmu."""
    H0, Om, delta, alpha = params
    if not _within_bounds(params):
        return 1e12
    mu_th = mu_model(z_vals, H0, Om, delta, alpha)
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
    Chi-squared over a grid of (H0, Om, delta, alpha) points, using
    diagonal errors. No vectorized batch model exists for this general
    (delta and alpha both free) case, so -- exactly like the original
    chronometer script's chi2_grid -- this loops over valid grid points in
    chunks, calling mu_model (and therefore one ODE solve) per point.
    """
    n_grid = len(params_grid)
    chi2_vals = np.full(n_grid, 1e12)

    valid = np.array([_within_bounds(p) for p in params_grid])
    if not np.any(valid):
        return chi2_vals

    valid_params = params_grid[valid]
    chi2_vals_valid = np.zeros(len(valid_params))

    for i in tqdm(range(len(valid_params)), desc="  chi2 grid", leave=False):
        H0, Om, delta, alpha = valid_params[i]
        mu_th = mu_model(z_vals, H0, Om, delta, alpha)
        if np.all(np.isfinite(mu_th)) and not np.any(mu_th >= 1e6 - 1.0):
            chi2_vals_valid[i] = np.sum((mu_vals - mu_th) ** 2 * inv_var)
        else:
            chi2_vals_valid[i] = 1e12

    chi2_vals[valid] = chi2_vals_valid
    return chi2_vals


# =============================================================================
# 4. BEST FIT: global optimizer + multi-start cross-check
# =============================================================================

def best_fit(z_vals, mu_vals, cov_inv, use_full_cov, inv_var,
             n_starts=N_MULTISTART, verbose=True):
    """Global fit (differential_evolution) followed by a Nelder-Mead polish,
    plus an independent multi-start Nelder-Mead scan as a cross-check."""
    chi2_fn = (lambda p, *a: chi2_cov(p, *a)) if use_full_cov else \
              (lambda p, *a: chi2_diag(p, *a))
    args = (z_vals, mu_vals, cov_inv) if use_full_cov else (z_vals, mu_vals, inv_var)

    print("  Running differential evolution...")
    de_result = differential_evolution(
        chi2_fn, bounds=BOUNDS, args=args,
        seed=42, maxiter=100, tol=1e-7, polish=True, popsize=12,  # REDUCED maxiter, popsize
    )
    best_x, best_chi2 = de_result.x, de_result.fun

    print(f"  Running {n_starts} multi-start local optimizations...")
    rng = np.random.default_rng(42)
    starts = [best_x] + [
        [rng.uniform(lo, hi) for (lo, hi) in BOUNDS] for _ in range(n_starts)
    ]

    local_results = []
    for x0 in tqdm(starts, desc="  Local optimizations", disable=not verbose):
        res = minimize(chi2_fn, x0, args=args, method='Nelder-Mead', bounds=BOUNDS,
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})  # REDUCED maxiter
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        if spread.size:
            print(f"  Multi-start scan: {spread.size}/{len(starts)} runs converged "
                  f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
            if spread.max() - spread.min() > 1.0:
                print("  -> spread across starts suggests a degenerate/multi-modal "
                      "chi^2 surface (with 4 free params and (delta, alpha) both "
                      "controlling the late-time behaviour, expect strong "
                      "degeneracies -- see the contour plots); trust the global "
                      "(differential_evolution) result.")
        else:
            print("  Multi-start scan: no local run converged to a finite chi^2 "
                  "(relying on the differential_evolution result alone).")

    return best_x, best_chi2, de_result.success


# =============================================================================
# 5. UNCERTAINTIES: curve_fit covariance + MCMC
# =============================================================================

def model_mu_curvefit(z_array, H0, Om, delta, alpha):
    """curve_fit-friendly signature."""
    mu = mu_model(z_array, H0, Om, delta, alpha)
    if np.any(~np.isfinite(mu)):
        return np.full_like(np.atleast_1d(z_array), 1e6, dtype=float)
    return mu


def fit_uncertainties_curvefit(z_vals, mu_vals, sigma, p0, use_full_cov, cov):
    lo = [b[0] for b in BOUNDS]
    hi = [b[1] for b in BOUNDS]
    if use_full_cov and cov is not None:
        popt, pcov = curve_fit(
            model_mu_curvefit, z_vals, mu_vals, p0=p0,
            sigma=cov, absolute_sigma=True, bounds=(lo, hi), maxfev=20000,  # REDUCED maxfev
        )
    else:
        popt, pcov = curve_fit(
            model_mu_curvefit, z_vals, mu_vals, p0=p0,
            sigma=sigma, absolute_sigma=True, bounds=(lo, hi), maxfev=20000,  # REDUCED maxfev
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
    ndim = 4
    spread = np.array([2.0, 0.03, 0.3, 0.15])   # small Gaussian ball around best fit

    pos = np.tile(best_x, (nwalkers, 1)).astype(float)
    for w in range(nwalkers):
        scale = spread.copy()
        for _ in range(50):
            candidate = best_x + scale * np.random.randn(ndim)
            for j, (lo, hi) in enumerate(BOUNDS):
                candidate[j] = np.clip(candidate[j], lo + 1e-6, hi - 1e-6)
            if np.isfinite(log_prob(candidate, z_vals, mu_vals, cov_inv, use_full_cov, inv_var)):
                pos[w] = candidate
                break
            scale *= 0.5   # shrink and retry if we keep landing at -inf

    pool = None
    if USE_PARALLEL:
        try:
            import multiprocessing
            n_cpus = multiprocessing.cpu_count()
            # REDUCED: use fewer cores to avoid oversubscription (half of available)
            n_threads = max(1, min(n_cpus // 2, nwalkers // 2))
            if n_threads > 1:
                pool = multiprocessing.Pool(processes=n_threads)
                print(f"  Using {n_threads} CPU cores for MCMC (reduced to avoid oversubscription)")
        except Exception as e:
            print(f"  Parallel processing not available: {e}")

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=(z_vals, mu_vals, cov_inv, use_full_cov, inv_var),
        pool=pool
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    if pool is not None:
        pool.close()

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    return sampler, flat_samples


def plot_walkers(sampler, outdir="."):
    """Plot MCMC walker traces for all 4 parameters."""
    chain = sampler.get_chain()
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for i in range(4):
        for walker in range(chain.shape[1]):
            axes[i].plot(chain[:, walker, i], alpha=0.3, lw=0.5)
        axes[i].set_ylabel(PARAM_LABELS[PARAM_NAMES[i]])
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "walker_chains_delta_alpha_sn.png"), dpi=300)
    plt.close()


# =============================================================================
# 6. PROFILE LIKELIHOOD & CONFIDENCE CONTOURS
# =============================================================================

def plot_chi2_profile_1d(param_name, best_x, chi2_best, z_vals, mu_vals, inv_var,
                          n_points=PROFILE_POINTS, outdir='.'):
    """1D profile chi^2(param): the other 3 parameters are re-fit at every
    grid value of `param_name` (diagonal errors)."""
    idx = PARAM_NAMES.index(param_name)
    other_idx = [i for i in range(4) if i != idx]
    p_fit = best_x[idx]
    p_lo_bound, p_hi_bound = BOUNDS[idx]

    if p_fit >= 0:
        p_lo = max(p_lo_bound, p_fit * 0.3 if p_fit > 0 else p_lo_bound)
        p_hi = min(p_hi_bound, p_fit * 2.5 if p_fit > 0 else p_hi_bound)
    else:
        p_lo = max(p_lo_bound, p_fit * 2.5)
        p_hi = min(p_hi_bound, p_fit * 0.3)
    if p_hi - p_lo < 1e-3:
        pad = 0.5 * (p_hi_bound - p_lo_bound) * 0.2
        p_lo, p_hi = max(p_lo_bound, p_fit - pad), min(p_hi_bound, p_fit + pad)

    grid = np.linspace(p_lo, p_hi, n_points)
    chi2_vals = np.empty(n_points)
    print(f"  Computing profile likelihood for {param_name}...")
    for i, val in enumerate(tqdm(grid, desc=f"  {param_name} profile")):
        def chi2_reduced(p_other):
            full = np.empty(4)
            full[idx] = val
            for k, oi in enumerate(other_idx):
                full[oi] = p_other[k]
            return chi2_diag(full, z_vals, mu_vals, inv_var)
        x0 = [best_x[oi] for oi in other_idx]
        bnds = [BOUNDS[oi] for oi in other_idx]
        res = minimize(chi2_reduced, x0, method='Nelder-Mead', bounds=bnds,
                       options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 3000})
        chi2_vals[i] = res.fun

    delta_chi2 = chi2_vals - chi2_best

    p_lo68 = p_hi68 = None
    below = delta_chi2 <= 1.0
    idx_below = np.where(below)[0]
    if idx_below.size:
        i_first, i_last = idx_below[0], idx_below[-1]
        if i_first > 0:
            p_lo68 = np.interp(1.0, [delta_chi2[i_first - 1], delta_chi2[i_first]],
                                [grid[i_first - 1], grid[i_first]])
        else:
            p_lo68 = grid[i_first]
        if i_last < n_points - 1:
            p_hi68 = np.interp(1.0, [delta_chi2[i_last + 1], delta_chi2[i_last]],
                                [grid[i_last + 1], grid[i_last]])
        else:
            p_hi68 = grid[i_last]

    label = PARAM_LABELS[param_name]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(grid, delta_chi2, color='navy', lw=2)
    ax.axvline(p_fit, color='gray', ls=':', lw=1, label=f'best fit ({p_fit:.3f})')
    if param_name == 'delta':
        ax.axvline(0.0, color='crimson', ls='--', lw=1.2, label=r'$\delta=0$')
    if p_lo68 is not None and p_hi68 is not None:
        ax.axvspan(p_lo68, p_hi68, color='navy', alpha=0.12, label=r'1$\sigma$ interval')
    for level, lvl_label in zip(CONF_LEVELS_1D, [r'1$\sigma$', r'2$\sigma$', r'3$\sigma$']):
        ax.axhline(level, color='gray', ls='--', lw=0.8)
        ax.text(grid[-1], level, lvl_label, va='bottom', ha='right', fontsize=9, color='gray')
    ax.set_xlabel(label)
    ax.set_ylabel(rf'$\Delta\chi^2({param_name})$')
    ax.set_title(rf'Profile likelihood: $\Delta\chi^2$ vs {label} (Pantheon+ SNe, other 3 refit)')
    ax.set_ylim(0, 12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f'chi2_profile_{param_name}_sn.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if p_lo68 is not None and p_hi68 is not None:
        print(f"  {param_name} 1sigma profile interval: [{p_lo68:.4f}, {p_hi68:.4f}]")

    return grid, chi2_vals


def plot_contour_2d(best_x, chi2_best, z_vals, mu_vals, inv_var,
                    vary=('delta', 'alpha'), n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour in two of the four parameters, with the other
    two held fixed at their best-fit values (diagonal errors)."""
    idx = {name: i for i, name in enumerate(PARAM_NAMES)}
    ix, iy = idx[vary[0]], idx[vary[1]]
    fixed_idx = [i for i in range(4) if i not in (ix, iy)]

    center = best_x[ix], best_x[iy]

    def scan_bounds(i):
        c = best_x[i]
        lo_b, hi_b = BOUNDS[i]
        if c > 0:
            lo, hi = max(lo_b, c * 0.3), min(hi_b, c * 2.2)
        else:
            lo, hi = max(lo_b, c * 2.2), min(hi_b, c * 0.3 if c < 0 else hi_b)
        if hi - lo < 1e-3:
            pad = 0.2 * (hi_b - lo_b)
            lo, hi = max(lo_b, c - pad), min(hi_b, c + pad)
        return lo, hi

    x_lo, x_hi = scan_bounds(ix)
    y_lo, y_hi = scan_bounds(iy)
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)

    params_flat = np.zeros((n_grid * n_grid, 4))
    params_flat[:, ix] = X.ravel()
    params_flat[:, iy] = Y.ravel()
    for fi in fixed_idx:
        params_flat[:, fi] = best_x[fi]

    print(f"  Computing {n_grid}x{n_grid} grid for {vary[0]}-{vary[1]} contour...")
    chi2_flat = chi2_grid_diag(params_flat, z_vals, mu_vals, inv_var)
    CHI2 = chi2_flat.reshape(n_grid, n_grid)
    delta_chi2 = CHI2 - chi2_best

    fixed_desc = ", ".join(f"{PARAM_LABELS[PARAM_NAMES[fi]]} fixed" for fi in fixed_idx)

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(X, Y, delta_chi2, levels=CONF_LEVELS_2D,
                    colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={CONF_LEVELS_2D[0]: r'1$\sigma$',
                       CONF_LEVELS_2D[1]: r'2$\sigma$',
                       CONF_LEVELS_2D[2]: r'3$\sigma$'})
    ax.contourf(X, Y, delta_chi2,
                levels=[0, *CONF_LEVELS_2D, max(delta_chi2.max(), CONF_LEVELS_2D[-1] + 1)],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    if vary[0] == 'delta' or vary[1] == 'delta':
        if vary[1] == 'delta':
            ax.axhline(0, color='crimson', ls='--', lw=1.2, label=r'$\delta=0$')
        else:
            ax.axvline(0, color='crimson', ls='--', lw=1.2, label=r'$\delta=0$')
    ax.plot(center[0], center[1], 'k*', ms=14, label='best fit')
    ax.set_xlabel(PARAM_LABELS[vary[0]])
    ax.set_ylabel(PARAM_LABELS[vary[1]])
    ax.set_title(rf'$\Delta\chi^2$ contours: {PARAM_LABELS[vary[0]]} vs {PARAM_LABELS[vary[1]]} '
                 rf'(Pantheon+ SNe, {fixed_desc})')
    ax.legend()
    fig.tight_layout()
    fname = f'contour_{vary[0]}_{vary[1]}_sn.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return X, Y, delta_chi2


def plot_contour_H0_Om_fixed_delta_alpha(best_x, chi2_best, z_vals, mu_vals, inv_var,
                                         n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour for H0 vs Om, with delta and alpha fixed at their
    best-fit values."""
    H0_fit, Om_fit, delta_fit, alpha_fit = best_x

    h_lo = max(BOUNDS[0][0], H0_fit * 0.85)
    h_hi = min(BOUNDS[0][1], H0_fit * 1.15)
    o_lo = max(BOUNDS[1][0], Om_fit * 0.7)
    o_hi = min(BOUNDS[1][1], Om_fit * 1.3)

    if h_hi - h_lo < 5:
        h_lo = max(BOUNDS[0][0], H0_fit - 5)
        h_hi = min(BOUNDS[0][1], H0_fit + 5)
    if o_hi - o_lo < 0.05:
        o_lo = max(BOUNDS[1][0], Om_fit - 0.05)
        o_hi = min(BOUNDS[1][1], Om_fit + 0.05)

    H0_grid = np.linspace(h_lo, h_hi, n_grid)
    Om_grid = np.linspace(o_lo, o_hi, n_grid)
    X, Y = np.meshgrid(H0_grid, Om_grid)

    params_flat = np.zeros((n_grid * n_grid, 4))
    params_flat[:, 0] = X.ravel()
    params_flat[:, 1] = Y.ravel()
    params_flat[:, 2] = delta_fit
    params_flat[:, 3] = alpha_fit

    print(f"  Computing {n_grid}x{n_grid} grid for H0-Om contour "
          f"(delta={delta_fit:.3f}, alpha={alpha_fit:.3f} fixed)...")
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
    ax.plot(H0_fit, Om_fit, 'k*', ms=14, label='best fit')

    for name, (h_val, h_err) in LITERATURE_H0.items():
        ax.axvline(h_val, color='gray', ls=':', alpha=0.5, lw=1)
        ax.text(h_val, ax.get_ylim()[1], name, rotation=90,
                va='top', ha='right', fontsize=7, alpha=0.6)

    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_ylabel(r'$\Omega_{m,0}$')
    ax.set_title(rf'$\Delta\chi^2$ contours: $H_0$ vs $\Omega_{{m,0}}$ (SNe, '
                 rf'$\delta={delta_fit:.3f}$, $\alpha={alpha_fit:.3f}$ fixed)')
    ax.legend()
    fig.tight_layout()
    fname = 'contour_H0_Om_fixed_delta_alpha_sn.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {fname}")
    return X, Y, delta_chi2


def adaptive_contour_if_needed(best_x, chi2_best, z_vals, mu_vals, inv_var,
                                vary=('delta', 'alpha'),
                                n_grid_min=20, n_grid_max=40, outdir='.'):
    """Adaptive contour plotting with variable resolution."""
    n_grid = n_grid_min
    X, Y, dchi2 = plot_contour_2d(
        best_x, chi2_best, z_vals, mu_vals, inv_var,
        vary=vary, n_grid=n_grid, outdir=outdir
    )

    grad_x = np.gradient(dchi2, axis=0)
    grad_y = np.gradient(dchi2, axis=1)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    if np.std(grad_mag) > 0.5 * np.mean(grad_mag) and n_grid < n_grid_max:
        n_grid = min(n_grid * 2, n_grid_max)
        print(f"  Refining contour grid to {n_grid}x{n_grid}...")
        X, Y, dchi2 = plot_contour_2d(
            best_x, chi2_best, z_vals, mu_vals, inv_var,
            vary=vary, n_grid=n_grid, outdir=outdir
        )

    return X, Y, dchi2


# =============================================================================
# 7. HUBBLE DIAGRAM (distance-modulus version)
# =============================================================================

def plot_hubble_diagram_clean(best_x, z_vals, mu_vals, mu_err, outdir='.'):
    """Distance-modulus Hubble diagram with residuals."""
    H0_fit, Om_fit, delta_fit, alpha_fit = best_x
    z_smooth = np.linspace(max(z_vals.min() * 0.5, 1e-4), z_vals.max() * 1.05, 250)
    mu_smooth = mu_model(z_smooth, H0_fit, Om_fit, delta_fit, alpha_fit)
    mu_at_data = mu_model(z_vals, H0_fit, Om_fit, delta_fit, alpha_fit)
    residuals = mu_vals - mu_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, mu_vals, yerr=mu_err, fmt='o', color='crimson',
                 ms=2, alpha=0.35, capsize=0, label='Pantheon+SH0ES')
    ax1.plot(z_smooth, mu_smooth, color='navy', lw=2,
             label=rf'model fit ($\delta={delta_fit:.3f}$, $\alpha={alpha_fit:.3f}$)')
    mu_lcdm_s = mu_lcdm(z_smooth, H0_fit, Om_fit)
    ax1.plot(z_smooth, mu_lcdm_s, color='green', lw=1.5, ls='--',
             label=r'$\Lambda$CDM (same $H_0,\Omega_{m,0}$, for reference)')
    ax1.set_ylabel(r'$\mu(z)$ [mag]')
    ax1.set_title(r'Hubble diagram (distance modulus): best fit ($\delta,\alpha$ both free)')
    ax1.legend(fontsize=9)

    ax2.errorbar(z_vals, residuals, yerr=mu_err, fmt='o', color='crimson',
                 ms=2, alpha=0.35, capsize=0)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$\mu_{\rm obs}-\mu_{\rm model}$')
    ax2.set_ylim(-0.6, 0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram_delta_alpha_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_H0_tension_comparison(best_x, perr, outdir='.'):
    """Compare this fit's H0 against literature reference values."""
    H0_fit = best_x[0]
    H0_err = perr[0] if perr is not None and np.isfinite(perr[0]) else 0.0

    all_vals = {r"This work ($\delta,\alpha$ free, SNe)": (H0_fit, H0_err, "crimson")}
    for name, (val, err) in LITERATURE_H0.items():
        all_vals[name] = (val, err, "steelblue" if "Planck" in name else "darkorange")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(all_vals.items()):
        ax.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)
        ax.axvspan(val - err, val + err, color=color, alpha=0.1)
    ax.set_yticks(range(len(all_vals)))
    ax.set_yticklabels(all_vals.keys())
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_title(r'$H_0$: this fit ($\delta,\alpha$ free, SNe) vs. literature')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'H0_tension_comparison_delta_alpha_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MODEL COMPARISON TABLE
# =============================================================================

def create_model_comparison_table(best_x, chi2_best, z_vals, mu_vals, cov_inv,
                                  use_full_cov, inv_var, outdir='.'):
    """AIC / BIC comparison of the delta+alpha model vs pure LambdaCDM."""
    n = len(z_vals)
    dof_model = n - 4
    dof_lcdm = n - 2

    print("\n  Fitting LambdaCDM for comparison...")
    bounds_lcdm = [BOUNDS[0], (0.01, 0.99)]
    chi2_lcdm_fn = (lambda p, *a: chi2_lcdm_cov(p, *a)) if use_full_cov else \
                   (lambda p: chi2_diag([p[0], p[1], 0.0, BOUNDS[3][0]], z_vals, mu_vals, inv_var))
    args_lcdm = (z_vals, mu_vals, cov_inv) if use_full_cov else ()

    try:
        de_lcdm = differential_evolution(
            chi2_lcdm_fn, bounds=bounds_lcdm, args=args_lcdm,
            seed=42, maxiter=100, tol=1e-7, polish=True, popsize=12,
        )
        H0_l, Om_l = de_lcdm.x
        chi2_lcdm_best = de_lcdm.fun
        print(f"  LambdaCDM best fit: H0={H0_l:.2f}, Om={Om_l:.3f}, chi2={chi2_lcdm_best:.2f}")
    except Exception as e:
        print(f"  Warning: LambdaCDM fit failed: {e}")
        H0_l, Om_l = best_x[0], best_x[1]
        if use_full_cov:
            chi2_lcdm_best = chi2_lcdm_cov([H0_l, Om_l], z_vals, mu_vals, cov_inv)
        else:
            chi2_lcdm_best = chi2_diag([H0_l, Om_l, 0.0, BOUNDS[3][0]], z_vals, mu_vals, inv_var)

    def calculate_stats(chi2_val, k, dof):
        aic = chi2_val + 2 * k
        bic = chi2_val + k * np.log(n)
        chi2_dof = chi2_val / dof if dof > 0 else np.inf
        return aic, bic, chi2_dof

    aic_model, bic_model, chi2_dof_model = calculate_stats(chi2_best, 4, dof_model)
    aic_lcdm, bic_lcdm, chi2_dof_lcdm = calculate_stats(chi2_lcdm_best, 2, dof_lcdm)

    delta_aic = aic_model - aic_lcdm
    delta_bic = bic_model - bic_lcdm

    table_data = [
        ['Model', 'delta+alpha free', 'LambdaCDM'],
        ['H0', f'{best_x[0]:.2f}', f'{H0_l:.2f}'],
        ['Om,0', f'{best_x[1]:.3f}', f'{Om_l:.3f}'],
        ['delta', f'{best_x[2]:.3f}', '0 (fixed)'],
        ['alpha', f'{best_x[3]:.3f}', '0 (fixed)'],
        ['chi2', f'{chi2_best:.2f}', f'{chi2_lcdm_best:.2f}'],
        ['k', '4', '2'],
        ['dof', f'{dof_model}', f'{dof_lcdm}'],
        ['chi2/dof', f'{chi2_dof_model:.2f}', f'{chi2_dof_lcdm:.2f}'],
        ['AIC', f'{aic_model:.2f}', f'{aic_lcdm:.2f}'],
        ['dAIC', f'{delta_aic:+.2f}', '0 (ref)'],
        ['BIC', f'{bic_model:.2f}', f'{bic_lcdm:.2f}'],
        ['dBIC', f'{delta_bic:+.2f}', '0 (ref)'],
    ]

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE (Pantheon+SH0ES, delta and alpha both free)")
    print("=" * 70)
    col_w = [max(len(r[i]) for r in table_data) + 2 for i in range(3)]
    for row in table_data:
        print("│" + "│".join(f"{row[i]:^{col_w[i]}}" for i in range(3)) + "│")
    print("=" * 70)

    filename = os.path.join(outdir, 'model_comparison_table_delta_alpha_sn.txt')
    with open(filename, 'w') as f:
        f.write("MODEL COMPARISON TABLE (Pantheon+SH0ES, delta and alpha both free)\n")
        f.write("=" * 70 + "\n")
        for row in table_data:
            f.write(f"{row[0]:<12} {row[1]:<18} {row[2]:<18}\n")
        f.write("=" * 70 + "\n")
        f.write("\nNOTE: delta+alpha free has k=4 free parameters, while LambdaCDM has k=2.\n")
        if delta_aic < -2:
            f.write("delta+alpha model strongly preferred by AIC\n")
        elif delta_aic < 2:
            f.write("Models essentially equivalent by AIC\n")
        else:
            f.write("LambdaCDM preferred by AIC\n")
        if delta_bic < -2:
            f.write("delta+alpha model strongly preferred by BIC\n")
        elif delta_bic < 2:
            f.write("Models essentially equivalent by BIC\n")
        else:
            f.write("LambdaCDM preferred by BIC\n")

    print(f"\nModel comparison table saved to: {filename}")
    return table_data


# =============================================================================
# 9. EXPORTS & SUMMARY
# =============================================================================

def export_best_fit_data(z_vals, mu_vals, mu_err, best_x, outdir='.'):
    H0_fit, Om_fit, delta_fit, alpha_fit = best_x
    mu_best = mu_model(z_vals, H0_fit, Om_fit, delta_fit, alpha_fit)
    residuals = mu_vals - mu_best

    z_smooth = np.linspace(max(z_vals.min() * 0.5, 1e-4), z_vals.max() * 1.05, 250)
    mu_smooth = mu_model(z_smooth, H0_fit, Om_fit, delta_fit, alpha_fit)

    data_fname = os.path.join(outdir, 'delta_alpha_lcdm_sn_fit_results.txt')
    with open(data_fname, 'w') as f:
        f.write("# z, mu_obs, sigma_mu, mu_model, residual\n")
        for zi, mui, si, mm, ri in zip(z_vals, mu_vals, mu_err, mu_best, residuals):
            f.write(f"{zi:.6f} {mui:.6f} {si:.6f} {mm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_fname}")

    curve_fname = os.path.join(outdir, 'delta_alpha_lcdm_sn_smooth_curve.txt')
    with open(curve_fname, 'w') as f:
        f.write("# z, mu_model(z)\n")
        for zi, mi in zip(z_smooth, mu_smooth):
            f.write(f"{zi:.6f} {mi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_fname}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir="."):
    filename = os.path.join(outdir, "fit_summary_delta_alpha_sn.txt")
    if perr is None:
        perr = np.full(4, np.nan)

    with open(filename, "w") as f:
        f.write("===== BEST FIT (Pantheon+SH0ES, delta and alpha both free) =====\n\n")
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
    assert NWALKERS >= 16
    assert NSTEPS >= 500
    assert BOUNDS[0][0] < BOUNDS[0][1]
    assert BOUNDS[1][0] < BOUNDS[1][1]
    assert BOUNDS[2][0] < 0.0 < BOUNDS[2][1]
    assert BOUNDS[3][0] > 0.0, "alpha must stay strictly positive (division by alpha in the model)"


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    """Main function, structured to mirror delta_alpha_lcdm_fit.py's pipeline."""
    validate_config()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results_delta_alpha_sn")
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
        print(f"Redshift range: {z_vals.min():.4f} to {z_vals.max():.4f}")
        print("Both delta and alpha are FREE in this run.\n")
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
    print("\n--- Best fit (global optimizer + multi-start cross-check) ---")
    best_x, chi2_best, converged = best_fit(
        z_vals, mu_vals, cov_inv, use_full_cov, inv_var
    )
    if not np.isfinite(chi2_best):
        print("\nCRITICAL: no finite chi^2 found anywhere in the "
              "(H0, Om, delta, alpha) search box. Check BOUNDS and the data files.")
        return

    H0_fit, Om_fit, delta_fit, alpha_fit = best_x
    dof = len(z_vals) - 4
    print(f"  converged: {converged}")
    print(f"  H0    = {H0_fit:.4f}")
    print(f"  Om    = {Om_fit:.4f}")
    print(f"  delta = {delta_fit:.4f}")
    print(f"  alpha = {alpha_fit:.4f}")
    print(f"  chi^2 = {chi2_best:.4f}  (chi^2/dof = {chi2_best/dof:.4f}, dof={dof})")

    # --- curve_fit uncertainties ---
    perr = None
    pcov = None
    print("\n--- curve_fit covariance (Gaussian uncertainties) ---")
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
            H0_fit, Om_fit, delta_fit, alpha_fit = best_x
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

    plot_walkers(sampler, outdir=outdir)

    print("\n--- Corner plot ---")
    fig_corner = corner.corner(
        flat_samples, labels=[PARAM_LABELS[n] for n in PARAM_NAMES],
        truths=list(best_x), show_titles=True,
        quantiles=[0.16, 0.5, 0.84],
    )
    fig_corner.savefig(os.path.join(outdir, 'corner_delta_alpha_sn.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    # --- Profile & contours (diagonal errors for speed) ---
    print("\n--- Profile likelihoods (delta, alpha) ---")
    chi2_best_diag = chi2_diag(best_x, z_vals, mu_vals, inv_var)
    plot_chi2_profile_1d('delta', best_x, chi2_best_diag, z_vals, mu_vals, inv_var, outdir=outdir)
    plot_chi2_profile_1d('alpha', best_x, chi2_best_diag, z_vals, mu_vals, inv_var, outdir=outdir)

    print("\n--- 2D confidence contours ---")
    adaptive_contour_if_needed(best_x, chi2_best_diag, z_vals, mu_vals, inv_var,
                                vary=('delta', 'alpha'), outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best_diag, z_vals, mu_vals, inv_var,
                                vary=('delta', 'Om'), outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best_diag, z_vals, mu_vals, inv_var,
                                vary=('alpha', 'H0'), outdir=outdir)

    print("\n--- H0-Om contour (delta and alpha fixed at best-fit) ---")
    X, Y, dchi2 = plot_contour_H0_Om_fixed_delta_alpha(
        best_x, chi2_best_diag, z_vals, mu_vals, inv_var, outdir=outdir
    )
    np.save(os.path.join(outdir, 'contour_H0_Om_delta_alpha_free_sn.npy'),
            {'X': X, 'Y': Y, 'delta_chi2': dchi2})

    print("  Saved: chi2_profile_delta_sn.png, chi2_profile_alpha_sn.png, "
          "contour_delta_alpha_sn.png, contour_delta_Om_sn.png, contour_alpha_H0_sn.png, "
          "contour_H0_Om_fixed_delta_alpha_sn.png")

    # --- Hubble diagram ---
    print("\n--- Hubble diagram ---")
    plot_hubble_diagram_clean(best_x, z_vals, mu_vals, mu_err, outdir=outdir)
    print("  Saved: hubble_diagram_delta_alpha_sn.png")

    print("\n--- H0 tension comparison vs literature ---")
    plot_H0_tension_comparison(best_x, perr, outdir=outdir)
    print("  Saved: H0_tension_comparison_delta_alpha_sn.png")

    # --- Exports ---
    print("\n--- Export best-fit data ---")
    export_best_fit_data(z_vals, mu_vals, mu_err, best_x, outdir=outdir)

    print("\n--- Model Comparison Table ---")
    create_model_comparison_table(
        best_x, chi2_best, z_vals, mu_vals, cov_inv,
        use_full_cov, inv_var, outdir=outdir
    )

    if perr is not None and pcov is not None:
        corr = pcov / np.outer(perr, perr)
        print("\nCorrelation matrix")
        print("--------------------------------")
        for row in corr:
            print(" ".join(f"{x:8.3f}" for x in row))
        np.savetxt(os.path.join(outdir, "correlation_matrix_delta_alpha_sn.txt"), corr, fmt="%.6f")
    else:
        perr = np.full(4, np.nan)

    write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir)

    print(f"\nDone. All figures and results saved to: {outdir}")


if __name__ == "__main__":
    main()