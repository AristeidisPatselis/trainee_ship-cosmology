"""
delta_alpha_lcdm_fit.py
=======================
Combined modified Friedmann equation, delta AND alpha both FREE:

    H(z)^2 = H0^2 [ Om*(1+z)^3 + (1-Om)*(H(z)/H0)^delta ]  -  alpha*(1+z)*H(z)*dH/dz

This is the most general member of the family: delta4_lcdm_fit.py fixes
delta=4 with no alpha term at all, delta4_alpha_lcdm_fit.py fixes delta=4
but frees alpha, and this script frees BOTH. Free parameters:
(H0, Om, delta, alpha).

delta=0 collapses the explicit dark-energy term to the ordinary constant
(1-Om) LambdaCDM term (still combined with the alpha correction); alpha=0
collapses the correction term entirely, leaving the plain delta-model of
delta_lcdm_fit.py. Neither limit is enforced -- both delta and alpha are
searched freely within their prior box.

Solved via the same u = H^2 substitution as H_dot_lcdm_fit.py /
delta4_alpha_lcdm_fit.py: the H*dH/dz term becomes a first-order ODE for
u(z), and the (1-Om)*(H/H0)^delta term is just an explicit algebraic piece
of the ODE's right-hand side, evaluated as (u/H0^2)^(delta/2) -- for
non-even delta this requires u > 0 strictly, which the solver's clipping
already guarantees.

Pipeline: global optimizer (differential_evolution) + multi-start
Nelder-Mead polish, curve_fit covariance + full MCMC posterior (4D),
1D profile likelihoods for BOTH delta and alpha, 2D Delta-chi^2 contours
for (delta, alpha), (delta, Om), (alpha, H0), corner plot, Hubble diagram,
AIC/BIC vs baseline LambdaCDM.

Expects z_vals.txt, H_vals.txt, sigma_vals.txt in DATA_DIR (one number per
line; stray "]"-style artifacts tolerated).
"""

# NOTE TO SELF: this is the fully-free sibling of the family. It duplicates
# its own model logic rather than importing from delta4_alpha_lcdm_fit.py,
# so it stays a single self-contained script -- if the combined-equation
# ODE ever changes, mirror the change across delta4_alpha_lcdm_fit.py too.
# With 4 free params this is the slowest of the three to run -- start with
# small CONFIG knobs (below) while debugging, bump up for the "real" run.

# --- Standard library --------------------------------------------------------
import os
import time
import warnings
from functools import lru_cache
from tqdm import tqdm

# --- Numerics / optimization --------------------------------------------------
import numpy as np
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.integrate import solve_ivp
from scipy.stats import chi2 as chi2_dist

# --- Plotting ------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec

# --- Bayesian inference & stats -------------------------------------------------
import emcee                              # affine-invariant MCMC ensemble sampler
import corner                             # corner (triangle) plots for posteriors

np.random.seed(42)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# CONFIG
# =============================================================================
# All "tunable knobs" live here so they don't need to be hunted down inside
# the functions below.

# --- DATA CONFIGURATION ---
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/'
Z_FILE = 'c_z_vals.txt'
H_FILE = 'c_H_vals.txt'
SIGMA_FILE = 'c_sigma_vals.txt'

# --- FIT CONFIGURATION ---
# Prior/search box for (H0, Om, delta, alpha). Also used as hard bounds: any
# point outside gets chi^2 = +inf / log-prob = -inf.
#
# delta range is deliberately asymmetric-ish and wide enough to comfortably
# contain both the standard LambdaCDM point (delta=0) and the delta=4
# scenario tested in the other two scripts, without being so wide that the
# (u/H0^2)^(delta/2) term becomes numerically extreme during ODE stepping.
BOUNDS = [(40.0, 100.0), (0.01, 0.99), (-2.0, 6.0), (0.01, 6.0)]   # H0, Om, delta, alpha
PARAM_NAMES = ['H0', 'Om', 'delta', 'alpha']
PARAM_LABELS = {'H0': r'$H_0$', 'Om': r'$\Omega_{m,0}$',
                'delta': r'$\delta$', 'alpha': r'$\alpha$'}

CONTOUR_GRID = 50        # grid resolution (per axis) for Delta-chi^2 contours
                          # -> cost per contour plot is CONTOUR_GRID^2 chi2 evals
                          # (drop to ~25 while debugging, bump back up for the
                          # "real" run -- 4 free params makes this the most
                          # expensive script of the three)

PROFILE_POINTS = 50       # number of grid values in each 1D profile likelihood

# emcee sampler settings
NWALKERS = 48             # more walkers than the 3-param scripts (2*ndim rule of
                          # thumb, plus extra margin since ndim itself is bigger)
NSTEPS = 4000
DISCARD = 800             # burn-in steps discarded before computing statistics
THIN = 15

# Delta-chi^2 thresholds
CONF_LEVELS_2D = [2.30, 6.18, 11.83]
CONF_LEVELS_1D = [1.0, 4.0, 9.0]

# --- OPTIMIZATION CONFIGURATION ---
USE_CACHING = True
USE_PARALLEL = True
ADAPTIVE_CONTOURS = True
BATCH_SIZE = 100
N_MULTISTART = 8
USE_ANALYTICAL = False   # No analytical solution for this general case

# Reference H0 values from the literature for diagnostic plotting
LITERATURE_H0 = {
    "Planck 2018 (CMB)": (67.4, 0.5),
    "SH0ES 2022 (Local)": (73.04, 1.04),
}


# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

def setup_matplotlib():
    """Enable LaTeX only if a real render actually succeeds on this machine."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
        fig_test = plt.figure()
        plt.text(0.5, 0.5, r"$\alpha$")
        fig_test.canvas.draw()
        plt.close(fig_test)
    except Exception as e:
        print(f"Note: LaTeX rendering unavailable, using mathtext instead. ({e})")
        rc('text', usetex=False)
        rc('font', family='DejaVu Sans')


def find_file_recursively(filename, data_dir):
    """Search for a file recursively in data_dir and its subdirectories."""
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath

    for root, dirs, files in os.walk(data_dir):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(
        f"Could not find '{filename}' in '{data_dir}' or its subdirectories.\n"
        f"Available files in {data_dir} and subdirectories:\n"
        f"{list_available_files(data_dir)}"
    )


def list_available_files(data_dir, max_files=20):
    """List available .txt files in data_dir and subdirectories."""
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.txt'):
                rel_path = os.path.relpath(os.path.join(root, f), data_dir)
                files.append(rel_path)
                if len(files) >= max_files:
                    files.append("... and more")
                    return "\n".join(files)
    return "\n".join(files) if files else "No .txt files found"


def load_clean_data(filename, data_dir):
    """Load one numeric value per line, with recursive search.

    Tolerates stray bracket artifacts (e.g. leftover '...]' text) by keeping
    only what follows the last ']' on each line -- this exists because the
    raw H(z)/sigma files sometimes have a leading fragment like "[12] 67.3"
    per line (e.g. copy-pasted output with array indices still attached).
    """
    filepath = find_file_recursively(filename, data_dir)

    data = []
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.split(']')[-1].strip()
            if clean_line:
                data.append(float(clean_line))
    return np.array(data)


def load_all_data_memory_efficient():
    """Memory-efficient data loading using numpy's loadtxt, with recursive
    file discovery under DATA_DIR (same convention as H_dot_lcdm_fit.py)."""
    script_dir = os.path.dirname(os.path.realpath(__file__))

    if not os.path.isabs(DATA_DIR):
        data_dir = os.path.join(script_dir, DATA_DIR)
    else:
        data_dir = DATA_DIR

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Loading data from: {data_dir}")
    print(f"  Searching for: {Z_FILE}, {H_FILE}, {SIGMA_FILE}")

    z_vals = np.loadtxt(find_file_recursively(Z_FILE, data_dir))
    H_vals = np.loadtxt(find_file_recursively(H_FILE, data_dir))
    sigma_vals = np.loadtxt(find_file_recursively(SIGMA_FILE, data_dir))

    return z_vals, H_vals, sigma_vals


# =============================================================================
# 2. MODEL: implicit H(z) via the u = H^2 substitution (with caching)
# =============================================================================
#
# Starting equation:
#   H^2 = H0^2*[Om*(1+z)^3 + (1-Om)*(H/H0)^delta]  -  alpha*(1+z)*H*dH/dz
#
# Substitute u = H^2  =>  du/dz = 2*H*dH/dz, so H*dH/dz = (1/2) du/dz:
#
#   u = H0^2*Om*(1+z)^3 + H0^2*(1-Om)*(u/H0^2)^(delta/2)  -  (alpha*(1+z)/2)*du/dz
#
#   => du/dz = [2 / (alpha*(1+z))] *
#              ( H0^2*Om*(1+z)^3 + H0^2*(1-Om)*(u/H0^2)^(delta/2)  -  u )
#
# With delta now free (not just the even integer 4), (u/H0^2)^(delta/2) can
# involve a fractional power, so u MUST stay strictly positive throughout
# the integration -- the u_safe clipping below (floor of 1e-8) enforces
# that no step ever hands a non-positive base to a fractional exponent.
# Boundary condition: z=0, H=H0  =>  u(0) = H0^2.

def _rhs_u(z, u, H0, Om, delta, alpha):
    u_safe = min(max(u[0], 1e-8), 1e12)   # guard against a bad step blowing up
                                          # (and against a non-positive base
                                          # hitting the fractional delta/2 power)
    x = 1.0 + z
    de_term = H0 ** 2 * (1 - Om) * (u_safe / H0 ** 2) ** (delta / 2.0)
    dudz = (2.0 / (alpha * x)) * (H0 ** 2 * Om * x ** 3 + de_term - u_safe)
    return [np.clip(dudz, -1e12, 1e12)]


@lru_cache(maxsize=512)
def _model_H_cached(z_tuple, H0, Om, delta, alpha):
    """Cached version of model_H for repeated calls with same parameters."""
    z_eval = np.array(z_tuple)
    if alpha <= 0 or H0 <= 0 or not (0 < Om < 1):
        return tuple([np.nan] * len(z_eval))

    z_max = max(z_eval.max(), 1e-6)
    t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))

    try:
        sol = solve_ivp(
            _rhs_u, (0.0, z_max), [H0 ** 2],
            args=(H0, Om, delta, alpha),
            t_eval=t_eval,
            method='LSODA',        # handles the stiffness small alpha causes
            rtol=1e-8, atol=1e-10,
            max_step=0.05,
        )
    except Exception:
        return tuple([np.nan] * len(z_eval))

    if not sol.success:
        return tuple([np.nan] * len(z_eval))

    u_of_z = np.interp(z_eval, sol.t, sol.y[0])
    if np.any(~np.isfinite(u_of_z)) or np.any(u_of_z <= 0):
        return tuple([np.nan] * len(z_eval))
    return tuple(np.sqrt(u_of_z))


def model_H(z_eval, H0, Om, delta, alpha):
    """Solve the ODE for u=H^2 and return H(z) at the requested redshifts.

    Returns an array of NaNs if the integration fails or produces an
    unphysical (non-finite / negative) u, so a chi^2 built on this can
    penalize it cleanly instead of crashing.
    """
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
            _rhs_u, (0.0, z_max), [H0 ** 2],
            args=(H0, Om, delta, alpha),
            t_eval=t_eval,
            method='LSODA',
            rtol=1e-8, atol=1e-10,
            max_step=0.05,
        )
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


# =============================================================================
# 3. CHI-SQUARED
# =============================================================================

def _within_bounds(params):
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, BOUNDS))


def chi2(params, z_vals, H_vals, sigma_vals):
    H0, Om, delta, alpha = params
    if not _within_bounds(params):
        return 1e12
    H_theory = model_H(z_vals, H0, Om, delta, alpha)
    if np.any(~np.isfinite(H_theory)):
        return 1e12
    return float(np.sum(((H_vals - H_theory) / sigma_vals) ** 2))


def chi2_grid(params_grid, z_vals, H_vals, sigma_vals):
    """Chi-squared over a grid of (H0, Om, delta, alpha) triples, for contour plots."""
    chi2_vals = np.full(len(params_grid), 1e12)

    valid = np.array([_within_bounds(p) for p in params_grid])
    if not np.any(valid):
        return chi2_vals

    valid_params = params_grid[valid]
    chi2_vals_valid = np.zeros(len(valid_params))

    batch_size = BATCH_SIZE
    for i in range(0, len(valid_params), batch_size):
        batch = valid_params[i:i + batch_size]
        for j, params in enumerate(batch):
            H_model = model_H(z_vals, *params)
            if np.all(np.isfinite(H_model)) and np.all(H_model > 0):
                chi2_vals_valid[i + j] = np.sum(((H_vals - H_model) / sigma_vals) ** 2)
            else:
                chi2_vals_valid[i + j] = 1e12

    chi2_vals[valid] = chi2_vals_valid
    return chi2_vals


def chi2_lcdm(params, z_vals, H_vals, sigma_vals):
    H0, Om = params
    if H0 <= 0 or not (0 < Om < 1):
        return 1e12
    return float(np.sum(((H_vals - H_lcdm(z_vals, H0, Om)) / sigma_vals) ** 2))


# =============================================================================
# 4. BEST FIT: global optimizer + multi-start cross-check
# =============================================================================

def best_fit(z_vals, H_vals, sigma_vals, n_starts=N_MULTISTART, verbose=True):
    """Global fit (differential_evolution) followed by a Nelder-Mead polish,
    plus an independent multi-start Nelder-Mead scan as a cross-check."""
    print("  Running differential evolution...")
    de_result = differential_evolution(
        chi2, bounds=BOUNDS, args=(z_vals, H_vals, sigma_vals),
        seed=42, maxiter=300, tol=1e-8, polish=True, popsize=25,
    )
    best_x, best_chi2 = de_result.x, de_result.fun

    print(f"  Running {n_starts} multi-start local optimizations...")
    rng = np.random.default_rng(42)
    starts = [best_x] + [
        [rng.uniform(lo, hi) for (lo, hi) in BOUNDS] for _ in range(n_starts)
    ]

    local_results = []
    for x0 in tqdm(starts, desc="  Local optimizations", disable=not verbose):
        res = minimize(chi2, x0, args=(z_vals, H_vals, sigma_vals),
                        method='Nelder-Mead', bounds=BOUNDS,
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 8000})
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        if spread.size:
            print(f"  Multi-start scan: {spread.size}/{len(starts)} runs converged "
                  f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
            if spread.max() - spread.min() > 0.5:
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

def model_H_curvefit(z_array, H0, Om, delta, alpha):
    """Vectorized wrapper with the curve_fit-friendly (z, *params) signature."""
    H = model_H(z_array, H0, Om, delta, alpha)
    if np.any(~np.isfinite(H)):
        # curve_fit can't handle NaNs; push residuals huge instead of crashing
        return np.full_like(np.atleast_1d(z_array), 1e6, dtype=float)
    return H


def fit_uncertainties_curvefit(z_vals, H_vals, sigma_vals, p0):
    lo = [b[0] for b in BOUNDS]
    hi = [b[1] for b in BOUNDS]
    popt, pcov = curve_fit(
        model_H_curvefit, z_vals, H_vals, p0=p0,
        sigma=sigma_vals, absolute_sigma=True, bounds=(lo, hi), maxfev=30000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov


def log_prior(theta):
    for val, (lo, hi) in zip(theta, BOUNDS):
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_likelihood(theta, z_vals, H_vals, sigma_vals):
    c = chi2(theta, z_vals, H_vals, sigma_vals)
    if c >= 1e11:
        return -np.inf
    return -0.5 * c


def log_prob(theta, z_vals, H_vals, sigma_vals):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_vals, H_vals, sigma_vals)


def run_mcmc(best_x, z_vals, H_vals, sigma_vals,
             nwalkers=NWALKERS, nsteps=NSTEPS, discard=DISCARD, thin=THIN):
    ndim = 4
    spread = np.array([2.0, 0.05, 0.3, 0.1])   # small Gaussian ball around best fit

    pos = np.tile(best_x, (nwalkers, 1)).astype(float)

    for w in range(nwalkers):
        scale = spread.copy()
        for _ in range(50):
            candidate = best_x + scale * np.random.randn(ndim)
            for j, (lo, hi) in enumerate(BOUNDS):
                candidate[j] = np.clip(candidate[j], lo + 1e-6, hi - 1e-6)
            if np.isfinite(log_prob(candidate, z_vals, H_vals, sigma_vals)):
                pos[w] = candidate
                break
            scale *= 0.5   # shrink and retry if we keep landing at -inf

    pool = None
    if USE_PARALLEL:
        try:
            import multiprocessing
            n_cpus = multiprocessing.cpu_count()
            n_threads = max(1, min(n_cpus, nwalkers // 2))
            if n_threads > 1:
                pool = multiprocessing.Pool(processes=n_threads)
                print(f"  Using {n_threads} CPU cores for MCMC")
        except Exception as e:
            print(f"  Parallel processing not available: {e}")

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(z_vals, H_vals, sigma_vals),
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
            axes[i].plot(
                chain[:, walker, i],
                alpha=0.3,
                lw=0.5
            )
        axes[i].set_ylabel(PARAM_LABELS[PARAM_NAMES[i]])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "walker_chains_delta_alpha.png"), dpi=300)
    plt.close()


# =============================================================================
# 6. PROFILE LIKELIHOOD & CONFIDENCE CONTOURS
# =============================================================================

def plot_chi2_profile_1d(param_name, best_x, chi2_best, z_vals, H_vals, sigma_vals,
                          n_points=PROFILE_POINTS, outdir='.'):
    """1D profile chi^2(param): the other 3 parameters are re-fit at every
    grid value of `param_name`, so this is a true profile likelihood."""
    idx = PARAM_NAMES.index(param_name)
    other_idx = [i for i in range(4) if i != idx]
    p_fit = best_x[idx]
    p_lo_bound, p_hi_bound = BOUNDS[idx]

    # center the scan window on the best fit but don't exceed the prior box
    if p_fit >= 0:
        p_lo = max(p_lo_bound, p_fit * 0.3 if p_fit > 0 else p_lo_bound)
        p_hi = min(p_hi_bound, p_fit * 2.5 if p_fit > 0 else p_hi_bound)
    else:
        p_lo = max(p_lo_bound, p_fit * 2.5)
        p_hi = min(p_hi_bound, p_fit * 0.3)
    if p_hi - p_lo < 1e-3:   # degenerate window (e.g. p_fit ~ 0) -> use a fixed pad
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
            return chi2(full, z_vals, H_vals, sigma_vals)
        x0 = [best_x[oi] for oi in other_idx]
        bnds = [BOUNDS[oi] for oi in other_idx]
        res = minimize(chi2_reduced, x0, method='Nelder-Mead', bounds=bnds,
                       options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
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
    if p_lo68 is not None and p_hi68 is not None:
        ax.axvspan(p_lo68, p_hi68, color='navy', alpha=0.12, label=r'1$\sigma$ interval')
    for level, lvl_label in zip(CONF_LEVELS_1D, [r'1$\sigma$', r'2$\sigma$', r'3$\sigma$']):
        ax.axhline(level, color='gray', ls='--', lw=0.8)
        ax.text(grid[-1], level, lvl_label, va='bottom', ha='right', fontsize=9, color='gray')
    ax.set_xlabel(label)
    ax.set_ylabel(rf'$\Delta\chi^2({param_name})$')
    ax.set_title(rf'Profile likelihood: $\Delta\chi^2$ vs {label} (other 3 params refit)')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f'chi2_profile_{param_name}.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if p_lo68 is not None and p_hi68 is not None:
        print(f"  {param_name} 1sigma profile interval: [{p_lo68:.4f}, {p_hi68:.4f}]")

    return grid, chi2_vals


def plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                     vary=('delta', 'alpha'), n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour in two of the four parameters, with the other two
    held fixed at their best-fit values."""
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
    chi2_flat = chi2_grid(params_flat, z_vals, H_vals, sigma_vals)
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
    ax.plot(center[0], center[1], 'k*', ms=14, label='best fit')
    ax.set_xlabel(PARAM_LABELS[vary[0]])
    ax.set_ylabel(PARAM_LABELS[vary[1]])
    ax.set_title(rf'$\Delta\chi^2$ contours: {PARAM_LABELS[vary[0]]} vs {PARAM_LABELS[vary[1]]} '
                 rf'({fixed_desc}, at best fit)')
    ax.legend()
    fig.tight_layout()
    fname = f'contour_{vary[0]}_{vary[1]}.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return X, Y, delta_chi2


def plot_contour_H0_Om_fixed_delta_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                         n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour for H0 vs Om, with delta and alpha fixed at their
    best-fit values. This shows the 2D confidence region for the cosmological
    parameters at the best-fit values of the modified gravity parameters."""
    H0_fit, Om_fit, delta_fit, alpha_fit = best_x

    # Create grid around the best-fit H0 and Om
    h_lo = max(BOUNDS[0][0], H0_fit * 0.85)
    h_hi = min(BOUNDS[0][1], H0_fit * 1.15)
    o_lo = max(BOUNDS[1][0], Om_fit * 0.7)
    o_hi = min(BOUNDS[1][1], Om_fit * 1.3)

    # If the window is too small, expand it
    if h_hi - h_lo < 5:
        h_lo = max(BOUNDS[0][0], H0_fit - 5)
        h_hi = min(BOUNDS[0][1], H0_fit + 5)
    if o_hi - o_lo < 0.05:
        o_lo = max(BOUNDS[1][0], Om_fit - 0.05)
        o_hi = min(BOUNDS[1][1], Om_fit + 0.05)

    H0_grid = np.linspace(h_lo, h_hi, n_grid)
    Om_grid = np.linspace(o_lo, o_hi, n_grid)
    X, Y = np.meshgrid(H0_grid, Om_grid)

    # Compute chi2 for each (H0, Om) pair with fixed delta and alpha.
    # NaN (not chi2()'s 1e12 sentinel) marks invalid cells here: contourf/contour
    # treat NaN as a genuine masked gap, whereas a finite-but-huge value like 1e12
    # gets linearly interpolated against its valid neighbors, which can bite small
    # notches out of the outer (3-sigma) contour edge right where the ODE solver
    # happened to fail for a particular (H0, Om) combination.
    chi2_vals = np.full((n_grid, n_grid), np.nan)

    print(f"  Computing {n_grid}x{n_grid} grid for H0-Om contour "
          f"(delta={delta_fit:.3f}, alpha={alpha_fit:.3f} fixed)...")

    for i in tqdm(range(n_grid), desc="  H0-Om contour"):
        for j in range(n_grid):
            params = (X[i, j], Y[i, j], delta_fit, alpha_fit)
            if not _within_bounds(params):
                continue
            H_theory = model_H(z_vals, *params)
            if np.all(np.isfinite(H_theory)) and np.all(H_theory > 0):
                chi2_vals[i, j] = np.sum(((H_vals - H_theory) / sigma_vals) ** 2)

    delta_chi2 = chi2_vals - chi2_best

    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot contours
    cs = ax.contour(X, Y, delta_chi2, levels=CONF_LEVELS_2D,
                    colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={CONF_LEVELS_2D[0]: r'1$\sigma$',
                       CONF_LEVELS_2D[1]: r'2$\sigma$',
                       CONF_LEVELS_2D[2]: r'3$\sigma$'})

    # Filled contour
    ax.contourf(X, Y, delta_chi2,
                levels=[0, *CONF_LEVELS_2D, max(np.nanmax(delta_chi2), CONF_LEVELS_2D[-1] + 1)],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)

    # Mark best fit
    ax.plot(H0_fit, Om_fit, 'k*', ms=14, label='best fit')

    # Add literature markers (optional)
    if 'LITERATURE_H0' in globals():
        for name, (h_val, h_err) in LITERATURE_H0.items():
            # We don't have Om for literature values, so just show H0
            ax.axvline(h_val, color='gray', ls=':', alpha=0.5, lw=1)
            ax.text(h_val, ax.get_ylim()[1], name, rotation=90,
                    va='top', ha='right', fontsize=7, alpha=0.6)

    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_ylabel(r'$\Omega_{m,0}$')
    ax.set_title(rf'$\Delta\chi^2$ contours: $H_0$ vs $\Omega_{{m,0}}$ '
                 rf'($\delta={delta_fit:.3f}$, $\alpha={alpha_fit:.3f}$ fixed)')
    ax.legend()

    fig.tight_layout()
    fname = 'contour_H0_Om_fixed_delta_alpha.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {fname}")
    return X, Y, delta_chi2


def adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                vary=('delta', 'alpha'),
                                n_grid_min=30, n_grid_max=80, outdir='.'):
    """Adaptive contour plotting with variable resolution."""
    if not ADAPTIVE_CONTOURS:
        return plot_contour_2d(
            best_x, chi2_best, z_vals, H_vals, sigma_vals,
            vary=vary, n_grid=CONTOUR_GRID, outdir=outdir
        )

    n_grid = n_grid_min
    X, Y, delta = plot_contour_2d(
        best_x, chi2_best, z_vals, H_vals, sigma_vals,
        vary=vary, n_grid=n_grid, outdir=outdir
    )

    grad_x = np.gradient(delta, axis=0)
    grad_y = np.gradient(delta, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    if np.std(grad_mag) > 0.5 * np.mean(grad_mag) and n_grid < n_grid_max:
        n_grid = min(n_grid * 2, n_grid_max)
        print(f"  Refining contour grid to {n_grid}x{n_grid}...")
        X, Y, delta = plot_contour_2d(
            best_x, chi2_best, z_vals, H_vals, sigma_vals,
            vary=vary, n_grid=n_grid, outdir=outdir
        )

    return X, Y, delta


# =============================================================================
# 7. HUBBLE DIAGRAM
# =============================================================================

def plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir='.'):
    H0_fit, Om_fit, delta_fit, alpha_fit = best_x
    z_smooth = np.linspace(0, z_vals.max() * 1.05, 300)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit, delta_fit, alpha_fit)
    H_at_data = model_H(z_vals, H0_fit, Om_fit, delta_fit, alpha_fit)
    residuals = H_vals - H_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', color='crimson',
                 ms=4, capsize=2, label='Cosmic chronometer data')
    ax1.plot(z_smooth, H_smooth, color='navy', lw=2,
              label=rf'model fit ($\delta={delta_fit:.3f}$, $\alpha={alpha_fit:.3f}$)')
    H_lcdm_smooth = H_lcdm(z_smooth, H0_fit, Om_fit)
    ax1.plot(z_smooth, H_lcdm_smooth, color='green', lw=1.5, ls='--',
              label=r'$\Lambda$CDM ($\delta=0$, same $H_0,\Omega_{m,0}$)')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble diagram: best fit (delta and alpha both free)')
    ax1.legend()

    ax2.errorbar(z_vals, residuals, yerr=sigma_vals, fmt='o', color='crimson', ms=4, capsize=2)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram_delta_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_H0_tension_comparison(best_x, perr, outdir='.'):
    """Compare this fit's H0 against literature reference values."""
    H0_fit, _, _, _ = best_x
    H0_err = perr[0] if perr is not None else 0.0

    all_vals = {"This work ($\\delta, \\alpha$ free)": (H0_fit, H0_err, "crimson")}
    for name, (val, err) in LITERATURE_H0.items():
        all_vals[name] = (val, err, "steelblue" if "Planck" in name else "darkorange")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(all_vals.items()):
        ax.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)
        ax.axvspan(val - err, val + err, color=color, alpha=0.1)
    ax.set_yticks(range(len(all_vals)))
    ax.set_yticklabels(all_vals.keys())
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_title(r'$H_0$: this fit ($\delta, \alpha$ free) vs. literature')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'H0_tension_comparison_delta_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MODEL COMPARISON TABLE
# =============================================================================

def create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir='.'):
    """Create a comprehensive model comparison table with all statistics."""
    n = len(z_vals)
    dof_model = n - 4
    dof_lcdm = n - 2

    # Fit LambdaCDM
    de_lcdm = differential_evolution(
        chi2_lcdm, bounds=[(BOUNDS[0][0], BOUNDS[0][1]), (0.01, 0.99)],
        args=(z_vals, H_vals, sigma_vals), seed=42, tol=1e-8,
    )
    H0_l, Om_l = de_lcdm.x
    chi2_lcdm_best = de_lcdm.fun

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
        ['Ωm,0', f'{best_x[1]:.3f}', f'{Om_l:.3f}'],
        ['δ', f'{best_x[2]:.3f}', '0 (fixed)'],
        ['α', f'{best_x[3]:.3f}', '0 (fixed)'],
        ['χ²', f'{chi2_best:.2f}', f'{chi2_lcdm_best:.2f}'],
        ['k', '4', '2'],
        ['dof', f'{dof_model}', f'{dof_lcdm}'],
        ['χ²/dof', f'{chi2_dof_model:.2f}', f'{chi2_dof_lcdm:.2f}'],
        ['AIC', f'{aic_model:.2f}', f'{aic_lcdm:.2f}'],
        ['ΔAIC', f'{delta_aic:+.2f}', '0 (reference)'],
        ['BIC', f'{bic_model:.2f}', f'{bic_lcdm:.2f}'],
        ['ΔBIC', f'{delta_bic:+.2f}', '0 (reference)'],
    ]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE (delta and alpha both free)")
    print("=" * 80)

    col_widths = [max(len(row[i]) for row in table_data) + 2 for i in range(3)]

    print("│" + "│".join(f"{col:^{col_widths[i]}}" for i, col in enumerate(['Parameter', 'delta+alpha free', 'LambdaCDM'])) + "│")
    print("├" + "─" * col_widths[0] + "┼" + "─" * col_widths[1] + "┼" + "─" * col_widths[2] + "┤")

    for row in table_data[1:]:
        print("│" + "│".join(f"{row[i]:^{col_widths[i]}}" for i in range(3)) + "│")

    print("=" * 80)
    print("NOTE: delta+alpha free has k=4 free parameters, while LambdaCDM has k=2.")

    filename = os.path.join(outdir, 'model_comparison_table_delta_alpha.txt')
    with open(filename, 'w') as f:
        f.write("MODEL COMPARISON TABLE (delta and alpha both free)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Parameter':<15} {'delta+alpha free':<20} {'LambdaCDM':<20}\n")
        f.write("-" * 80 + "\n")
        for row in table_data[1:]:
            f.write(f"{row[0]:<15} {row[1]:<20} {row[2]:<20}\n")
        f.write("=" * 80 + "\n")
        f.write("\nNOTE: delta+alpha free has k=4 free parameters, while LambdaCDM has k=2.\n")

        f.write("\nINTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        if delta_aic < -2:
            f.write("✓ delta+alpha free model is strongly preferred by AIC\n")
        elif delta_aic < 0:
            f.write("✓ delta+alpha free model is slightly preferred by AIC\n")
        elif delta_aic < 2:
            f.write("○ Models are essentially equivalent by AIC\n")
        else:
            f.write("✗ LambdaCDM is preferred by AIC\n")

        if delta_bic < -2:
            f.write("✓ delta+alpha free model is strongly preferred by BIC\n")
        elif delta_bic < 0:
            f.write("✓ delta+alpha free model is slightly preferred by BIC\n")
        elif delta_bic < 2:
            f.write("○ Models are essentially equivalent by BIC\n")
        else:
            f.write("✗ LambdaCDM is preferred by BIC\n")

    print(f"\nModel comparison table saved to: {filename}")
    return table_data


# =============================================================================
# 9. EXPORTS & DIAGNOSTICS
# =============================================================================

def export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, outdir='.'):
    """Export the best-fit model predictions and residuals to text files."""
    H0_fit, Om_fit, delta_fit, alpha_fit = best_x
    H_best = model_H(z_vals, H0_fit, Om_fit, delta_fit, alpha_fit)
    residuals = H_vals - H_best

    z_smooth = np.linspace(0, z_vals.max() * 1.1, 200)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit, delta_fit, alpha_fit)

    header = "# z, H_obs, sigma_H, H_model, residual\n"
    data_filename = os.path.join(outdir, 'delta_alpha_lcdm_fit_results.txt')
    with open(data_filename, 'w') as f:
        f.write(header)
        for zi, Hi, si, Hm, ri in zip(z_vals, H_vals, sigma_vals, H_best, residuals):
            f.write(f"{zi:.6f} {Hi:.6f} {si:.6f} {Hm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_filename}")

    curve_filename = os.path.join(outdir, 'delta_alpha_lcdm_smooth_curve.txt')
    with open(curve_filename, 'w') as f:
        f.write("# z, H_model(z)\n")
        for zi, Hi in zip(z_smooth, H_smooth):
            f.write(f"{zi:.6f} {Hi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_filename}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir="."):
    """Save a comprehensive summary of parameters and statistical markers to disk."""
    filename = os.path.join(outdir, "fit_summary_delta_alpha.txt")

    if perr is None:
        perr = np.full(4, np.nan)

    with open(filename, "w") as f:
        f.write("===== BEST FIT (delta and alpha both free) =====\n\n")
        for n, v, e in zip(PARAM_NAMES, best_x, perr):
            f.write(f"{n:8s} = {v:.6f} +/- {e:.6f}\n")
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
            for n, t in zip(PARAM_NAMES, tau):
                f.write(f"{n:8s} {t:.2f}\n")
        except Exception:
            pass
        f.write("\n")
        p = np.percentile(flat_samples, [16, 50, 84], axis=0)
        f.write("===== MCMC =====\n\n")
        for i, n in enumerate(PARAM_NAMES):
            lo, med, hi = p[:, i]
            f.write(
                f"{n:8s} = {med:.6f} "
                f"(+{hi-med:.6f}/-{med-lo:.6f})\n"
            )
    print(f"  Exported fit summary to: {filename}")


def validate_config():
    """Validate configuration parameters."""
    assert CONTOUR_GRID >= 20, "CONTOUR_GRID should be at least 20"
    assert PROFILE_POINTS >= 20, "PROFILE_POINTS should be at least 20"
    assert NWALKERS >= 16, "NWALKERS should be at least 16"
    assert NSTEPS >= 1000, "NSTEPS should be at least 1000"

    assert BOUNDS[0][0] < BOUNDS[0][1], "Invalid H0 bounds"
    assert BOUNDS[1][0] < BOUNDS[1][1], "Invalid Om bounds"
    assert BOUNDS[2][0] < BOUNDS[2][1], "Invalid delta bounds"
    assert BOUNDS[3][0] < BOUNDS[3][1], "Invalid alpha bounds"

    for i in range(len(CONF_LEVELS_2D) - 1):
        assert CONF_LEVELS_2D[i] < CONF_LEVELS_2D[i + 1], "Confidence levels not increasing"


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    """Main function, structured to mirror H_dot_lcdm_fit.py's pipeline."""
    validate_config()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results_delta_alpha")
    os.makedirs(outdir, exist_ok=True)
    print(f"Results will be saved to: {outdir}\n")

    setup_matplotlib()

    z_vals, H_vals, sigma_vals = load_all_data_memory_efficient()
    print(f"\nLoaded {len(z_vals)} data points.")
    print(f"Redshift range: {z_vals.min():.3f} to {z_vals.max():.3f}")
    print("Both delta and alpha are FREE in this run.\n")

    print("\n--- Best fit (global optimizer + multi-start cross-check) ---")
    best_x, chi2_best, converged = best_fit(z_vals, H_vals, sigma_vals)
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

    perr = None
    print("\n--- curve_fit covariance (Gaussian/Laplace uncertainties) ---")
    try:
        popt, perr, pcov = fit_uncertainties_curvefit(z_vals, H_vals, sigma_vals, best_x)
        for name, val, err in zip(PARAM_NAMES, popt, perr):
            print(f"  {name:6s} = {val:.4f} +/- {err:.4f}")
        chi2_cf = chi2(popt, z_vals, H_vals, sigma_vals)
        if chi2_cf < chi2_best:
            best_x, chi2_best = popt, chi2_cf
            H0_fit, Om_fit, delta_fit, alpha_fit = best_x
    except Exception as e:
        print(f"  curve_fit uncertainty estimation failed: {e}")

    print("\n--- MCMC posterior (emcee) ---")
    sampler, flat_samples = run_mcmc(best_x, z_vals, H_vals, sigma_vals)
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = percentiles[:, i]
        print(f"  {name:6s} = {med:.4f} (+{hi-med:.4f} / -{med-lo:.4f})")

    plot_walkers(sampler, outdir=outdir)

    print("\n--- Corner plot ---")
    fig_corner = corner.corner(
        flat_samples, labels=[PARAM_LABELS[n] for n in PARAM_NAMES],
        truths=list(best_x), show_titles=True,
    )
    fig_corner.savefig(os.path.join(outdir, 'corner_delta_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    print("\n--- Profile likelihoods (delta, alpha) ---")
    plot_chi2_profile_1d('delta', best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)
    plot_chi2_profile_1d('alpha', best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)

    print("\n--- 2D confidence contours ---")
    adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                vary=('delta', 'alpha'), outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                vary=('delta', 'Om'), outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                vary=('alpha', 'H0'), outdir=outdir)

    # NEW: H0-Om contour with delta and alpha fixed at best-fit values
    print("\n--- H0-Om contour (delta and alpha fixed at best-fit) ---")
    X, Y, delta_chi2 = plot_contour_H0_Om_fixed_delta_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                      n_grid=CONTOUR_GRID, outdir=outdir)

    # Save contour data for later comparison
    contour_data = {'X': X, 'Y': Y, 'delta_chi2': delta_chi2}
    np.save(os.path.join(outdir, 'contour_H0_Om_delta_alpha_free.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(outdir, 'contour_H0_Om_delta_alpha_free.npy')}")

    print("  Saved: chi2_profile_delta.png, chi2_profile_alpha.png, "
          "contour_delta_alpha.png, contour_delta_Om.png, contour_alpha_H0.png, "
          "contour_H0_Om_fixed_delta_alpha.png")

    print("\n--- Hubble diagram ---")
    plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir=outdir)
    print("  Saved: hubble_diagram_delta_alpha.png")

    print("\n--- H0 tension comparison vs literature ---")
    plot_H0_tension_comparison(best_x, perr, outdir=outdir)
    print("  Saved: H0_tension_comparison_delta_alpha.png")

    print("\n--- Export best-fit data ---")
    export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, outdir=outdir)

    print("\n--- Model Comparison Table ---")
    create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)

    print(f"\nDone. All figures and results saved to: {outdir}")

    if perr is not None and 'pcov' in locals():
        corr = pcov / np.outer(perr, perr)
        print("\nCorrelation matrix")
        print("--------------------------------")
        for row in corr:
            print(" ".join(f"{x:8.3f}" for x in row))
        np.savetxt(
            os.path.join(outdir, "correlation_matrix_delta_alpha.txt"),
            corr,
            fmt="%.6f"
        )
    else:
        if perr is None:
            perr = np.full(4, np.nan)

    write_fit_summary(
        best_x,
        perr,
        chi2_best,
        dof,
        flat_samples,
        sampler,
        outdir
    )

if __name__ == "__main__":
    main()
