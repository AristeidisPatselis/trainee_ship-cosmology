"""
omega_fT_fit.py
================
Omega-parametrization fit, following Anagnostopoulos, Basilakos & Saridakis
2019 (arXiv:1907.07533), "Bayesian analysis of f(T) gravity using fsigma8
data".

Instead of modifying the H(z) dependence of the dark-energy term directly
(as delta_lcdm_fit.py and H_dot_lcdm_fit.py do), this script parametrizes
the *effective dark-energy density fraction itself*, Omega_DE(z), through
the paper's y(z,b) function:

    E(z)^2 = H(z)^2/H0^2 = Om*(1+z)^3 + (1-Om)*y(z,b)

with y(z,b) -> 1 in the appropriate b -> 0 (or b -> 0+) limit, recovering
flat LambdaCDM. b is the single "distortion parameter" that quantifies the
departure of f(T) gravity away from a pure cosmological constant. This is
exactly analogous to delta in delta_lcdm_fit.py, except delta re-scales the
whole DE term by a power of E, whereas b here lives *inside* an f(T)-model-
specific functional form for y.

Three viable one-parameter f(T) models are implemented (paper's Sec. II C,
Eqs. 27-37), all of which reduce to LambdaCDM as b -> 0(+):

  1. f1CDM (power-law, Eq. 29):        y(z,b) = E(z,b)^(2b)
  2. f2CDM (sqrt-exponential, Eq. 33): y(z,b) = [1-(1+E^b)e^(-E/b)] /
                                                  [1-(1+1/b)e^(-1/b)]
  3. f3CDM (exponential, Eq. 37):      y(z,b) = [1-(1+2E^2/b)e^(-E^2/b)] /
                                                  [1-(1+2/b)e^(-1/b)]

In every case E(z) appears on BOTH sides of the Friedmann equation (inside
y as well as outside), so -- exactly as in delta_lcdm_fit.py -- there is no
closed form for E(z) except at the LambdaCDM limit, and E(z) has to be
solved for root-by-root with a bracketed solver (brentq) at every redshift.
This script re-uses that same implicit-equation architecture, generalized
to dispatch across the three y(z,b) functions above.

Only the background (H(z)) data are used here (DESI) --
the paper's fsigma8/SNIa/CMBshift probes constrain the *perturbation* sector
(Eq. 19-26) and growth-rate data, which isn't part of this pipeline. We
therefore do not reproduce the paper's exact best-fit numbers (Table I),
only its background-level model definitions and its AIC/BIC/DIC
model-comparison machinery (Table II), applied to our own H(z) dataset.

Parameter order is (H0, Om, b) for every f(T) model, matching the (H0, Om,
delta)/(H0, Om, alpha) convention of the other two scripts.

Pipeline per model: global optimizer + multi-start polish, curve_fit
covariance, emcee MCMC, corner plot, 1D profile likelihood for b, 2D
delta-chi^2 contours (b vs Om, b vs H0), per-model Hubble diagram. Then,
across all three f(T) models plus flat LambdaCDM: a combined AIC/BIC/DIC
model-comparison table (DIC computed straight from the MCMC chains,
following the paper's Eq. 51-53), a combined Hubble-diagram overlay, and a
b -> 0 consistency check per model.
"""

# NOTE TO SELF: sibling of delta_lcdm_fit.py / H_dot_lcdm_fit.py. Same
# overall pipeline shape, but the "modification" here lives inside y(z,b)
# rather than in the DE term's exponent, and there are three of them to
# compare (that's the point of the paper -- which f(T) form the data like
# best). Keep the three y-functions and their b-bounds in one place
# (Y_FUNCS / MODEL_B_BOUNDS) so adding a fourth model later is a one-line
# registry entry, not a rewrite.

# --- Standard library --------------------------------------------------------
import os
import warnings
from functools import lru_cache
from tqdm import tqdm

# --- Numerics / optimization --------------------------------------------------
import numpy as np
from scipy.optimize import minimize, differential_evolution, curve_fit, brentq

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

# --- DATA CONFIGURATION --- (same convention/path as delta_lcdm_fit.py)
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/'
Z_FILE = 'd_z_vals.txt'
H_FILE = 'd_H_vals.txt'
SIGMA_FILE = 'd_sigma_vals.txt'

# --- FIT CONFIGURATION ---
H0_BOUNDS = (40.0, 100.0)
OM_BOUNDS = (0.01, 0.99)

# b-bounds are model-specific: f1CDM's y=E^(2b) is regular at b=0, so b can
# straddle 0 freely. f2CDM/f3CDM both contain 1/b terms and are genuinely
# singular at b=0 (same situation as alpha=0 in H_dot_lcdm_fit.py) -- so we
# keep b bounded away from 0 there and do the b->0 consistency check at a
# small-but-finite b instead of exactly zero.
# These bounds were checked explicitly (see dev notes) by scanning the
# implicit E(z) equation for extra roots across (z, b): f3CDM genuinely
# develops a second, spurious branch above b ~ 0.217 (not just a hard-to-
# find root -- the equation itself becomes multi-valued there), and
# f1CDM's positive branch runs out of real solutions above b ~ 1.0-1.05
# (E grows faster than the matter term can support at high z). f2CDM
# stays single-valued across its whole listed range. All three bounds
# still comfortably bracket the paper's fitted values (|b| <~ 0.2 in every
# case, Table I), so this doesn't cut into the physically relevant region.
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

CONTOUR_GRID = 60
PROFILE_POINTS = 60

# emcee sampler settings
NWALKERS = 32
NSTEPS = 3000
DISCARD = 500
THIN = 15

# Delta-chi^2 thresholds
CONF_LEVELS_2D = [2.30, 6.18, 11.83]
CONF_LEVELS_1D = [1.0, 4.0, 9.0]

# --- OPTIMIZATION CONFIGURATION ---
USE_CACHING = True
ADAPTIVE_CONTOURS = True
BATCH_SIZE = 100
N_MULTISTART = 8


def get_bounds(model_name):
    return [H0_BOUNDS, OM_BOUNDS, MODEL_B_BOUNDS[model_name]]


# =============================================================================
# 1. SETUP & DATA LOADING (identical convention to delta_lcdm_fit.py)
# =============================================================================

def setup_matplotlib():
    """Enable LaTeX only if a real render actually succeeds on this machine."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
        fig_test = plt.figure()
        plt.text(0.5, 0.5, r"$b$")
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


def load_all_data_memory_efficient():
    """Memory-efficient data loading, same recursive-search convention as
    delta_lcdm_fit.py / H_dot_lcdm_fit.py."""
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
# 2. MODELS: three f(T) y(z,b) forms, each solved implicitly for E(z)
# =============================================================================
#
# Note E(z) = H(z)/H0 never involves H0 itself (H0 only rescales H = H0*E
# at the very end), so the brentq root-find here is 1 dimension leaner than
# delta_lcdm_fit.py's _H_single: we solve for E directly, not H.

def y_f1(E, b):
    """Power-law model, Eq. (29): y(z,b) = E(z,b)^(2b). Regular at b=0
    (E^0 = 1 identically), which is why f1CDM's b-bounds are allowed to
    straddle zero."""
    if E <= 0:
        return np.nan
    try:
        return E ** (2.0 * b)
    except Exception:
        return np.nan


def y_f2(E, b):
    """Square-root exponential model, Eq. (33), obtained from Eq. (32)
    y(z,p) = [1-(1+pE)e^(-pE)] / [1-(1+p)e^(-p)] via p = 1/b:

        y(z,b) = [1 - (1 + E/b)*e^(-E/b)] / [1 - (1 + 1/b)*e^(-1/b)]

    Note the numerator's linear term is E/b (i.e. p*E with p=1/b), NOT
    E^b -- at E=1 (z=0) this makes numerator == denominator exactly, so
    y(1,b) = 1 identically for *any* b (this is the alpha-normalization
    built into Eq. 31, not just a b->0 limit).

    Singular as b -> 0 (both numerator and denominator involve 1/b), but
    the ratio -> 1 in that limit -- same character as alpha=0 in
    H_dot_lcdm_fit.py.
    """
    if E <= 0 or b == 0:
        return np.nan
    try:
        num = 1.0 - (1.0 + E / b) * np.exp(-E / b)
        den = 1.0 - (1.0 + 1.0 / b) * np.exp(-1.0 / b)
        if den == 0 or not np.isfinite(den):
            return np.nan
        return num / den
    except Exception:
        return np.nan


def y_f3(E, b):
    """Exponential model, Eq. (37). Same b->0 singular/regularizing
    character as y_f2, but with E^2 in place of E."""
    if E <= 0 or b == 0:
        return np.nan
    try:
        E2 = E * E
        num = 1.0 - (1.0 + 2.0 * E2 / b) * np.exp(-E2 / b)
        den = 1.0 - (1.0 + 2.0 / b) * np.exp(-1.0 / b)
        if den == 0 or not np.isfinite(den):
            return np.nan
        return num / den
    except Exception:
        return np.nan


Y_FUNCS = {'f1CDM': y_f1, 'f2CDM': y_f2, 'f3CDM': y_f3}


def _E_at_z(z, Om, b, model_name, E_guess):
    """Solve E - sqrt(Om*(1+z)^3 + (1-Om)*y(E,b)) = 0 for one z, bracketing
    tightly around E_guess (the solution at the previous, nearby z -- see
    _E_curve below). y_f2/y_f3 involve exponentials of E (or E^2) and are
    not guaranteed monotonic in E globally, so a wide bracket can straddle
    more than one root and hand brentq a spurious, discontinuous branch.
    Bracketing around a nearby known-good solution instead keeps us on the
    physical branch that continuously connects to E(0)=1.
    """
    y_func = Y_FUNCS[model_name]

    def eq(E):
        if E <= 0:
            return 1e10
        y = y_func(E, b)
        if y is None or not np.isfinite(y):
            return 1e10
        inside = Om * (1 + z) ** 3 + (1 - Om) * y
        if inside <= 0 or not np.isfinite(inside):
            return 1e10
        return E - np.sqrt(inside)

    try:
        lo, hi = 0.5 * E_guess, 2.0 * E_guess
        flo, fhi = eq(lo), eq(hi)
        tries = 0
        while (not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0) and tries < 12:
            lo *= 0.6
            hi *= 1.6
            flo, fhi = eq(lo), eq(hi)
            tries += 1
        if not np.isfinite(flo) or not np.isfinite(fhi) or flo * fhi > 0:
            return np.nan
        return brentq(eq, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200)
    except Exception:
        return np.nan


def _E_curve(z_tuple, Om, b, model_name):
    """Solve E(z) at every z in z_tuple via continuation in ascending z,
    starting from the exact, model-independent anchor E(0)=1 and using
    each solution as the bracket center for the next (nearby, slightly
    larger) z. Returns an array in the ORIGINAL order of z_tuple."""
    z_arr = np.asarray(z_tuple, dtype=float)
    order = np.argsort(z_arr)
    z_sorted = z_arr[order]

    E_sorted = np.empty_like(z_sorted)
    E_prev = 1.0  # exact: E(z=0) = 1 for every b, in every model here
    for i, zz in enumerate(z_sorted):
        if zz <= 0:
            E_sorted[i] = 1.0
        else:
            E_sorted[i] = _E_at_z(zz, Om, b, model_name, E_prev)
        if np.isfinite(E_sorted[i]):
            E_prev = E_sorted[i]
        # if this z failed, keep E_prev as-is (last good anchor) so later
        # z values aren't derailed by one bad point

    E_out = np.empty_like(E_sorted)
    E_out[order] = E_sorted
    return E_out


@lru_cache(maxsize=4096)
def _E_curve_cached(z_tuple, Om, b, model_name):
    return tuple(_E_curve(z_tuple, Om, b, model_name))


def model_E(z_eval, Om, b, model_name):
    """Vectorized wrapper: solves for E=H/H0 at every z in z_eval.

    The full dataset's z array is fixed throughout a fitting run, so it's
    cheap and correct to cache the WHOLE solved curve keyed on
    (z_tuple, Om, b, model_name) rather than caching per-z (continuation
    means per-z results depend on the whole ascending sequence anyway).
    """
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    z_tuple = tuple(float(z) for z in z_eval)
    if USE_CACHING:
        return np.array(_E_curve_cached(z_tuple, float(Om), float(b), model_name))
    return _E_curve(z_tuple, float(Om), float(b), model_name)


def model_H(z_eval, H0, Om, b, model_name):
    return H0 * model_E(z_eval, Om, b, model_name)


def H_lcdm(z, H0, Om):
    """Standard flat LambdaCDM, used as the baseline for AIC/BIC/DIC."""
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


# =============================================================================
# 3. CHI-SQUARED
# =============================================================================

def _within_bounds(params, model_name):
    bounds = get_bounds(model_name)
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, bounds))


def chi2(params, z_vals, H_vals, sigma_vals, model_name):
    H0, Om, b = params
    if not _within_bounds(params, model_name):
        return 1e12
    H_model = model_H(z_vals, H0, Om, b, model_name)
    if np.any(~np.isfinite(H_model)) or np.any(H_model <= 0):
        return 1e12
    return float(np.sum(((H_vals - H_model) / sigma_vals) ** 2))


def chi2_grid(params_grid, z_vals, H_vals, sigma_vals, model_name):
    """Chi-squared over a grid of (H0, Om, b) triples, for contour plots."""
    chi2_vals = np.full(len(params_grid), 1e12)

    valid = np.array([_within_bounds(p, model_name) for p in params_grid])
    if not np.any(valid):
        return chi2_vals

    valid_params = params_grid[valid]
    chi2_vals_valid = np.zeros(len(valid_params))

    batch_size = BATCH_SIZE
    for i in range(0, len(valid_params), batch_size):
        batch = valid_params[i:i + batch_size]
        for j, params in enumerate(batch):
            H_model = model_H(z_vals, *params, model_name)
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

def best_fit(z_vals, H_vals, sigma_vals, model_name, n_starts=N_MULTISTART, verbose=True):
    """Global fit (differential_evolution) followed by a Nelder-Mead polish."""
    bounds = get_bounds(model_name)
    print(f"  [{model_name}] Running differential evolution...")
    de_result = differential_evolution(
        chi2, bounds=bounds, args=(z_vals, H_vals, sigma_vals, model_name),
        seed=42, maxiter=200, tol=1e-8, polish=True, popsize=20,
    )
    best_x, best_chi2 = de_result.x, de_result.fun

    print(f"  [{model_name}] Running {n_starts} multi-start local optimizations...")
    rng = np.random.default_rng(42)
    starts = [best_x] + [
        [rng.uniform(lo, hi) for (lo, hi) in bounds] for _ in range(n_starts)
    ]

    local_results = []
    for x0 in tqdm(starts, desc=f"  [{model_name}] Local optimizations", disable=not verbose):
        res = minimize(chi2, x0, args=(z_vals, H_vals, sigma_vals, model_name),
                        method='Nelder-Mead', bounds=bounds,
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        if spread.size:
            print(f"  [{model_name}] Multi-start scan: {len(spread)}/{len(starts)} runs converged "
                  f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
            if spread.max() - spread.min() > 0.5:
                print(f"  [{model_name}] -> spread across starts suggests a degenerate/"
                      "multi-modal chi^2 surface")

    return best_x, best_chi2, de_result.success


# =============================================================================
# 5. UNCERTAINTIES: curve_fit covariance + MCMC
# =============================================================================

def fit_uncertainties_curvefit(z_vals, H_vals, sigma_vals, p0, model_name):
    bounds = get_bounds(model_name)
    lo = [b_[0] for b_ in bounds]
    hi = [b_[1] for b_ in bounds]

    def model_H_curvefit(z_array, H0, Om, b):
        H = model_H(z_array, H0, Om, b, model_name)
        if np.any(~np.isfinite(H)) or np.any(H <= 0):
            return np.full_like(np.atleast_1d(z_array), 1e6, dtype=float)
        return H

    popt, pcov = curve_fit(
        model_H_curvefit, z_vals, H_vals, p0=p0,
        sigma=sigma_vals, absolute_sigma=True, bounds=(lo, hi), maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr, pcov


def log_prior(theta, model_name):
    bounds = get_bounds(model_name)
    for val, (lo, hi) in zip(theta, bounds):
        if not (lo < val < hi):
            return -np.inf
    return 0.0


def log_likelihood(theta, z_vals, H_vals, sigma_vals, model_name):
    c = chi2(theta, z_vals, H_vals, sigma_vals, model_name)
    if c >= 1e11:
        return -np.inf
    return -0.5 * c


def log_prob(theta, z_vals, H_vals, sigma_vals, model_name):
    lp = log_prior(theta, model_name)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, z_vals, H_vals, sigma_vals, model_name)


def run_mcmc(best_x, z_vals, H_vals, sigma_vals, model_name,
             nwalkers=NWALKERS, nsteps=NSTEPS, discard=DISCARD, thin=THIN):
    ndim = 3
    bounds = get_bounds(model_name)
    b_lo, b_hi = bounds[2]
    spread = np.array([2.0, 0.05, 0.1 * (b_hi - b_lo)])
    pos = np.zeros((nwalkers, ndim))

    for i in range(nwalkers):
        pos[i] = best_x + spread * np.random.randn(ndim)
        for j, (lo, hi) in enumerate(bounds):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(z_vals, H_vals, sigma_vals, model_name)
    )
    print(f"  [{model_name}] Running emcee ({nwalkers} walkers x {nsteps} steps)...")
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
    plt.savefig(os.path.join(outdir, "walker_chains.png"), dpi=300)
    plt.close(fig)


# =============================================================================
# 6. PROFILE LIKELIHOOD & CONFIDENCE CONTOURS (all wrt b, the distortion
#    parameter -- the direct analogue of delta_lcdm_fit.py's delta profile)
# =============================================================================

def plot_chi2_profile_b(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
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
            return chi2([p2[0], p2[1], bb], z_vals, H_vals, sigma_vals, model_name)
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
    ax.set_title(rf'Profile likelihood: $\Delta\chi^2$ vs $b$ -- {MODEL_LABELS[model_name]}')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chi2_profile_b.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if b_lo68 is not None and b_hi68 is not None:
        print(f"  [{model_name}] b 1sigma profile interval: [{b_lo68:.4f}, {b_hi68:.4f}]")

    return bs, chi2_vals


def plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
                     vary=('b', 'Om'), n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour via a grid evaluation (no analytic fast path,
    same brentq-per-point cost as delta_lcdm_fit.py's contour function)."""
    idx = {'H0': 0, 'Om': 1, 'b': 2}
    ix, iy = idx[vary[0]], idx[vary[1]]
    iz = ({0, 1, 2} - {ix, iy}).pop()
    bounds = get_bounds(model_name)

    center = best_x[ix], best_x[iy]

    x_lo, x_hi = max(bounds[ix][0], center[0] * 0.3 if center[0] > 0 else bounds[ix][0]), \
        min(bounds[ix][1], center[0] * 2.2 if center[0] > 0 else bounds[ix][1])
    y_lo, y_hi = max(bounds[iy][0], center[1] * 0.3 if center[1] > 0 else bounds[iy][0]), \
        min(bounds[iy][1], center[1] * 2.2 if center[1] > 0 else bounds[iy][1])
    # If the varying parameter is b and best-fit b is near/below zero, the
    # "0.3x / 2.2x" scaling collapses -- fall back to a symmetric window.
    if x_hi - x_lo < 1e-6:
        x_lo, x_hi = max(bounds[ix][0], center[0] - 0.5), min(bounds[ix][1], center[0] + 0.5)
    if y_hi - y_lo < 1e-6:
        y_lo, y_hi = max(bounds[iy][0], center[1] - 0.5), min(bounds[iy][1], center[1] + 0.5)

    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)

    params_flat = np.zeros((n_grid * n_grid, 3))
    params_flat[:, ix] = X.ravel()
    params_flat[:, iy] = Y.ravel()
    params_flat[:, iz] = best_x[iz]

    print(f"  [{model_name}] Computing {n_grid}x{n_grid} grid for {vary[0]}-{vary[1]} contour...")
    chi2_flat = chi2_grid(params_flat, z_vals, H_vals, sigma_vals, model_name)
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
                 rf'-- {MODEL_LABELS[model_name]}')
    ax.legend()
    fig.tight_layout()

    fname = f'contour_{vary[0]}_{vary[1]}.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return X, Y, delta_chi2


def adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
                                vary=('b', 'Om'), n_grid_min=30, n_grid_max=80, outdir='.'):
    if not ADAPTIVE_CONTOURS:
        return plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
                                vary=vary, n_grid=CONTOUR_GRID, outdir=outdir)

    n_grid = n_grid_min
    X, Y, delta = plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
                                   vary=vary, n_grid=n_grid, outdir=outdir)

    grad_x = np.gradient(delta, axis=0)
    grad_y = np.gradient(delta, axis=1)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    if np.std(grad_mag) > 0.5 * np.mean(grad_mag) and n_grid < n_grid_max:
        n_grid = min(n_grid * 2, n_grid_max)
        print(f"  [{model_name}] Refining contour grid to {n_grid}x{n_grid}...")
        X, Y, delta = plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
                                       vary=vary, n_grid=n_grid, outdir=outdir)

    return X, Y, delta


# =============================================================================
# 7. HUBBLE DIAGRAMS
# =============================================================================

def plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, model_name, outdir='.'):
    H0_fit, Om_fit, b_fit = best_x
    z_smooth = np.linspace(0, z_vals.max() * 1.05, 300)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit, b_fit, model_name)
    H_at_data = model_H(z_vals, H0_fit, Om_fit, b_fit, model_name)
    residuals = H_vals - H_at_data

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', color='crimson',
                 ms=4, capsize=2, label='DESI data')
    ax1.plot(z_smooth, H_smooth, color='navy', lw=2,
              label=rf'{model_name} fit ($b={b_fit:.3f}$)')
    H_lcdm_smooth = H_lcdm(z_smooth, H0_fit, Om_fit)
    ax1.plot(z_smooth, H_lcdm_smooth, color='green', lw=1.5, ls='--',
              label=r'$\Lambda$CDM ($b=0$, same $H_0,\Omega_{m,0}$)')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title(f'Hubble diagram: best fit -- {MODEL_LABELS[model_name]}')
    ax1.legend()

    ax2.errorbar(z_vals, residuals, yerr=sigma_vals, fmt='o', color='crimson', ms=4, capsize=2)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_hubble_diagram_all_models(results, z_vals, H_vals, sigma_vals, lcdm_fit, outdir='.'):
    """Overlay all three f(T) models + LambdaCDM on one Hubble diagram."""
    z_smooth = np.linspace(0, z_vals.max() * 1.05, 300)
    H0_l, Om_l = lcdm_fit

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', color='k',
                 ms=4, capsize=2, label='DESI data', zorder=5)

    H_lcdm_smooth = H_lcdm(z_smooth, H0_l, Om_l)
    ax1.plot(z_smooth, H_lcdm_smooth, color='gray', lw=2, ls='--', label=r'$\Lambda$CDM')
    ax2.axhline(0, color='gray', lw=1.5, ls='--')

    for model_name in MODEL_LIST:
        best_x = results[model_name]['best_x']
        H0_fit, Om_fit, b_fit = best_x
        H_smooth = model_H(z_smooth, H0_fit, Om_fit, b_fit, model_name)
        H_at_data = model_H(z_vals, H0_fit, Om_fit, b_fit, model_name)
        ax1.plot(z_smooth, H_smooth, color=MODEL_COLORS[model_name], lw=2,
                  label=f'{model_name} ($b={b_fit:.3f}$)')
        ax2.plot(z_vals, H_vals - H_at_data, 'o', color=MODEL_COLORS[model_name], ms=4, alpha=0.7)

    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble diagram: all f(T) models vs $\\Lambda$CDM')
    ax1.legend(fontsize=9)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram_all_models.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MODEL COMPARISON: AIC / BIC / DIC across f1CDM, f2CDM, f3CDM, LambdaCDM
#    (mirrors the paper's Table II, Eqs. 51-53, but computed on H(z) only)
# =============================================================================

def compute_dic(flat_samples, chi2_func, chi2_args):
    """DIC = D(theta_bar) + 2*pD, pD = D_bar - D(theta_bar), D = chi^2
    (the -2*log-likelihood up to an additive constant that's identical
    across all models being compared here, so it cancels in delta-DIC)."""
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
            dic, pD = compute_dic(lcdm_result['flat_samples'], chi2_lcdm,
                                   (lcdm_result['z_vals'], lcdm_result['H_vals'], lcdm_result['sigma_vals']))
        else:
            k = 3
            chi2_best = results[name]['chi2_best']
            dic, pD = compute_dic(results[name]['flat_samples'], chi2,
                                   (results[name]['z_vals'], results[name]['H_vals'],
                                    results[name]['sigma_vals'], name))
        dof = n - k
        aic = chi2_best + 2 * k
        bic = chi2_best + k * np.log(n)
        stats[name] = dict(k=k, dof=dof, chi2=chi2_best, chi2_dof=chi2_best / dof,
                            aic=aic, bic=bic, dic=dic, pD=pD)

    aic_min = min(s['aic'] for s in stats.values())
    bic_min = min(s['bic'] for s in stats.values())
    dic_min = min(s['dic'] for s in stats.values())

    print("\n" + "=" * 92)
    print("MODEL COMPARISON TABLE (Omega-parametrization f(T) models vs LambdaCDM)")
    print("=" * 92)
    header = f"{'Model':<12}{'k':>4}{'dof':>6}{'chi2':>10}{'chi2/dof':>11}{'AIC':>10}{'dAIC':>9}{'BIC':>10}{'dBIC':>9}{'DIC':>10}{'dDIC':>9}"
    print(header)
    print("-" * 92)
    lines = [header, "-" * 92]
    for name in all_names:
        s = stats[name]
        d_aic = s['aic'] - aic_min
        d_bic = s['bic'] - bic_min
        d_dic = s['dic'] - dic_min
        row = (f"{name:<12}{s['k']:>4}{s['dof']:>6}{s['chi2']:>10.3f}{s['chi2_dof']:>11.3f}"
               f"{s['aic']:>10.3f}{d_aic:>9.3f}{s['bic']:>10.3f}{d_bic:>9.3f}{s['dic']:>10.3f}{d_dic:>9.3f}")
        print(row)
        lines.append(row)
    print("=" * 92)
    lines.append("=" * 92)

    lines.append("\nJeffreys-scale interpretation (paper's convention): "
                 "dIC<=2 statistically indistinguishable from the best model, "
                 "2<dIC<6 mild tension, dIC>=10 strong tension.")

    filename = os.path.join(outdir, 'model_comparison_table.txt')
    with open(filename, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nModel comparison table saved to: {filename}")

    return stats


# =============================================================================
# 9. CONSISTENCY CHECK: b -> 0 (or b -> 0+) recovers LambdaCDM
# =============================================================================

def consistency_check_b_zero(best_x, z_vals, H_vals, sigma_vals, model_name):
    """Evaluates chi^2 directly via model_H (bypassing the fit's b-bounds,
    which exist to keep the OPTIMIZER off multi-valued/singular regions,
    not to mark where the physics itself stops making sense) as b shrinks
    toward the LambdaCDM limit."""
    H0_fit, Om_fit, b_fit = best_x
    print(f"\n[{model_name}] Behaviour of chi^2 as b -> 0 (H0, Om fixed at best fit):")
    if model_name == 'f1CDM':
        fracs = [1.0, 0.5, 0.2, 0.1, 0.0]  # f1CDM is regular at b=0 exactly
    else:
        fracs = [1.0, 0.5, 0.2, 0.1, 0.01]  # f2/f3CDM: stop just short of the 1/b singularity
    for frac in fracs:
        bb = b_fit * frac if frac > 0 else 0.0
        H_model = model_H(z_vals, H0_fit, Om_fit, bb, model_name)
        if np.any(~np.isfinite(H_model)) or np.any(H_model <= 0):
            print(f"  b={bb:<9.4f} chi^2=  (invalid: model broke down at this b)")
            continue
        c = float(np.sum(((H_vals - H_model) / sigma_vals) ** 2))
        print(f"  b={bb:<9.4f} chi^2={c:.3f}")
    c_lcdm_direct = float(np.sum(((H_vals - H_lcdm(z_vals, H0_fit, Om_fit)) / sigma_vals) ** 2))
    print(f"  [cross-check] direct LambdaCDM chi^2 at same (H0,Om): {c_lcdm_direct:.3f} "
          f"(should be approached as b -> 0 above)")


# =============================================================================
# 10. EXPORTS
# =============================================================================

def export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, model_name, outdir='.'):
    H0_fit, Om_fit, b_fit = best_x
    H_best = model_H(z_vals, H0_fit, Om_fit, b_fit, model_name)
    residuals = H_vals - H_best

    z_smooth = np.linspace(0, z_vals.max() * 1.1, 200)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit, b_fit, model_name)

    data_filename = os.path.join(outdir, f'{model_name}_fit_results.txt')
    with open(data_filename, 'w') as f:
        f.write("# z, H_obs, sigma_H, H_model, residual\n")
        for zi, Hi, si, Hm, ri in zip(z_vals, H_vals, sigma_vals, H_best, residuals):
            f.write(f"{zi:.6f} {Hi:.6f} {si:.6f} {Hm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_filename}")

    curve_filename = os.path.join(outdir, f'{model_name}_smooth_curve.txt')
    with open(curve_filename, 'w') as f:
        f.write("# z, H_model(z)\n")
        for zi, Hi in zip(z_smooth, H_smooth):
            f.write(f"{zi:.6f} {Hi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_filename}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, model_name, outdir="."):
    filename = os.path.join(outdir, "fit_summary.txt")
    with open(filename, "w") as f:
        f.write(f"===== BEST FIT: {model_name} =====\n\n")
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


def validate_config():
    assert CONTOUR_GRID >= 20, "CONTOUR_GRID should be at least 20"
    assert PROFILE_POINTS >= 20, "PROFILE_POINTS should be at least 20"
    assert NWALKERS >= 16, "NWALKERS should be at least 16"
    assert NSTEPS >= 1000, "NSTEPS should be at least 1000"
    assert H0_BOUNDS[0] < H0_BOUNDS[1], "Invalid H0 bounds"
    assert OM_BOUNDS[0] < OM_BOUNDS[1], "Invalid Om bounds"
    for name, (lo, hi) in MODEL_B_BOUNDS.items():
        assert lo < hi, f"Invalid b bounds for {name}"
    for i in range(len(CONF_LEVELS_2D) - 1):
        assert CONF_LEVELS_2D[i] < CONF_LEVELS_2D[i + 1], "Confidence levels not increasing"


# =============================================================================
# 11. PER-MODEL PIPELINE
# =============================================================================

def run_one_model(model_name, z_vals, H_vals, sigma_vals, base_outdir):
    outdir = os.path.join(base_outdir, model_name)
    os.makedirs(outdir, exist_ok=True)
    print("\n" + "#" * 78)
    print(f"# MODEL: {MODEL_LABELS[model_name]}")
    print("#" * 78)

    print(f"\n--- [{model_name}] Best fit (global optimizer + multi-start) ---")
    best_x, chi2_best, converged = best_fit(z_vals, H_vals, sigma_vals, model_name)
    H0_fit, Om_fit, b_fit = best_x
    dof = len(z_vals) - 3
    print(f"  converged: {converged}")
    print(f"  H0 = {H0_fit:.4f}   Om = {Om_fit:.4f}   b = {b_fit:.4f}")
    print(f"  chi^2 = {chi2_best:.4f}  (chi^2/dof = {chi2_best/dof:.4f}, dof={dof})")

    print(f"\n--- [{model_name}] curve_fit covariance ---")
    perr = None
    try:
        popt, perr, pcov = fit_uncertainties_curvefit(z_vals, H_vals, sigma_vals, best_x, model_name)
        for name, val, err in zip(PARAM_NAMES, popt, perr):
            print(f"  {name:6s} = {val:.4f} +/- {err:.4f}")
        chi2_cf = chi2(popt, z_vals, H_vals, sigma_vals, model_name)
        if chi2_cf < chi2_best:
            best_x, chi2_best = popt, chi2_cf
    except Exception as e:
        print(f"  curve_fit uncertainty estimation failed: {e}")
        perr = np.full(3, np.nan)

    print(f"\n--- [{model_name}] MCMC posterior (emcee) ---")
    sampler, flat_samples = run_mcmc(best_x, z_vals, H_vals, sigma_vals, model_name)
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = percentiles[:, i]
        print(f"  {name:6s} = {med:.4f} (+{hi-med:.4f} / -{med-lo:.4f})")

    plot_walkers(sampler, model_name, outdir)

    print(f"\n--- [{model_name}] Corner plot ---")
    fig_corner = corner.corner(flat_samples, labels=[PARAM_LABELS[n] for n in PARAM_NAMES],
                                truths=list(best_x), show_titles=True)
    fig_corner.suptitle(MODEL_LABELS[model_name], y=1.02)
    fig_corner.savefig(os.path.join(outdir, f'corner_{model_name}.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    print(f"\n--- [{model_name}] Profile likelihood & contours for b ---")
    plot_chi2_profile_b(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name, outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
                                vary=('b', 'Om'), outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals, model_name,
                                vary=('b', 'H0'), outdir=outdir)

    print(f"\n--- [{model_name}] Hubble diagram ---")
    plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, model_name, outdir=outdir)

    print(f"\n--- [{model_name}] Export best-fit data ---")
    export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, model_name, outdir=outdir)

    consistency_check_b_zero(best_x, z_vals, H_vals, sigma_vals, model_name)

    write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, model_name, outdir=outdir)

    print(f"\n[{model_name}] Done. Outputs in: {outdir}")

    return {
        'best_x': best_x, 'chi2_best': chi2_best, 'flat_samples': flat_samples,
        'z_vals': z_vals, 'H_vals': H_vals, 'sigma_vals': sigma_vals,
    }


def run_lcdm_baseline(z_vals, H_vals, sigma_vals, outdir):
    """Quick LambdaCDM fit + MCMC, purely so it has an MCMC-based DIC on the
    same footing as the three f(T) models in the comparison table."""
    print("\n" + "#" * 78)
    print("# BASELINE: flat LambdaCDM")
    print("#" * 78)

    bounds = [H0_BOUNDS, (0.05, 0.95)]
    de_result = differential_evolution(chi2_lcdm, bounds=bounds,
                                        args=(z_vals, H_vals, sigma_vals), seed=42, tol=1e-8)
    H0_l, Om_l = de_result.x
    chi2_l = de_result.fun
    print(f"  H0 = {H0_l:.4f}   Om = {Om_l:.4f}   chi^2 = {chi2_l:.4f}")

    def log_prior_l(theta):
        H0, Om = theta
        if not (bounds[0][0] < H0 < bounds[0][1] and bounds[1][0] < Om < bounds[1][1]):
            return -np.inf
        return 0.0

    def log_prob_l(theta):
        lp = log_prior_l(theta)
        if not np.isfinite(lp):
            return -np.inf
        c = chi2_lcdm(theta, z_vals, H_vals, sigma_vals)
        if c >= 1e11:
            return -np.inf
        return lp - 0.5 * c

    ndim, nwalkers = 2, 24
    spread = np.array([2.0, 0.03])
    pos = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        pos[i] = np.array([H0_l, Om_l]) + spread * np.random.randn(ndim)
        pos[i, 0] = np.clip(pos[i, 0], bounds[0][0] + 1e-6, bounds[0][1] - 1e-6)
        pos[i, 1] = np.clip(pos[i, 1], bounds[1][0] + 1e-6, bounds[1][1] - 1e-6)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_l)
    print("  Running emcee for LambdaCDM baseline...")
    sampler.run_mcmc(pos, 2000, progress=True)
    flat_samples = sampler.get_chain(discard=400, thin=10, flat=True)

    fig_corner = corner.corner(flat_samples, labels=[r'$H_0$', r'$\Omega_{m,0}$'],
                                truths=[H0_l, Om_l], show_titles=True)
    fig_corner.savefig(os.path.join(outdir, 'corner_LambdaCDM.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    return {
        'best_x': np.array([H0_l, Om_l]), 'chi2_best': chi2_l, 'flat_samples': flat_samples,
        'z_vals': z_vals, 'H_vals': H_vals, 'sigma_vals': sigma_vals,
    }


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    validate_config()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results_omega_fT_desi")
    os.makedirs(outdir, exist_ok=True)
    print(f"Results will be saved to: {outdir}\n")

    setup_matplotlib()

    z_vals, H_vals, sigma_vals = load_all_data_memory_efficient()
    print(f"\nLoaded {len(z_vals)} data points.")
    print(f"Redshift range: {z_vals.min():.3f} to {z_vals.max():.3f}\n")

    results = {}
    for model_name in MODEL_LIST:
        results[model_name] = run_one_model(model_name, z_vals, H_vals, sigma_vals, outdir)

    lcdm_result = run_lcdm_baseline(z_vals, H_vals, sigma_vals, outdir)

    print("\n--- Combined Hubble diagram (all models) ---")
    plot_hubble_diagram_all_models(results, z_vals, H_vals, sigma_vals, lcdm_result['best_x'], outdir=outdir)

    print("\n--- Model comparison table (f1CDM, f2CDM, f3CDM, LambdaCDM) ---")
    stats = create_model_comparison_table(results, lcdm_result, len(z_vals), outdir=outdir)

    print(f"\nDone. All figures and results saved to: {outdir}")
    print("\nSummary of best-fit b (distortion parameter) per model:")
    for model_name in MODEL_LIST:
        b_fit = results[model_name]['best_x'][2]
        print(f"  {model_name:8s}: b = {b_fit:.4f}  (b=0 is the LambdaCDM limit)")


if __name__ == "__main__":
    main()