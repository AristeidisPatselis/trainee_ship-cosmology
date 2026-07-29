"""
delta4_lcdm_fit.py
===================
Modified Friedmann equation fit with delta FIXED at 4:

    H(z)^2 = H0^2 * [ Om * (1+z)^3 + (1-Om) * (H(z)/H0)^4 ]

This is the delta=4 sibling of delta_lcdm_fit.py: same model family
(H(z)^2 = H0^2*[Om*(1+z)^3 + (1-Om)*(H/H0)^delta]), same overall pipeline
(config, caching, global+multi-start optimizer, curve_fit uncertainties,
emcee MCMC, confidence contour, Hubble diagram, model-comparison table,
fit summary) -- the ONLY physics difference is that delta is pinned at 4
instead of being a free parameter, so the fit is over (H0, Om) alone.

Why this file still hand-rolls its own H(z) solver instead of just calling
delta_lcdm_fit.py's brentq-based model_H() with delta=4 plugged in:
at delta=4 the residual equation
    eq(H) = H - H0*sqrt(Om*(1+z)^3 + (1-Om)*(H/H0)^4)
is NOT monotonic in H, so the two-point bracket-widening trick that works
for the general-delta script can straddle a sign-preserving region instead
of the true root and silently return garbage. Luckily delta=4 makes the
equation exactly QUADRATIC in y = H^2:

    (1-Om)/H0^2 * y^2  -  y  +  H0^2*Om*(1+z)^3  =  0

so it has a closed-form solution -- no brentq/bracketing needed at all.
Of the two roots, only the "-" root reduces correctly to the
matter-domination limit; we use the numerically stable rationalized form
y = 2c / (1 + sqrt(1-4ac)) for it, which avoids cancellation when a is
small (Om close to 1).

One more delta=4-specific feature carried over: the equation only has a
real solution when 4*Om*(1-Om)*(1+z)^3 <= 1 for every z in the dataset
(H0 cancels out of this condition entirely). For typical cosmic-chronometer
redshift ranges this confines the viable Om region to a narrow sliver near
0 or 1, so BOUNDS and the optimizer/MCMC starting points below are built to
respect that instead of assuming a comfortable (0,1) interior like the
general-delta script can.

Parameter order is (H0, Om), matching delta_lcdm_fit.py's (H0, Om, delta)
convention with delta simply dropped.
"""

# NOTE TO SELF: this is the delta=4 sibling of delta_lcdm_fit.py -- if the
# general (free-delta) version ever changes its overall pipeline structure
# (caching, optimizer, uncertainty, MCMC, plotting conventions), mirror the
# change here too. The model solve itself (closed-form quadratic-in-H^2)
# is intentionally NOT shared with delta_lcdm_fit.py's brentq solver, for
# the non-monotonicity reason explained above.

# --- Standard library --------------------------------------------------------
import os
import warnings
from functools import lru_cache
from tqdm import tqdm

# --- Numerics / optimization --------------------------------------------------
import numpy as np
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.stats import chi2 as chi2_dist

# --- Plotting ------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib import rc

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

DELTA_FIXED = 4.0        # <-- the whole point of this script; not fitted

# --- DATA CONFIGURATION ---
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes/Data_Sets/'
Z_FILE = 'c_z_vals.txt'
H_FILE = 'c_H_vals.txt'
SIGMA_FILE = 'c_sigma_vals.txt'

# --- FIT CONFIGURATION ---
# NOTE: the Om range is pushed close to the edges (0, 1) on purpose. The
# delta=4 equation only has a real solution when
#     4 * Om * (1-Om) * (1+z)^3 <= 1
# for every z in the dataset (see discriminant_allowed_Om_range() below) --
# for typical cosmic-chronometer redshift ranges this confines the viable
# region to a narrow sliver near Om=0 or Om=1, so BOUNDS needs to actually
# reach those edges or the fit will find nothing.
BOUNDS = [(40.0, 100.0), (1e-4, 1.0 - 1e-4)]   # H0, Om
PARAM_NAMES = ['H0', 'Om']
PARAM_LABELS = {'H0': r'$H_0$', 'Om': r'$\Omega_{m,0}$'}

CONTOUR_GRID = 120   # only 2 free params -> can afford a finer grid than
                      # the 3-param (delta free) script

# emcee sampler settings
NWALKERS = 32
NSTEPS = 4000
DISCARD = 1000
THIN = 15

# Delta-chi^2 thresholds for 2 degrees of freedom (H0, Om jointly):
#   1 sigma -> 2.30, 2 sigma -> 6.18, 3 sigma -> 11.83
CONF_LEVELS_2D = [2.30, 6.18, 11.83]

# Reference H0 values from the literature, used only for the final comparison
# plot to visualize where this fit sits relative to the H0 tension.
LITERATURE_H0 = {
    "Planck 2018 (CMB)": (67.4, 0.5),
    "SH0ES 2022 (Local)": (73.04, 1.04),
}

# --- OPTIMIZATION CONFIGURATION ---
USE_CACHING = True
BATCH_SIZE = 100
N_MULTISTART = 8


# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

def setup_matplotlib():
    """Enable LaTeX only if a real render actually succeeds on this machine."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
        fig_test = plt.figure()
        plt.text(0.5, 0.5, r"$\delta$")
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
    only what follows the last ']' on each line.
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
    file discovery under DATA_DIR (same convention as delta_lcdm_fit.py)."""
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
# 2. MODEL: closed-form H(z) via delta=4 quadratic-in-H^2 (NO brentq needed)
# =============================================================================
#
# H^2 = H0^2 [ Om*(1+z)^3 + (1-Om)*(H/H0)^4 ]
#
# Unlike general delta (which needs an implicit brentq root-find, see
# delta_lcdm_fit.py), delta=4 makes this QUADRATIC in y = H^2:
#
#   (1-Om)/H0^2 * y^2  -  y  +  H0^2*Om*(1+z)^3  =  0
#   i.e.  a*y^2 - y + c = 0,  with  a = (1-Om)/H0^2,  c = H0^2*Om*(1+z)^3
#
# so it has an exact closed-form solution. (A naive two-point
# bracket-widening root-finder, like the one used for general delta,
# actually FAILS here: eq(H) is not monotonic for delta=4, so the bracket
# can straddle a sign-preserving region instead of the true root. The
# quadratic-in-H^2 trick sidesteps that completely.)
#
# Of the two roots of the quadratic, only the "-" root reduces correctly to
# the matter-domination limit (y -> H0^2*Om*(1+z)^3 as Om -> 1, i.e. a -> 0);
# the "+" root blows up as a -> 0 and is unphysical. We use the numerically
# stable rationalized form y = 2c / (1 + sqrt(1-4ac)) for the "-" root,
# which avoids catastrophic cancellation when a is small (Om close to 1).

def _H_single(z, H0, Om):
    """Exact H(z) for the delta=4 model."""
    a = (1.0 - Om) / H0 ** 2
    c = H0 ** 2 * Om * (1 + z) ** 3
    disc = 1.0 - 4.0 * a * c
    if disc < 0 or not np.isfinite(disc):
        return np.nan
    y = 2.0 * c / (1.0 + np.sqrt(disc))
    if y <= 0 or not np.isfinite(y):
        return np.nan
    return np.sqrt(y)


@lru_cache(maxsize=4096)
def _H_single_cached(z, H0, Om):
    return _H_single(z, H0, Om)


def model_H(z_eval, H0, Om):
    """Vectorized wrapper: solves for H at every z in z_eval (delta=4 fixed).

    Kept as a Python-level loop for parity with delta_lcdm_fit.py's
    model_H() interface, even though each individual evaluation here is a
    cheap closed-form solve rather than a brentq call.
    """
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))

    if USE_CACHING:
        out = [_H_single_cached(float(z), float(H0), float(Om)) for z in z_eval]
    else:
        out = [_H_single(float(z), float(H0), float(Om)) for z in z_eval]
    return np.array(out)


def H_lcdm(z, H0, Om):
    """Standard flat LambdaCDM (delta=0), used only as the baseline for
    the Hubble-diagram comparison curve and the AIC/BIC baseline."""
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


def discriminant_allowed_Om_range(z_vals):
    """The delta=4 equation has a real solution for H(z) only where
    4*Om*(1-Om)*(1+z)^3 <= 1 (this is exactly the disc = 1-4*a*c check in
    _H_single, expressed in terms of Om and z alone; H0 cancels out
    completely and plays no role here).

    Om*(1-Om) is a downward parabola peaking at Om=0.5, so this condition
    excludes a band around Om=0.5 and only allows Om below some Om_lo or
    above some Om_hi -- and the higher z climbs, the tighter (1+z)^3
    squeezes that allowed sliver toward Om=0 and Om=1. Since chi2() rejects
    a whole parameter point if ANY z fails, it's the single highest-z data
    point that sets the binding constraint.

    Returns (Om_lo, Om_hi, threshold) if a nontrivial band is excluded, or
    (None, None, threshold) if the full [0,1] range is already unrestricted
    (possible for a low-z dataset where even Om=0.5 stays viable).
    """
    z_max = np.max(z_vals)
    threshold = 1.0 / (4.0 * (1 + z_max) ** 3)   # max of Om*(1-Om) allowed
    if threshold >= 0.25:
        return None, None, threshold
    disc = 1.0 - 4.0 * threshold
    Om_lo = (1.0 - np.sqrt(disc)) / 2.0
    Om_hi = (1.0 + np.sqrt(disc)) / 2.0
    return Om_lo, Om_hi, threshold


# =============================================================================
# 3. CHI-SQUARED
# =============================================================================

def _within_bounds(params):
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, BOUNDS))


def chi2(params, z_vals, H_vals, sigma_vals):
    H0, Om = params
    if not _within_bounds(params):
        return 1e12
    H_model = model_H(z_vals, H0, Om)
    if np.any(~np.isfinite(H_model)) or np.any(H_model <= 0):
        return 1e12
    return float(np.sum(((H_vals - H_model) / sigma_vals) ** 2))


def chi2_grid(params_grid, z_vals, H_vals, sigma_vals):
    """Chi-squared over a grid of (H0, Om) pairs, for the confidence contour."""
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
# 4. BEST FIT: global optimizer + multi-start cross-check (discriminant-aware)
# =============================================================================

def _discriminant_aware_starts(z_vals, n_starts, rng):
    """Random starting points drawn preferentially from the physically
    allowed Om sliver (see discriminant_allowed_Om_range), since blind
    uniform draws over BOUNDS can land almost entirely in the excluded
    band around Om=0.5 once z climbs high enough."""
    Om_lo, Om_hi = BOUNDS[1]
    dlo, dhi, _ = discriminant_allowed_Om_range(z_vals)

    starts = []
    for _ in range(n_starts):
        H0_0 = rng.uniform(*BOUNDS[0])
        if dlo is None:
            Om_0 = rng.uniform(Om_lo, Om_hi)
        else:
            # draw from whichever allowed sliver (near 0 or near 1)
            if rng.uniform() < 0.5:
                Om_0 = rng.uniform(Om_lo, max(dlo, Om_lo))
            else:
                Om_0 = rng.uniform(min(dhi, Om_hi), Om_hi)
        starts.append([H0_0, Om_0])
    return starts


def best_fit(z_vals, H_vals, sigma_vals, n_starts=N_MULTISTART, verbose=True):
    """Global fit (differential_evolution) followed by a Nelder-Mead polish,
    with extra discriminant-aware starts to make sure the multi-start scan
    actually samples the (possibly narrow) physically allowed Om sliver."""
    print("  Running differential evolution...")
    de_result = differential_evolution(
        chi2, bounds=BOUNDS, args=(z_vals, H_vals, sigma_vals),
        seed=42, maxiter=300, tol=1e-8, polish=True, popsize=25,
    )
    best_x, best_chi2 = de_result.x, de_result.fun

    print(f"  Running {n_starts} multi-start local optimizations...")
    rng = np.random.default_rng(42)
    starts = [best_x] + _discriminant_aware_starts(z_vals, n_starts, rng)

    local_results = []
    for x0 in tqdm(starts, desc="  Local optimizations", disable=not verbose):
        res = minimize(chi2, x0, args=(z_vals, H_vals, sigma_vals),
                        method='Nelder-Mead', bounds=BOUNDS,
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if not np.isfinite(best_chi2):
        print("\n  CRITICAL: no (H0, Om) combination found gives a real, "
              "finite chi^2 for delta=4 with this dataset -- the delta=4 "
              "model may be excluded by the data's redshift range.")

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        if spread.size:
            print(f"  Multi-start scan: {spread.size}/{len(starts)} runs converged "
                  f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
            if spread.max() - spread.min() > 0.5:
                print("  -> spread across starts suggests a degenerate/multi-modal "
                      "chi^2 surface")
        else:
            print("  Multi-start scan: 0 runs converged to finite chi^2.")

    return best_x, best_chi2, de_result.success


# =============================================================================
# 5. UNCERTAINTIES: curve_fit covariance + MCMC
# =============================================================================

def model_H_curvefit(z_array, H0, Om):
    """Vectorized wrapper with the curve_fit-friendly (z, *params) signature."""
    H = model_H(z_array, H0, Om)
    if np.any(~np.isfinite(H)) or np.any(H <= 0):
        return np.full_like(np.atleast_1d(z_array), 1e6, dtype=float)
    return H


def fit_uncertainties_curvefit(z_vals, H_vals, sigma_vals, p0):
    lo = [b[0] for b in BOUNDS]
    hi = [b[1] for b in BOUNDS]
    popt, pcov = curve_fit(
        model_H_curvefit, z_vals, H_vals, p0=p0,
        sigma=sigma_vals, absolute_sigma=True, bounds=(lo, hi), maxfev=20000,
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
    ndim = 2

    # Jitter scale starts small and shrinks further if needed -- with the
    # allowed Om region potentially a very narrow sliver (see
    # discriminant_allowed_Om_range), a walker can easily jitter itself
    # into the excluded band and start at log_prob = -inf, where it would
    # never move.
    base_spread = np.array([2.0, 0.05])
    pos = np.tile(np.asarray(best_x, dtype=float), (nwalkers, 1))

    for i in range(nwalkers):
        spread = base_spread.copy()
        for _ in range(50):
            candidate = best_x + spread * np.random.randn(ndim)
            for j, (lo, hi) in enumerate(BOUNDS):
                candidate[j] = np.clip(candidate[j], lo + 1e-6, hi - 1e-6)
            if np.isfinite(log_prob(candidate, z_vals, H_vals, sigma_vals)):
                pos[i] = candidate
                break
            spread *= 0.5   # shrink and retry if we keep landing outside the sliver

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(z_vals, H_vals, sigma_vals)
    )
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    return sampler, flat_samples


def plot_walkers(sampler, outdir="."):
    chain = sampler.get_chain()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i in range(2):
        for walker in range(chain.shape[1]):
            axes[i].plot(
                chain[:, walker, i],
                alpha=0.3,
                lw=0.5
            )
        axes[i].set_ylabel(PARAM_LABELS[PARAM_NAMES[i]])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "walker_chains_delta4.png"), dpi=300)
    plt.close()


# =============================================================================
# 6. CONFIDENCE CONTOUR (H0 vs Om -- the full 2D parameter space)
# =============================================================================
# With only 2 free parameters and delta fixed, this single contour IS the
# full joint constraint (no third parameter to profile/slice against, unlike
# delta_lcdm_fit.py's contour_delta_Om / contour_delta_H0 slices).

def plot_confidence_contour_H0_Om(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                   n_grid=CONTOUR_GRID, outdir='.'):
    H0_fit, Om_fit = best_x
    Om_lo_disc, Om_hi_disc, _ = discriminant_allowed_Om_range(z_vals)

    H0_grid = np.linspace(BOUNDS[0][0], BOUNDS[0][1], n_grid)
    Om_grid = np.linspace(BOUNDS[1][0], BOUNDS[1][1], n_grid)
    H0_mesh, Om_mesh = np.meshgrid(H0_grid, Om_grid)

    params_flat = np.column_stack([H0_mesh.ravel(), Om_mesh.ravel()])

    print(f"  Computing {n_grid}x{n_grid} grid for H0-Om confidence contour...")
    chi2_flat = chi2_grid(params_flat, z_vals, H_vals, sigma_vals)
    CHI2 = chi2_flat.reshape(n_grid, n_grid)
    delta_chi2 = CHI2 - chi2_best
    # replace non-finite (no-real-solution) cells with a large-but-finite
    # placeholder rather than passing inf straight to contourf/contour
    delta_chi2_plot = np.where(np.isfinite(delta_chi2), delta_chi2, CONF_LEVELS_2D[-1] * 5)

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(H0_mesh, Om_mesh, delta_chi2_plot, levels=CONF_LEVELS_2D,
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={CONF_LEVELS_2D[0]: r'1$\sigma$',
                       CONF_LEVELS_2D[1]: r'2$\sigma$',
                       CONF_LEVELS_2D[2]: r'3$\sigma$'})
    ax.contourf(H0_mesh, Om_mesh, delta_chi2_plot,
                levels=[0, *CONF_LEVELS_2D, max(delta_chi2_plot.max(), CONF_LEVELS_2D[-1] + 1)],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    if Om_lo_disc is not None:
        # H0-independent exclusion band: delta=4 has no real solution here
        # for ANY H0 (see discriminant_allowed_Om_range).
        ax.axhspan(Om_lo_disc, Om_hi_disc, color='black', alpha=0.25, hatch='//',
                   label='no real solution (any $H_0$)')
    ax.plot(H0_fit, Om_fit, 'k*', ms=14, label='best fit')
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_ylabel(r'$\Omega_{m,0}$')
    ax.set_title(r'$\Delta\chi^2$ contours: $H_0$ vs $\Omega_{m,0}$ ($\delta=4$ fixed)')
    ax.legend()
    fig.tight_layout()

    fname = 'contour_H0_Om_delta4.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return H0_mesh, Om_mesh, delta_chi2


# =============================================================================
# 7. HUBBLE DIAGRAM
# =============================================================================

def plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir='.'):
    H0_fit, Om_fit = best_x
    z_smooth = np.linspace(0, z_vals.max() * 1.05, 300)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit)
    H_at_data = model_H(z_vals, H0_fit, Om_fit)
    residuals = H_vals - H_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', color='crimson',
                 ms=4, capsize=2, label='Cosmic chronometer data')
    ax1.plot(z_smooth, H_smooth, color='navy', lw=2,
              label=r'model fit ($\delta=4$ fixed)')
    H_lcdm_smooth = H_lcdm(z_smooth, H0_fit, Om_fit)
    ax1.plot(z_smooth, H_lcdm_smooth, color='green', lw=1.5, ls='--',
              label=r'$\Lambda$CDM ($\delta=0$, same $H_0,\Omega_{m,0}$)')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title(r'Hubble diagram: best fit ($\delta=4$ fixed)')
    ax1.legend()

    ax2.errorbar(z_vals, residuals, yerr=sigma_vals, fmt='o', color='crimson', ms=4, capsize=2)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram_delta4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_H0_tension_comparison(best_x, perr, outdir='.'):
    """Compare this fit's H0 against a couple of literature reference values."""
    H0_fit, _ = best_x
    H0_err = perr[0] if perr is not None else 0.0

    all_vals = {"This work ($\\delta=4$ fit)": (H0_fit, H0_err, "crimson")}
    for name, (val, err) in LITERATURE_H0.items():
        all_vals[name] = (val, err, "steelblue" if "Planck" in name else "darkorange")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(all_vals.items()):
        ax.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)
        ax.axvspan(val - err, val + err, color=color, alpha=0.1)
    ax.set_yticks(range(len(all_vals)))
    ax.set_yticklabels(all_vals.keys())
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_title(r'$H_0$: this fit ($\delta=4$) vs. literature')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'H0_tension_comparison_delta4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MODEL COMPARISON TABLE
# =============================================================================

def create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir='.'):
    """Create a comprehensive model comparison table with all statistics.

    NOTE: unlike delta_lcdm_fit.py (free delta, k=3) vs LambdaCDM (k=2),
    here BOTH models have the same number of free parameters (k=2: H0, Om
    -- delta is fixed in both cases, just to different values: 4 here,
    0 for LambdaCDM). So the AIC/BIC penalty terms are identical and the
    comparison collapses to a straight chi^2 comparison.
    """
    n = len(z_vals)
    dof_model = n - 2
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

    aic_model, bic_model, chi2_dof_model = calculate_stats(chi2_best, 2, dof_model)
    aic_lcdm, bic_lcdm, chi2_dof_lcdm = calculate_stats(chi2_lcdm_best, 2, dof_lcdm)

    delta_aic = aic_model - aic_lcdm
    delta_bic = bic_model - bic_lcdm

    table_data = [
        ['Model', 'delta=4-LCDM', 'LambdaCDM'],
        ['H0', f'{best_x[0]:.2f}', f'{H0_l:.2f}'],
        ['Ωm,0', f'{best_x[1]:.3f}', f'{Om_l:.3f}'],
        ['δ', f'{DELTA_FIXED:.1f} (fixed)', '0 (fixed)'],
        ['χ²', f'{chi2_best:.2f}', f'{chi2_lcdm_best:.2f}'],
        ['k', '2', '2'],
        ['dof', f'{dof_model}', f'{dof_lcdm}'],
        ['χ²/dof', f'{chi2_dof_model:.2f}', f'{chi2_dof_lcdm:.2f}'],
        ['AIC', f'{aic_model:.2f}', f'{aic_lcdm:.2f}'],
        ['ΔAIC', f'{delta_aic:+.2f}', '0 (reference)'],
        ['BIC', f'{bic_model:.2f}', f'{bic_lcdm:.2f}'],
        ['ΔBIC', f'{delta_bic:+.2f}', '0 (reference)'],
    ]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE (delta=4 fixed)")
    print("=" * 80)

    col_widths = [max(len(row[i]) for row in table_data) + 2 for i in range(3)]

    print("│" + "│".join(f"{col:^{col_widths[i]}}" for i, col in enumerate(['Parameter', 'delta=4-LCDM', 'LambdaCDM'])) + "│")
    print("├" + "─" * col_widths[0] + "┼" + "─" * col_widths[1] + "┼" + "─" * col_widths[2] + "┤")

    for row in table_data[1:]:
        print("│" + "│".join(f"{row[i]:^{col_widths[i]}}" for i in range(3)) + "│")

    print("=" * 80)
    print("NOTE: both models have k=2 free params here (delta fixed in both), "
          "so Delta-AIC/BIC reduce to the raw chi^2 difference.")

    filename = os.path.join(outdir, 'model_comparison_table_delta4.txt')
    with open(filename, 'w') as f:
        f.write("MODEL COMPARISON TABLE (delta=4 fixed)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Parameter':<15} {'delta=4-LCDM':<20} {'LambdaCDM':<20}\n")
        f.write("-" * 80 + "\n")
        for row in table_data[1:]:
            f.write(f"{row[0]:<15} {row[1]:<20} {row[2]:<20}\n")
        f.write("=" * 80 + "\n")
        f.write("\nNOTE: both models have k=2 free params here (delta fixed in "
                "both cases, just to different values), so Delta-AIC/BIC "
                "reduce to the raw chi^2 difference.\n")

        f.write("\nINTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        if delta_aic < -2:
            f.write("✓ delta=4 model is strongly preferred by AIC\n")
        elif delta_aic < 0:
            f.write("✓ delta=4 model is slightly preferred by AIC\n")
        elif delta_aic < 2:
            f.write("○ Models are essentially equivalent by AIC\n")
        else:
            f.write("✗ LambdaCDM is preferred by AIC\n")

        if delta_bic < -2:
            f.write("✓ delta=4 model is strongly preferred by BIC\n")
        elif delta_bic < 0:
            f.write("✓ delta=4 model is slightly preferred by BIC\n")
        elif delta_bic < 2:
            f.write("○ Models are essentially equivalent by BIC\n")
        else:
            f.write("✗ LambdaCDM is preferred by BIC\n")

    print(f"\nModel comparison table saved to: {filename}")

    return table_data


# =============================================================================
# 9. ADDITIONAL PLOTS AND EXPORTS
# =============================================================================

def discriminant_report(z_vals):
    """Print the delta=4 real-solution constraint on Om for this dataset
    (see discriminant_allowed_Om_range) -- purely diagnostic/explanatory."""
    Om_lo, Om_hi, threshold = discriminant_allowed_Om_range(z_vals)
    print("\n--- Discriminant check (delta=4 real-solution constraint) ---")
    if Om_lo is None:
        print(f"  Full Om in (0,1) admits a real solution for all z in this "
              f"dataset (threshold {threshold:.4f} >= 0.25 -> unrestricted).")
    else:
        print(f"  For this dataset's max redshift, the delta=4 equation only "
              f"has a real H(z) solution when\n"
              f"    Om <= {Om_lo:.5f}   or   Om >= {Om_hi:.5f}\n"
              f"  (Om in between is EXCLUDED for every H0 -- this is a "
              f"genuine feature of the delta=4 model, not a numerical "
              f"artifact.)")


def export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, outdir='.'):
    """Export the best-fit model predictions and residuals to text files."""
    H0_fit, Om_fit = best_x
    H_best = model_H(z_vals, H0_fit, Om_fit)
    residuals = H_vals - H_best

    z_smooth = np.linspace(0, z_vals.max() * 1.1, 200)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit)

    header = "# z, H_obs, sigma_H, H_model, residual\n"
    data_filename = os.path.join(outdir, 'delta4_lcdm_fit_results.txt')
    with open(data_filename, 'w') as f:
        f.write(header)
        for zi, Hi, si, Hm, ri in zip(z_vals, H_vals, sigma_vals, H_best, residuals):
            f.write(f"{zi:.6f} {Hi:.6f} {si:.6f} {Hm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_filename}")

    curve_filename = os.path.join(outdir, 'delta4_lcdm_smooth_curve.txt')
    with open(curve_filename, 'w') as f:
        f.write("# z, H_model(z)\n")
        for zi, Hi in zip(z_smooth, H_smooth):
            f.write(f"{zi:.6f} {Hi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_filename}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir="."):
    filename = os.path.join(outdir, "fit_summary_delta4.txt")

    with open(filename, "w") as f:
        f.write("===== BEST FIT (delta=4 fixed) =====\n\n")
        for n, v, e in zip(PARAM_NAMES, best_x, perr):
            f.write(f"{n:8s} = {v:.6f} +/- {e:.6f}\n")
        f.write(f"{'delta':8s} = {DELTA_FIXED:.6f} (fixed, not a free parameter)\n")
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


def validate_config():
    """Validate configuration parameters."""
    assert CONTOUR_GRID >= 20, "CONTOUR_GRID should be at least 20"
    assert NWALKERS >= 16, "NWALKERS should be at least 16"
    assert NSTEPS >= 1000, "NSTEPS should be at least 1000"

    assert BOUNDS[0][0] < BOUNDS[0][1], "Invalid H0 bounds"
    assert BOUNDS[1][0] < BOUNDS[1][1], "Invalid Om bounds"

    for i in range(len(CONF_LEVELS_2D) - 1):
        assert CONF_LEVELS_2D[i] < CONF_LEVELS_2D[i + 1], "Confidence levels not increasing"


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    """Main function, structured to mirror delta_lcdm_fit.py's pipeline,
    with delta fixed at DELTA_FIXED instead of being a free parameter."""
    validate_config()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results_delta4")
    os.makedirs(outdir, exist_ok=True)
    print(f"Results will be saved to: {outdir}\n")

    setup_matplotlib()

    z_vals, H_vals, sigma_vals = load_all_data_memory_efficient()
    print(f"\nLoaded {len(z_vals)} data points.")
    print(f"Redshift range: {z_vals.min():.3f} to {z_vals.max():.3f}")
    print(f"\ndelta is FIXED at {DELTA_FIXED} for this run (not a free parameter).")

    discriminant_report(z_vals)

    print("\n--- Best fit (global optimizer + multi-start cross-check) ---")
    best_x, chi2_best, converged = best_fit(z_vals, H_vals, sigma_vals)
    H0_fit, Om_fit = best_x
    dof = len(z_vals) - 2
    print(f"  converged: {converged}")
    print(f"  H0    = {H0_fit:.4f}")
    print(f"  Om    = {Om_fit:.4f}")
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
    except Exception as e:
        print(f"  curve_fit uncertainty estimation failed: {e}")

    print("\n--- MCMC posterior (emcee) ---")
    sampler, flat_samples = run_mcmc(best_x, z_vals, H_vals, sigma_vals)
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    for i, name in enumerate(PARAM_NAMES):
        lo, med, hi = percentiles[:, i]
        print(f"  {name:6s} = {med:.4f} (+{hi-med:.4f} / -{med-lo:.4f})")

    plot_walkers(sampler, outdir)

    print("\n--- Corner plot ---")
    fig_corner = corner.corner(
        flat_samples, labels=[PARAM_LABELS[n] for n in PARAM_NAMES],
        truths=list(best_x), show_titles=True,
    )
    fig_corner.savefig(os.path.join(outdir, 'corner_delta4_lcdm.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    print("\n--- Confidence contour (H0 vs Om, delta=4 fixed) ---")
    H0_mesh, Om_mesh, delta_chi2 = plot_confidence_contour_H0_Om(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)
    print("  Saved: contour_H0_Om_delta4.png")

    # Save contour data for later comparison
    contour_data = {'X': H0_mesh, 'Y': Om_mesh, 'delta_chi2': delta_chi2}
    np.save(os.path.join(outdir, 'contour_H0_Om_delta4.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(outdir, 'contour_H0_Om_delta4.npy')}")

    print("\n--- Hubble diagram ---")
    plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir=outdir)
    print("  Saved: hubble_diagram_delta4.png")

    print("\n--- H0 tension comparison vs literature ---")
    plot_H0_tension_comparison(best_x, perr, outdir=outdir)
    print("  Saved: H0_tension_comparison_delta4.png")

    print("\n--- Export best-fit data ---")
    export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, outdir=outdir)

    print("\n--- Model Comparison Table ---")
    table_data = create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)

    print(f"\nDone. All figures and results saved to: {outdir}")

    if perr is not None and 'pcov' in locals():
        corr = pcov / np.outer(perr, perr)
        print("\nCorrelation matrix")
        print("--------------------------------")
        for row in corr:
            print(" ".join(f"{x:8.3f}" for x in row))
        np.savetxt(
            os.path.join(outdir, "correlation_matrix_delta4.txt"),
            corr,
            fmt="%.6f"
        )
    else:
        perr = np.full(2, np.nan)

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