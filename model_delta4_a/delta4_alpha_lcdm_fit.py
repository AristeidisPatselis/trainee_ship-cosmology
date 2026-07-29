"""
delta4_alpha_lcdm_fit.py
========================
Combined modified Friedmann equation, delta FIXED at 4, alpha FREE:

    H(z)^2 = H0^2 [ Om*(1+z)^3 + (1-Om)*(H(z)/H0)^4 ]  -  alpha*(1+z)*H(z)*dH/dz

Free parameters: (H0, Om, alpha), with delta=4 fixed. alpha=0 collapses the
correction term and recovers the *pure* delta=4 algebraic model from
delta4_lcdm_fit.py (which has its own restrictive discriminant -- see the
note in that script and in consistency_check_alpha_small() below).

Solved via the same u = H^2 substitution trick as H_dot_lcdm_fit.py: this
turns the H*dH/dz term into a first-order ODE for u(z), and the added
delta=4 term is just an explicit algebraic piece of the ODE's right-hand
side (no implicit root-finding needed for it, unlike the pure algebraic
delta4_lcdm_fit.py case).

Pipeline: global optimizer (differential_evolution) + multi-start
Nelder-Mead polish, curve_fit covariance + full MCMC posterior, profile
likelihood chi^2(alpha), 2D Delta-chi^2 contours in (alpha, Om) and
(alpha, H0), corner plot, Hubble diagram, AIC/BIC vs baseline LambdaCDM.
"""

# --- Standard library --------------------------------------------------------
import os
import warnings
from functools import lru_cache
from tqdm import tqdm

# --- Numerics / optimization --------------------------------------------------
import numpy as np
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.integrate import solve_ivp

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

DELTA_FIXED = 4.0        # fixed dark-energy exponent; alpha is what's fit here

# --- DATA CONFIGURATION ---
# Override with: DELTA4_DATA_DIR=/some/path python delta4_alpha_lcdm_fit.py
# (keeps the script runnable on another machine / after a folder move without editing code)
DATA_DIR = os.environ.get(
    'DELTA4_DATA_DIR',
    '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes/Data_Sets/'
)
Z_FILE = 'c_z_vals.txt'
H_FILE = 'c_H_vals.txt'
SIGMA_FILE = 'c_sigma_vals.txt'

# Prior/search box for (H0, Om, alpha). Also used as hard bounds: any point
# outside gets chi^2 = +inf / log-prob = -inf.
BOUNDS = [(40.0, 100.0), (0.01, 0.99), (0.01, 6.0)]   # H0, Om, alpha
PARAM_NAMES = ['H0', 'Om', 'alpha']
PARAM_LABELS = {'H0': r'$H_0$', 'Om': r'$\Omega_{m,0}$', 'alpha': r'$\alpha$'}

CONTOUR_GRID = 60        # grid resolution (per axis) for Delta-chi^2 contours
PROFILE_POINTS = 60       # number of alpha values in the 1D profile likelihood

# emcee sampler settings
NWALKERS = 32
NSTEPS = 3000
DISCARD = 500             # burn-in steps discarded before computing statistics
THIN = 15

# Delta-chi^2 thresholds
CONF_LEVELS_2D = [2.30, 6.18, 11.83]
CONF_LEVELS_1D = [1.0, 4.0, 9.0]

# Reference H0 values from the literature for diagnostic plotting
LITERATURE_H0 = {
    "Planck 2018 (CMB)": (67.4, 0.5),
    "SH0ES 2022 (Local)": (73.04, 1.04),
}

# --- OPTIMIZATION CONFIGURATION ---
USE_CACHING = True
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
    """Load one numeric value per line, with recursive search."""
    filepath = find_file_recursively(filename, data_dir)
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.split(']')[-1].strip()
            if clean_line:
                data.append(float(clean_line))
    return np.array(data)


def load_all_data_memory_efficient():
    """Memory-efficient data loading using numpy's loadtxt with recursive discovery."""
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

def _rhs_u(z, u, H0, Om, alpha):
    u_safe = min(max(u[0], 1e-8), 1e12)   # guard against a bad step blowing up
    x = 1.0 + z
    de_term = H0 ** 2 * (1 - Om) * (u_safe / H0 ** 2) ** 2   # delta=4 -> squared
    dudz = (2.0 / (alpha * x)) * (H0 ** 2 * Om * x ** 3 + de_term - u_safe)
    return [np.clip(dudz, -1e12, 1e12)]


def _solve_ode_uncached(z_tuple, H0, Om, alpha):
    """Single source of truth for the u=H^2 ODE solve. Returns a tuple of
    H(z) values (or all-NaN on any failure/invalid-parameter case)."""
    z_eval = np.array(z_tuple)
    if alpha <= 0 or H0 <= 0 or not (0 < Om < 1):
        return tuple([np.nan] * len(z_eval))

    z_max = max(z_eval.max(), 1e-6)
    t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))

    try:
        sol = solve_ivp(
            _rhs_u, (0.0, z_max), [H0 ** 2],
            args=(H0, Om, alpha),
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


_solve_ode_cached = lru_cache(maxsize=2048)(_solve_ode_uncached)


def model_H(z_eval, H0, Om, alpha):
    """Solve the ODE for u=H^2 and return H(z) at the requested redshifts.

    Both the cached and uncached paths call the same _solve_ode_uncached
    implementation, so there's only one place the ODE-solving logic can
    drift or develop a bug.
    """
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    z_tuple = tuple(z_eval)
    solver = _solve_ode_cached if USE_CACHING else _solve_ode_uncached
    return np.array(solver(z_tuple, float(H0), float(Om), float(alpha)))


def H_delta4_only(z, H0, Om):
    """The pure algebraic delta=4 model (alpha=0 'limit'), for reference /
    consistency checks. Closed form -- see delta4_lcdm_fit.py for the full
    formulation of this quadratic-in-H^2 solution."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    a = (1.0 - Om) / H0 ** 2
    c = H0 ** 2 * Om * (1 + z) ** 3
    disc = 1.0 - 4.0 * a * c
    H = np.full_like(z, np.nan)
    ok = disc >= 0
    y = np.where(ok, 2.0 * c / (1.0 + np.sqrt(np.clip(disc, 0, None))), np.nan)
    H[ok & (y > 0)] = np.sqrt(y[ok & (y > 0)])
    return H


def H_lcdm(z, H0, Om):
    """Standard flat LambdaCDM, used only as the baseline for AIC/BIC."""
    return H0 * np.sqrt(Om * (1 + z) ** 3 + (1 - Om))


# =============================================================================
# 3. CHI-SQUARED
# =============================================================================

def _within_bounds(params):
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, BOUNDS))


def chi2(params, z_vals, H_vals, sigma_vals):
    H0, Om, alpha = params
    if not _within_bounds(params):
        return 1e12
    H_theory = model_H(z_vals, H0, Om, alpha)
    if np.any(~np.isfinite(H_theory)):
        return 1e12
    return float(np.sum(((H_vals - H_theory) / sigma_vals) ** 2))


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
        seed=42, maxiter=200, tol=1e-8, polish=True, popsize=20,
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
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        if spread.size:
            print(f"Multi-start scan: {len(spread)}/{len(starts)} runs converged "
                  f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
            if spread.max() - spread.min() > 0.5:
                print("  -> spread across starts suggests a degenerate/multi-modal "
                      "chi^2 surface (expect this in (Om, alpha) -- see the contour "
                      "plot); trust the global (differential_evolution) result.")
        else:
            print("Multi-start scan: no local run converged to a finite chi^2 "
                  "(relying on the differential_evolution result alone).")

    return best_x, best_chi2, de_result.success


# =============================================================================
# 5. UNCERTAINTIES: curve_fit covariance + MCMC
# =============================================================================

def model_H_curvefit(z_array, H0, Om, alpha):
    """Vectorized wrapper with the curve_fit-friendly (z, *params) signature."""
    H = model_H(z_array, H0, Om, alpha)
    if np.any(~np.isfinite(H)):
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
    ndim = 3
    base_spread = np.array([2.0, 0.05, 0.1])   # small Gaussian ball around best fit
    pos = np.tile(best_x, (nwalkers, 1)).astype(float)
    for w in range(nwalkers):
        spread = base_spread.copy()
        for _ in range(50):
            candidate = best_x + spread * np.random.randn(ndim)
            for j, (lo, hi) in enumerate(BOUNDS):
                candidate[j] = np.clip(candidate[j], lo + 1e-6, hi - 1e-6)
            if np.isfinite(log_prob(candidate, z_vals, H_vals, sigma_vals)):
                pos[w] = candidate
                break
            spread *= 0.5   # shrink and retry if we keep landing at -inf

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(z_vals, H_vals, sigma_vals)
    )
    sampler.run_mcmc(pos, nsteps, progress=True)
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    return sampler, flat_samples


def plot_walkers(sampler, outdir="."):
    chain = sampler.get_chain()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        for walker in range(chain.shape[1]):
            axes[i].plot(chain[:, walker, i], alpha=0.3, lw=0.5)
        axes[i].set_ylabel(PARAM_LABELS[PARAM_NAMES[i]])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "walker_chains_delta4_alpha.png"), dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# 6. PROFILE LIKELIHOOD & CONFIDENCE CONTOURS
# =============================================================================

def plot_chi2_profile_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                             n_points=PROFILE_POINTS, outdir='.'):
    """1D profile chi^2(alpha): H0 and Om are re-fit at every alpha."""
    H0_fit, Om_fit, alpha_fit = best_x
    alpha_lo = max(BOUNDS[2][0], alpha_fit * 0.3)
    alpha_hi = min(BOUNDS[2][1], alpha_fit * 2.5)
    alphas = np.linspace(alpha_lo, alpha_hi, n_points)

    chi2_vals = np.empty(n_points)
    for i, a in enumerate(alphas):
        def chi2_reduced(p2):
            return chi2([p2[0], p2[1], a], z_vals, H_vals, sigma_vals)
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
        ax.text(alphas[-1], level, label, va='bottom', ha='right',
                fontsize=9, color='gray')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\Delta\chi^2(\alpha)$')
    ax.set_title(r'Profile likelihood: $\Delta\chi^2$ vs $\alpha$ ($\delta=4$ fixed, '
                 r'$H_0,\,\Omega_{m,0}$ refit at each point)')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chi2_profile_alpha_delta4_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if alpha_lo68 is not None and alpha_hi68 is not None:
        print(f"  alpha 1sigma profile interval: [{alpha_lo68:.4f}, {alpha_hi68:.4f}]")

    return alphas, chi2_vals


def plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                     vary=('alpha', 'Om'), n_grid=CONTOUR_GRID, outdir='.'):
    """Delta-chi^2 contour in two of the three parameters, with the third
    held fixed at its best-fit value."""
    idx = {'H0': 0, 'Om': 1, 'alpha': 2}
    ix, iy = idx[vary[0]], idx[vary[1]]
    iz = ({0, 1, 2} - {ix, iy}).pop()

    center = best_x[ix], best_x[iy]

    x_lo, x_hi = max(BOUNDS[ix][0], center[0] * 0.3), min(BOUNDS[ix][1], center[0] * 2.2)
    y_lo, y_hi = max(BOUNDS[iy][0], center[1] * 0.3), min(BOUNDS[iy][1], center[1] * 2.2)
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    CHI2 = np.empty_like(X)

    params = np.array(best_x, dtype=float)
    for i in range(n_grid):
        for j in range(n_grid):
            params[ix], params[iy] = X[i, j], Y[i, j]
            params[iz] = best_x[iz]
            CHI2[i, j] = chi2(params, z_vals, H_vals, sigma_vals)

    delta_chi2 = CHI2 - chi2_best

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(X, Y, delta_chi2, levels=CONF_LEVELS_2D,
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={CONF_LEVELS_2D[0]: r'1$\sigma$',
                       CONF_LEVELS_2D[1]: r'2$  \sigma$',
                       CONF_LEVELS_2D[2]: r'3$\sigma$'})
    ax.contourf(X, Y, delta_chi2,
                levels=[0, *CONF_LEVELS_2D, max(delta_chi2.max(), CONF_LEVELS_2D[-1] + 1)],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    ax.plot(center[0], center[1], 'k*', ms=14, label='best fit')
    ax.set_xlabel(PARAM_LABELS[vary[0]])
    ax.set_ylabel(PARAM_LABELS[vary[1]])
    ax.set_title(rf'$\Delta\chi^2$ contours: {PARAM_LABELS[vary[0]]} vs {PARAM_LABELS[vary[1]]} '
                 rf'($\delta=4$ fixed, {PARAM_LABELS[PARAM_NAMES[iz]]} fixed at best fit)')
    ax.legend()
    fig.tight_layout()
    fname = f'contour_{vary[0]}_{vary[1]}_delta4_alpha.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return X, Y, delta_chi2


# =============================================================================
# 7. HUBBLE DIAGRAM
# =============================================================================

def plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir='.'):
    H0_fit, Om_fit, alpha_fit = best_x
    z_smooth = np.linspace(0, z_vals.max() * 1.05, 300)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit, alpha_fit)
    H_at_data = model_H(z_vals, H0_fit, Om_fit, alpha_fit)
    residuals = H_vals - H_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', color='crimson',
                 ms=4, capsize=2, label='Cosmic chronometer data')
    ax1.plot(z_smooth, H_smooth, color='navy', lw=2,
              label=rf'model fit ($\delta=4$, $\alpha={alpha_fit:.3f}$)')
    H_lcdm_smooth = H_lcdm(z_smooth, H0_fit, Om_fit)
    ax1.plot(z_smooth, H_lcdm_smooth, color='green', lw=1.5, ls='--',
              label=r'$\Lambda$CDM ($\delta=0$, same $H_0,\Omega_{m,0}$)')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble diagram: best fit (delta=4 fixed, alpha free)')
    ax1.legend()

    ax2.errorbar(z_vals, residuals, yerr=sigma_vals, fmt='o', color='crimson', ms=4, capsize=2)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram_delta4_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_H0_tension_comparison(best_x, perr, outdir='.'):
    """Compare this fit's H0 against literature reference values."""
    H0_fit, _, _ = best_x
    H0_err = perr[0] if perr is not None else 0.0

    all_vals = {"This work ($\\delta=4, \\alpha$ fit)": (H0_fit, H0_err, "crimson")}
    for name, (val, err) in LITERATURE_H0.items():
        all_vals[name] = (val, err, "steelblue" if "Planck" in name else "darkorange")

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(all_vals.items()):
        ax.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)
        ax.axvspan(val - err, val + err, color=color, alpha=0.1)
    ax.set_yticks(range(len(all_vals)))
    ax.set_yticklabels(all_vals.keys())
    ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax.set_title(r'$H_0$: this fit ($\delta=4, \alpha$ free) vs. literature')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'H0_tension_comparison_delta4_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# 8. MODEL COMPARISON TABLE
# =============================================================================

def create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir='.'):
    """Create a comprehensive model comparison table with all statistics."""
    n = len(z_vals)
    dof_model = n - 3
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

    aic_model, bic_model, chi2_dof_model = calculate_stats(chi2_best, 3, dof_model)
    aic_lcdm, bic_lcdm, chi2_dof_lcdm = calculate_stats(chi2_lcdm_best, 2, dof_lcdm)

    delta_aic = aic_model - aic_lcdm
    delta_bic = bic_model - bic_lcdm

    table_data = [
        ['Parameter', 'delta=4 + alpha', 'LambdaCDM'],
        ['H0', f'{best_x[0]:.2f}', f'{H0_l:.2f}'],
        ['Ωm,0', f'{best_x[1]:.3f}', f'{Om_l:.3f}'],
        ['alpha', f'{best_x[2]:.3f}', '0.000 (fixed)'],
        ['δ', f'{DELTA_FIXED:.1f} (fixed)', '0 (fixed)'],
        ['χ²', f'{chi2_best:.2f}', f'{chi2_lcdm_best:.2f}'],
        ['k', '3', '2'],
        ['dof', f'{dof_model}', f'{dof_lcdm}'],
        ['χ²/dof', f'{chi2_dof_model:.2f}', f'{chi2_dof_lcdm:.2f}'],
        ['AIC', f'{aic_model:.2f}', f'{aic_lcdm:.2f}'],
        ['ΔAIC', f'{delta_aic:+.2f}', '0 (reference)'],
        ['BIC', f'{bic_model:.2f}', f'{bic_lcdm:.2f}'],
        ['ΔBIC', f'{delta_bic:+.2f}', '0 (reference)'],
    ]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE (delta=4 fixed, alpha free)")
    print("=" * 80)

    col_widths = [max(len(row[i]) for row in table_data) + 2 for i in range(3)]

    print("│" + "│".join(f"{col:^{col_widths[i]}}" for i, col in enumerate(['Parameter', 'delta=4 + alpha', 'LambdaCDM'])) + "│")
    print("├" + "─" * col_widths[0] + "┼" + "─" * col_widths[1] + "┼" + "─" * col_widths[2] + "┤")

    for row in table_data[1:]:
        print("│" + "│".join(f"{row[i]:^{col_widths[i]}}" for i in range(3)) + "│")

    print("=" * 80)
    print("NOTE: delta=4 + alpha has k=3 free parameters, while LambdaCDM has k=2.")

    filename = os.path.join(outdir, 'model_comparison_table_delta4_alpha.txt')
    with open(filename, 'w') as f:
        f.write("MODEL COMPARISON TABLE (delta=4 fixed, alpha free)\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Parameter':<15} {'delta=4 + alpha':<20} {'LambdaCDM':<20}\n")
        f.write("-" * 80 + "\n")
        for row in table_data[1:]:
            f.write(f"{row[0]:<15} {row[1]:<20} {row[2]:<20}\n")
        f.write("=" * 80 + "\n")
        f.write("\nNOTE: delta=4 + alpha has k=3 free parameters, while LambdaCDM has k=2.\n")

        f.write("\nINTERPRETATION:\n")
        f.write("-" * 40 + "\n")
        if delta_aic < -2:
            f.write("✓ delta=4 + alpha model is strongly preferred by AIC\n")
        elif delta_aic < 0:
            f.write("✓ delta=4 + alpha model is slightly preferred by AIC\n")
        elif delta_aic < 2:
            f.write("○ Models are essentially equivalent by AIC\n")
        else:
            f.write("✗ LambdaCDM is preferred by AIC\n")

        if delta_bic < -2:
            f.write("✓ delta=4 + alpha model is strongly preferred by BIC\n")
        elif delta_bic < 0:
            f.write("✓ delta=4 + alpha model is slightly preferred by BIC\n")
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
    H0_fit, Om_fit, alpha_fit = best_x
    H_best = model_H(z_vals, H0_fit, Om_fit, alpha_fit)
    residuals = H_vals - H_best

    z_smooth = np.linspace(0, z_vals.max() * 1.1, 200)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit, alpha_fit)

    header = "# z, H_obs, sigma_H, H_model, residual\n"
    data_filename = os.path.join(outdir, 'delta4_alpha_lcdm_fit_results.txt')
    with open(data_filename, 'w') as f:
        f.write(header)
        for zi, Hi, si, Hm, ri in zip(z_vals, H_vals, sigma_vals, H_best, residuals):
            f.write(f"{zi:.6f} {Hi:.6f} {si:.6f} {Hm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_filename}")

    curve_filename = os.path.join(outdir, 'delta4_alpha_lcdm_smooth_curve.txt')
    with open(curve_filename, 'w') as f:
        f.write("# z, H_model(z)\n")
        for zi, Hi in zip(z_smooth, H_smooth):
            f.write(f"{zi:.6f} {Hi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_filename}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir="."):
    """Save a comprehensive summary of parameters and statistical markers to disk."""
    filename = os.path.join(outdir, "fit_summary_delta4_alpha.txt")
    if perr is None:
        perr = np.full(3, np.nan)

    with open(filename, "w") as f:
        f.write("===== BEST FIT (delta=4 fixed, alpha free) =====\n\n")
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
            f.write(f"{n:8s} = {med:.6f} (+{hi-med:.6f}/-{med-lo:.6f})\n")
    print(f"  Exported fit summary to: {filename}")


def consistency_check_alpha_small(best_x, z_vals, H_vals, sigma_vals):
    """Behaviour check of chi2 as alpha shrinks."""
    H0_fit, Om_fit, _ = best_x
    print("\nBehaviour of chi^2 as alpha shrinks (H0, Om fixed at best fit):")
    for a in [1.0, 0.5, 0.2, 0.1, 0.05]:
        c = chi2([H0_fit, Om_fit, a], z_vals, H_vals, sigma_vals)
        print(f"  alpha={a:<5} chi^2={c:.3f}" if c < 1e11 else f"  alpha={a:<5} chi^2=inf (no real/stable solution)")

    # cross-check against the true alpha=0 algebraic limit directly
    H_pure = H_delta4_only(z_vals, H0_fit, Om_fit)
    if np.any(np.isnan(H_pure)):
        print("  Pure delta=4 (alpha->0) limit: NOT solvable for this (H0,Om) at "
              "this dataset's redshifts.")
    else:
        chi2_pure = float(np.sum(((H_vals - H_pure) / sigma_vals) ** 2))
        print(f"  Pure delta=4 (alpha->0) limit: chi^2={chi2_pure:.3f}")


def validate_config():
    """Validate configuration parameters."""
    assert CONTOUR_GRID >= 20, "CONTOUR_GRID should be at least 20"
    assert NWALKERS >= 16, "NWALKERS should be at least 16"
    assert NSTEPS >= 1000, "NSTEPS should be at least 1000"
    assert BOUNDS[0][0] < BOUNDS[0][1], "Invalid H0 bounds"
    assert BOUNDS[1][0] < BOUNDS[1][1], "Invalid Om bounds"
    assert BOUNDS[2][0] < BOUNDS[2][1], "Invalid alpha bounds"

    for i in range(len(CONF_LEVELS_2D) - 1):
        assert CONF_LEVELS_2D[i] < CONF_LEVELS_2D[i + 1], "Confidence levels not increasing"


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    validate_config()
    setup_matplotlib()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results_delta4_alpha")
    os.makedirs(outdir, exist_ok=True)
    print(f"Results will be saved to: {outdir}\n")

    z_vals, H_vals, sigma_vals = load_all_data_memory_efficient()
    print(f"Loaded {len(z_vals)} data points.")
    print(f"delta is FIXED at {DELTA_FIXED} for this run; alpha is free.\n")

    if len(z_vals) <= 3:
        raise ValueError(
            f"Need more than 3 data points to fit 3 free parameters "
            f"(H0, Om, alpha); got {len(z_vals)}."
        )

    print("--- Best fit (global optimizer + multi-start cross-check) ---")
    best_x, chi2_best, converged = best_fit(z_vals, H_vals, sigma_vals)
    if not np.isfinite(chi2_best):
        print("\nCRITICAL: no finite chi^2 found anywhere in the search box.")
        return

    H0_fit, Om_fit, alpha_fit = best_x
    dof = len(z_vals) - 3
    print(f"converged: {converged}")
    print(f"H0    = {H0_fit:.4f}")
    print(f"Om    = {Om_fit:.4f}")
    print(f"alpha = {alpha_fit:.4f}")
    print(f"chi^2 = {chi2_best:.4f}  (chi^2/dof = {chi2_best/dof:.4f}, dof={dof})")

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

    plot_walkers(sampler, outdir=outdir)

    print("\n--- Corner plot ---")
    fig_corner = corner.corner(
        flat_samples, labels=[PARAM_LABELS[n] for n in PARAM_NAMES],
        truths=list(best_x), show_titles=True,
    )
    fig_corner.savefig(os.path.join(outdir, 'corner_delta4_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    print("\n--- Profile likelihood & confidence contours for alpha ---")
    plot_chi2_profile_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)
    plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                     vary=('alpha', 'Om'), outdir=outdir)
    plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                     vary=('alpha', 'H0'), outdir=outdir)

    print("\n--- Hubble diagram ---")
    plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir=outdir)

    print("\n--- H0 tension comparison vs literature ---")
    plot_H0_tension_comparison(best_x, perr, outdir=outdir)

    print("\n--- Export best-fit data ---")
    export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, outdir=outdir)

    consistency_check_alpha_small(best_x, z_vals, H_vals, sigma_vals)

    print("\n--- Model Comparison Table ---")
    create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)

    # Save contour data for later comparison
    print("\n--- Saving contour data for comparison ---")
    # Create H0-Om grid at the best-fit alpha
    H0_lo = max(BOUNDS[0][0], H0_fit * 0.85)
    H0_hi = min(BOUNDS[0][1], H0_fit * 1.15)
    Om_lo = max(BOUNDS[1][0], Om_fit * 0.7)
    Om_hi = min(BOUNDS[1][1], Om_fit * 1.3)

    if H0_hi - H0_lo < 5:
        H0_lo = max(BOUNDS[0][0], H0_fit - 5)
        H0_hi = min(BOUNDS[0][1], H0_fit + 5)
    if Om_hi - Om_lo < 0.05:
        Om_lo = max(BOUNDS[1][0], Om_fit - 0.05)
        Om_hi = min(BOUNDS[1][1], Om_fit + 0.5)

    n_grid = 60
    H0_grid = np.linspace(H0_lo, H0_hi, n_grid)
    Om_grid = np.linspace(Om_lo, Om_hi, n_grid)
    H0_mesh, Om_mesh = np.meshgrid(H0_grid, Om_grid)

    # Compute chi2 for each (H0, Om) pair with alpha fixed at best fit
    chi2_vals = np.full((n_grid, n_grid), 1e12)
    print(f"  Computing {n_grid}x{n_grid} H0-Om grid for contour saving...")
    for i in range(n_grid):
        for j in range(n_grid):
            params = np.array([H0_mesh[i, j], Om_mesh[i, j], alpha_fit])
            chi2_vals[i, j] = chi2(params, z_vals, H_vals, sigma_vals)

    delta_chi2 = chi2_vals - chi2_best

    # Save contour data
    contour_data = {'X': H0_mesh, 'Y': Om_mesh, 'delta_chi2': delta_chi2}
    np.save(os.path.join(outdir, 'contour_H0_Om_delta4_alpha.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(outdir, 'contour_H0_Om_delta4_alpha.npy')}")

    if perr is not None and 'pcov' in locals():
        corr = pcov / np.outer(perr, perr)
        print("\nCorrelation matrix")
        print("--------------------------------")
        for row in corr:
            print(" ".join(f"{x:8.3f}" for x in row))
        np.savetxt(
            os.path.join(outdir, "correlation_matrix_delta4_alpha.txt"),
            corr,
            fmt="%.6f"
        )
    else:
        perr = np.full(3, np.nan) 

    write_fit_summary(
        best_x,
        perr,
        chi2_best,
        dof,
        flat_samples,
        sampler,
        outdir
    )

    print(f"\nDone. All figures and results written to: {outdir}")

if __name__ == "__main__":
    main()