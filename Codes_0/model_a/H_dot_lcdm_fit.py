"""
H_dot_lcdm_fit.py
==================
Modified Friedmann equation fit:

    H(z)^2 = H0^2 * Om * (1+z)^3  -  alpha * (1+z) * H(z) * dH/dz

Free parameters: (H0, Om, alpha). alpha=0 collapses the correction term, but
note this is NOT the standard LambdaCDM limit -- there is no explicit
(1-Om) dark-energy term anywhere in this equation. The entire late-time
behaviour has to come from the alpha*(1+z)*H*dH/dz piece, which is exactly
what makes alpha the interesting parameter to pin down here.

OPTIMIZED VERSION: Includes caching, vectorization, parallel processing,
adaptive contours, and performance improvements.
"""

# NOTE TO SELF: this supersedes test0/1/2/3.py -- those had either the b*H^delta
# term mixed in (not what the thesis equation actually says) or a less stable
# direct-H ODE. Keep working from *this* file; copy it first if experimenting.

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
BOUNDS = [(50.0, 100.0), (0.01, 3.0), (0.01, 6.0)]   # H0, Om, alpha
PARAM_NAMES = ['H0', 'Om', 'alpha']
PARAM_LABELS = {'H0': r'$H_0$', 'Om': r'$\Omega_{m,0}$', 'alpha': r'$\alpha$'}

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
USE_PARALLEL = True
ADAPTIVE_CONTOURS = True
BATCH_SIZE = 100
N_MULTISTART = 8
USE_ANALYTICAL = True  # Set to False to fall back to the original numerical ODE integration


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
                    files.append(f"... and more")
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
    """Memory-efficient data loading using numpy's loadtxt."""
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
    u_safe = min(max(u[0], 1e-8), 1e10)
    x = 1.0 + z
    dudz = (2.0 / (alpha * x)) * (H0**2 * Om * x**3 - u_safe)
    return [np.clip(dudz, -1e12, 1e12)]


@lru_cache(maxsize=256)
def _model_H_cached(z_tuple, H0, Om, alpha):
    """Cached version of model_H for repeated calls with same parameters."""
    z_eval = np.array(z_tuple)
    if alpha <= 0 or H0 <= 0 or Om <= 0:
        return tuple([np.nan] * len(z_eval))
    
    z_max = max(z_eval.max(), 1e-6)
    t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))
    
    try:
        sol = solve_ivp(
            _rhs_u, (0.0, z_max), [H0**2],
            args=(H0, Om, alpha),
            t_eval=t_eval,
            method='LSODA',
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


def model_H_analytical(z_eval, H0, Om, alpha):
    """Blasingly fast, exact closed-form analytical solution of the ODE."""
    if alpha <= 0 or H0 <= 0 or Om <= 0:
        return np.full_like(z_eval, np.nan)
    
    x = 1.0 + z_eval
    
    # Clip extreme exponent boundaries to avoid potential float overflow/underflow
    exponent = np.clip(-2.0 / alpha, -100.0, 100.0)
    pow_term = x ** exponent
    
    # Exact algebraic solution
    u_of_z = (H0**2) * pow_term + (2.0 * H0**2 * Om / (3.0 * alpha + 2.0)) * (x**3 - pow_term)
    u_of_z = np.clip(u_of_z, 1e-10, None)
    
    return np.sqrt(u_of_z)


def model_H(z_eval, H0, Om, alpha):
    """Solve the ODE for u=H^2 and return H(z) at the requested redshifts."""
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    
    if USE_ANALYTICAL:
        return model_H_analytical(z_eval, H0, Om, alpha)
    
    if USE_CACHING:
        z_tuple = tuple(z_eval)
        result = _model_H_cached(z_tuple, float(H0), float(Om), float(alpha))
        return np.array(result)
    else:
        if alpha == 0 or H0 <= 0 or Om <= 0:
            return np.full_like(z_eval, np.nan)
        
        z_max = max(z_eval.max(), 1e-6)
        t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))
        
        try:
            sol = solve_ivp(
                _rhs_u, (0.0, z_max), [H0**2],
                args=(H0, Om, alpha),
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
    return H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))


# =============================================================================
# 3. CHI-SQUARED (with vectorized batch evaluation)
# =============================================================================

def _within_bounds(params):
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, BOUNDS))


def chi2(params, z_vals, H_vals, sigma_vals):
    H0, Om, alpha = params
    if not _within_bounds(params):
        return 1e12
    H_model = model_H(z_vals, H0, Om, alpha)
    if np.any(~np.isfinite(H_model)):
        return 1e12
    return float(np.sum(((H_vals - H_model) / sigma_vals) ** 2))


def chi2_grid_vectorized(params_grid, z_vals, H_vals, sigma_vals):
    """Fully vectorized chi-squared using NumPy broadcasting for ultra-fast contouring."""
    n_grid = len(params_grid)
    chi2_vals = np.full(n_grid, 1e12)
    
    valid = np.array([_within_bounds(p) for p in params_grid])
    if not np.any(valid):
        return chi2_vals
        
    valid_params = params_grid[valid]
    
    H0 = valid_params[:, 0][:, np.newaxis]      # Shape (V, 1)
    Om = valid_params[:, 1][:, np.newaxis]      # Shape (V, 1)
    alpha = valid_params[:, 2][:, np.newaxis]   # Shape (V, 1)
    
    z_eval = z_vals[np.newaxis, :]              # Shape (1, M)
    x = 1.0 + z_eval                            # Shape (1, M)
    
    exponent = np.clip(-2.0 / alpha, -100.0, 100.0)
    pow_term = x ** exponent                    # Shape (V, M)
    
    u_of_z = (H0**2) * pow_term + (2.0 * H0**2 * Om / (3.0 * alpha + 2.0)) * (x**3 - pow_term)
    u_of_z = np.clip(u_of_z, 1e-10, None)
    H_model = np.sqrt(u_of_z)                   # Shape (V, M)
    
    residuals = (H_vals[np.newaxis, :] - H_model) / sigma_vals[np.newaxis, :]
    chi2_vals_valid = np.sum(residuals**2, axis=1)
    
    chi2_vals[valid] = chi2_vals_valid
    return chi2_vals


def chi2_grid(params_grid, z_vals, H_vals, sigma_vals):
    """Vectorized chi-squared for grid calculations (contours)."""
    if USE_ANALYTICAL:
        return chi2_grid_vectorized(params_grid, z_vals, H_vals, sigma_vals)
        
    chi2_vals = np.full(len(params_grid), 1e12)
    
    valid = np.array([_within_bounds(p) for p in params_grid])
    if not np.any(valid):
        return chi2_vals
    
    valid_params = params_grid[valid]
    chi2_vals_valid = np.zeros(len(valid_params))
    
    batch_size = BATCH_SIZE
    for i in range(0, len(valid_params), batch_size):
        batch = valid_params[i:i+batch_size]
        for j, params in enumerate(batch):
            H_model = model_H(z_vals, *params)
            if np.all(np.isfinite(H_model)):
                chi2_vals_valid[i+j] = np.sum(((H_vals - H_model) / sigma_vals) ** 2)
            else:
                chi2_vals_valid[i+j] = 1e12
    
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
    """Global fit (differential_evolution) followed by a Nelder-Mead polish."""
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
        print(f"  Multi-start scan: {len(spread)}/{len(starts)} runs converged "
              f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
        if spread.size and spread.max() - spread.min() > 0.5:
            print("  -> spread across starts suggests a degenerate/multi-modal "
                  "chi^2 surface")

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
    
    spread = np.array([2.0, 0.05, 0.1])
    pos = np.zeros((nwalkers, ndim))
    
    for i in range(nwalkers):
        pos[i] = best_x + spread * np.random.randn(ndim)
        for j, (lo, hi) in enumerate(BOUNDS):
            pos[i, j] = np.clip(pos[i, j], lo + 1e-6, hi - 1e-6)

    pool = None
    if USE_PARALLEL:
        if USE_ANALYTICAL:
            # Analytical model evaluation is so fast (microseconds) that process creation and IPC 
            # overhead of multiprocessing actually slows down the execution. Sequential is run instead.
            print("  Using ultra-fast analytical model. Sequential MCMC is faster than parallel overhead.")
        else:
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
    chain = sampler.get_chain()

    fig, axes = plt.subplots(3,1, figsize=(10,7), sharex=True)

    for i in range(3):
        for walker in range(chain.shape[1]):
            axes[i].plot(
                chain[:,walker,i],
                alpha=0.3,
                lw=0.5
            )
        axes[i].set_ylabel(PARAM_LABELS[PARAM_NAMES[i]])

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"walker_chains.png"), dpi=300)
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
    print("  Computing profile likelihood...")
    for i, a in enumerate(tqdm(alphas, desc="  Alpha profile")):
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
    ax.set_title(r'Profile likelihood: $\Delta\chi^2$ vs $\alpha$ ($H_0,\,\Omega_{m,0}$ refit at each point)')
    ax.set_ylim(0, 10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    
    fig.savefig(os.path.join(outdir, 'chi2_profile_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    if alpha_lo68 is not None and alpha_hi68 is not None:
        print(f"  alpha 1sigma profile interval: [{alpha_lo68:.4f}, {alpha_hi68:.4f}]")

    return alphas, chi2_vals


def plot_confidence_ellipses_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                    n_grid=CONTOUR_GRID, outdir='.'):
    """Create confidence ellipses plot for H0, Om, and chi2 at best alpha."""
    H0_fit, Om_fit, alpha_fit = best_x
    
    # Create a 2D grid in H0-Om space at fixed alpha
    H0_lo, H0_hi = max(BOUNDS[0][0], H0_fit * 0.85), min(BOUNDS[0][1], H0_fit * 1.15)
    Om_lo, Om_hi = max(BOUNDS[1][0], Om_fit * 0.5), min(BOUNDS[1][1], Om_fit * 2.0)
    
    H0_grid = np.linspace(H0_lo, H0_hi, n_grid)
    Om_grid = np.linspace(Om_lo, Om_hi, n_grid)
    H0_mesh, Om_mesh = np.meshgrid(H0_grid, Om_grid)
    
    # Compute chi2 grid
    params_flat = np.zeros((n_grid * n_grid, 3))
    params_flat[:, 0] = H0_mesh.ravel()
    params_flat[:, 1] = Om_mesh.ravel()
    params_flat[:, 2] = alpha_fit
    
    chi2_flat = chi2_grid(params_flat, z_vals, H_vals, sigma_vals)
    CHI2_mesh = chi2_flat.reshape(n_grid, n_grid)
    delta_chi2 = CHI2_mesh - chi2_best
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.3], height_ratios=[1, 0.3])
    
    # Main contour plot
    ax_main = plt.subplot(gs[0, 0])
    cs = ax_main.contour(H0_mesh, Om_mesh, delta_chi2, levels=CONF_LEVELS_2D,
                          colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax_main.clabel(cs, fmt={CONF_LEVELS_2D[0]: r'1$\sigma$',
                             CONF_LEVELS_2D[1]: r'2$\sigma$',
                             CONF_LEVELS_2D[2]: r'3$\sigma$'})
    ax_main.contourf(H0_mesh, Om_mesh, delta_chi2,
                      levels=[0, *CONF_LEVELS_2D, max(delta_chi2.max(), CONF_LEVELS_2D[-1] + 1)],
                      colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    ax_main.plot(H0_fit, Om_fit, 'k*', ms=14, label='best fit')
    ax_main.set_xlabel(r'$H_0$ [km/s/Mpc]')
    ax_main.set_ylabel(r'$\Omega_{m,0}$')
    ax_main.set_title(rf'Confidence contours: $H_0$ vs $\Omega_{{m,0}}$ at $\alpha={alpha_fit:.3f}$')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Marginal distributions
    ax_top = plt.subplot(gs[0, 1])
    ax_top.hist(H0_mesh.ravel()[delta_chi2.ravel() <= CONF_LEVELS_2D[0]], 
                bins=30, orientation='horizontal', color='#1f77b4', alpha=0.7,
                density=True)
    ax_top.axhline(H0_fit, color='k', ls='--', lw=1)
    ax_top.set_ylabel(r'$H_0$ [km/s/Mpc]')
    ax_top.set_title('1$\sigma$ marginal')
    ax_top.tick_params(axis='x', labelrotation=90)
    
    ax_right = plt.subplot(gs[1, 0])
    ax_right.hist(Om_mesh.ravel()[delta_chi2.ravel() <= CONF_LEVELS_2D[0]], 
                  bins=30, color='#1f77b4', alpha=0.7, density=True)
    ax_right.axvline(Om_fit, color='k', ls='--', lw=1)
    ax_right.set_xlabel(r'$\Omega_{m,0}$')
    ax_right.set_title('1$\sigma$ marginal')
    
    # Empty corner
    ax_corner = plt.subplot(gs[1, 1])
    ax_corner.axis('off')
    
    # Add chi2 information
    textstr = (f'$\chi^2 = {chi2_best:.2f}$\n'
               f'$\chi^2/\\nu = {chi2_best/(len(z_vals)-3):.2f}$\n'
               f'$\Delta\chi^2$ contours:\n'
               f'  1$\sigma$: {CONF_LEVELS_2D[0]:.2f}\n'
               f'  2$\sigma$: {CONF_LEVELS_2D[1]:.2f}\n'
               f'  3$\sigma$: {CONF_LEVELS_2D[2]:.2f}')
    ax_corner.text(0.5, 0.5, textstr, transform=ax_corner.transAxes,
                   fontsize=12, verticalalignment='center',
                   horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'confidence_ellipses_alpha_best.png'), 
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return H0_mesh, Om_mesh, delta_chi2


def plot_contour_2d_optimized(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                               vary=('alpha', 'Om'), n_grid=CONTOUR_GRID, outdir='.'):
    """Optimized Delta-chi^2 contour using vectorized grid evaluation."""
    idx = {'H0': 0, 'Om': 1, 'alpha': 2}
    ix, iy = idx[vary[0]], idx[vary[1]]
    iz = ({0, 1, 2} - {ix, iy}).pop()

    center = best_x[ix], best_x[iy]

    x_lo, x_hi = max(BOUNDS[ix][0], center[0] * 0.3), min(BOUNDS[ix][1], center[0] * 2.2)
    y_lo, y_hi = max(BOUNDS[iy][0], center[1] * 0.3), min(BOUNDS[iy][1], center[1] * 2.2)
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    params_flat = np.zeros((n_grid * n_grid, 3))
    params_flat[:, ix] = X.ravel()
    params_flat[:, iy] = Y.ravel()
    params_flat[:, iz] = best_x[iz]
    
    print(f"  Computing {n_grid}x{n_grid} grid for {vary[0]}-{vary[1]} contour...")
    chi2_flat = chi2_grid(params_flat, z_vals, H_vals, sigma_vals)
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
                 rf'({PARAM_LABELS[PARAM_NAMES[iz]]} fixed)')
    ax.legend()
    fig.tight_layout()
    
    fname = f'contour_{vary[0]}_{vary[1]}.png'
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return X, Y, delta_chi2


def adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                                vary=('alpha', 'Om'), 
                                n_grid_min=30, n_grid_max=80, outdir='.'):
    """Adaptive contour plotting with variable resolution."""
    if not ADAPTIVE_CONTOURS:
        return plot_contour_2d_optimized(
            best_x, chi2_best, z_vals, H_vals, sigma_vals,
            vary=vary, n_grid=CONTOUR_GRID, outdir=outdir
        )
    
    n_grid = n_grid_min
    X, Y, delta = plot_contour_2d_optimized(
        best_x, chi2_best, z_vals, H_vals, sigma_vals,
        vary=vary, n_grid=n_grid, outdir=outdir
    )
    
    grad_x = np.gradient(delta, axis=0)
    grad_y = np.gradient(delta, axis=1)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    if np.std(grad_mag) > 0.5 * np.mean(grad_mag) and n_grid < n_grid_max:
        n_grid = min(n_grid * 2, n_grid_max)
        print(f"  Refining contour grid to {n_grid}x{n_grid}...")
        X, Y, delta = plot_contour_2d_optimized(
            best_x, chi2_best, z_vals, H_vals, sigma_vals,
            vary=vary, n_grid=n_grid, outdir=outdir
        )
    
    return X, Y, delta


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
              label=rf'model fit ($\alpha={alpha_fit:.3f}$)')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble diagram: best fit')
    ax1.legend()

    ax2.errorbar(z_vals, residuals, yerr=sigma_vals, fmt='o', color='crimson', ms=4, capsize=2)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram.png'), dpi=300, bbox_inches='tight')
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
        chi2_lcdm, bounds=[(50.0, 90.0), (0.05, 0.95)],
        args=(z_vals, H_vals, sigma_vals), seed=42, tol=1e-8,
    )
    H0_l, Om_l = de_lcdm.x
    chi2_lcdm_best = de_lcdm.fun
    
    # Calculate statistics
    def calculate_stats(chi2_val, k, dof):
        aic = chi2_val + 2 * k
        bic = chi2_val + k * np.log(n)
        chi2_dof = chi2_val / dof if dof > 0 else np.inf
        return aic, bic, chi2_dof
    
    aic_model, bic_model, chi2_dof_model = calculate_stats(chi2_best, 3, dof_model)
    aic_lcdm, bic_lcdm, chi2_dof_lcdm = calculate_stats(chi2_lcdm_best, 2, dof_lcdm)
    
    # Calculate delta values
    delta_aic = aic_model - aic_lcdm
    delta_bic = bic_model - bic_lcdm
    
    # Create table data
    table_data = [
        ['Model', 'H_dot-alpha', 'LambdaCDM'],
        ['H0', f'{best_x[0]:.2f}', f'{H0_l:.2f}'],
        ['Ωm,0', f'{best_x[1]:.3f}', f'{Om_l:.3f}'],
        ['α', f'{best_x[2]:.3f}', '-'],
        ['χ²', f'{chi2_best:.2f}', f'{chi2_lcdm_best:.2f}'],
        ['k', '3', '2'],
        ['dof', f'{dof_model}', f'{dof_lcdm}'],
        ['χ²/dof', f'{chi2_dof_model:.2f}', f'{chi2_dof_lcdm:.2f}'],
        ['AIC', f'{aic_model:.2f}', f'{aic_lcdm:.2f}'],
        ['ΔAIC', f'{delta_aic:+.2f}', '0 (reference)'],
        ['BIC', f'{bic_model:.2f}', f'{bic_lcdm:.2f}'],
        ['ΔBIC', f'{delta_bic:+.2f}', '0 (reference)'],
    ]
    
    # Print table
    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    
    # Find max width for each column
    col_widths = [max(len(row[i]) for row in table_data) + 2 for i in range(3)]
    
    # Print header
    print("│" + "│".join(f"{col:^{col_widths[i]}}" for i, col in enumerate(['Parameter', 'H_dot-alpha', 'LambdaCDM'])) + "│")
    print("├" + "─" * col_widths[0] + "┼" + "─" * col_widths[1] + "┼" + "─" * col_widths[2] + "┤")
    
    # Print rows
    for row in table_data[1:]:  # Skip header row
        print("│" + "│".join(f"{row[i]:^{col_widths[i]}}" for i in range(3)) + "│")
    
    print("="*80)
    
    # Also save to file
    filename = os.path.join(outdir, 'model_comparison_table.txt')
    with open(filename, 'w') as f:
        f.write("MODEL COMPARISON TABLE\n")
        f.write("="*80 + "\n")
        f.write(f"{'Parameter':<15} {'H_dot-alpha':<20} {'LambdaCDM':<20}\n")
        f.write("-"*80 + "\n")
        for row in table_data[1:]:
            f.write(f"{row[0]:<15} {row[1]:<20} {row[2]:<20}\n")
        f.write("="*80 + "\n")
        
        # Add interpretation
        f.write("\nINTERPRETATION:\n")
        f.write("-"*40 + "\n")
        if delta_aic < -2:
            f.write("✓ H_dot-alpha model is strongly preferred by AIC\n")
        elif delta_aic < 0:
            f.write("✓ H_dot-alpha model is slightly preferred by AIC\n")
        elif delta_aic < 2:
            f.write("○ Models are essentially equivalent by AIC\n")
        else:
            f.write("✗ LambdaCDM is preferred by AIC\n")
            
        if delta_bic < -2:
            f.write("✓ H_dot-alpha model is strongly preferred by BIC\n")
        elif delta_bic < 0:
            f.write("✓ H_dot-alpha model is slightly preferred by BIC\n")
        elif delta_bic < 2:
            f.write("○ Models are essentially equivalent by BIC\n")
        else:
            f.write("✗ LambdaCDM is preferred by BIC\n")
    
    print(f"\nModel comparison table saved to: {filename}")
    
    return table_data


def model_comparison(best_x, chi2_best, z_vals, H_vals, sigma_vals):
    """Legacy function for backward compatibility."""
    return create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals)


# =============================================================================
# 9. ADDITIONAL PLOTS AND EXPORTS
# =============================================================================

def consistency_check_alpha_small(best_x, z_vals, H_vals, sigma_vals):
    """As alpha shrinks, the correction term sits in a 1/alpha coefficient."""
    H0_fit, Om_fit, _ = best_x
    print("\nBehaviour of chi^2 as alpha shrinks (H0, Om fixed at best fit):")
    for a in [1.0, 0.5, 0.2, 0.1, 0.05]:
        c = chi2([H0_fit, Om_fit, a], z_vals, H_vals, sigma_vals)
        print(f"  alpha={a:<5} chi^2={c:.3f}")


def export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, outdir='.'):
    """Export the best-fit model predictions and residuals to text files."""
    H0_fit, Om_fit, alpha_fit = best_x
    H_best = model_H(z_vals, H0_fit, Om_fit, alpha_fit)
    residuals = H_vals - H_best

    z_smooth = np.linspace(0, z_vals.max() * 1.1, 200)
    H_smooth = model_H(z_smooth, H0_fit, Om_fit, alpha_fit)

    header = "# z, H_obs, sigma_H, H_model, residual\n"
    data_filename = os.path.join(outdir, 'H_dot_fit_results.txt')
    with open(data_filename, 'w') as f:
        f.write(header)
        for zi, Hi, si, Hm, ri in zip(z_vals, H_vals, sigma_vals, H_best, residuals):
            f.write(f"{zi:.6f} {Hi:.6f} {si:.6f} {Hm:.6f} {ri:.6f}\n")
    print(f"  Exported best-fit results to: {data_filename}")

    curve_filename = os.path.join(outdir, 'H_dot_smooth_curve.txt')
    with open(curve_filename, 'w') as f:
        f.write("# z, H_model(z)\n")
        for zi, Hi in zip(z_smooth, H_smooth):
            f.write(f"{zi:.6f} {Hi:.6f}\n")
    print(f"  Exported smooth model curve to: {curve_filename}")


def write_fit_summary(best_x, perr, chi2_best, dof, flat_samples, sampler, outdir="."):
    filename = os.path.join(outdir,"fit_summary.txt")

    with open(filename,"w") as f:
        f.write("===== BEST FIT =====\n\n")
        for n,v,e in zip(PARAM_NAMES,best_x,perr):
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
            for n,t in zip(PARAM_NAMES,tau):
                f.write(f"{n:8s} {t:.2f}\n")
        except:
            pass
        f.write("\n")
        p = np.percentile(flat_samples,[16,50,84],axis=0)
        f.write("===== MCMC =====\n\n")
        for i,n in enumerate(PARAM_NAMES):
            lo,med,hi = p[:,i]
            f.write(
                f"{n:8s} = {med:.6f} "
                f"(+{hi-med:.6f}/-{med-lo:.6f})\n"
            )


def validate_config():
    """Validate configuration parameters."""
    assert CONTOUR_GRID >= 20, "CONTOUR_GRID should be at least 20"
    assert PROFILE_POINTS >= 20, "PROFILE_POINTS should be at least 20"
    assert NWALKERS >= 16, "NWALKERS should be at least 16"
    assert NSTEPS >= 1000, "NSTEPS should be at least 1000"
    
    assert BOUNDS[0][0] < BOUNDS[0][1], "Invalid H0 bounds"
    assert BOUNDS[1][0] < BOUNDS[1][1], "Invalid Om bounds"
    assert BOUNDS[2][0] < BOUNDS[2][1], "Invalid alpha bounds"
    
    for i in range(len(CONF_LEVELS_2D)-1):
        assert CONF_LEVELS_2D[i] < CONF_LEVELS_2D[i+1], "Confidence levels not increasing"


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    """Main function with all optimizations enabled."""
    validate_config()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    outdir = os.path.join(script_dir, "results")
    os.makedirs(outdir, exist_ok=True)
    print(f"Results will be saved to: {outdir}\n")
    
    setup_matplotlib()

    z_vals, H_vals, sigma_vals = load_all_data_memory_efficient()
    print(f"\nLoaded {len(z_vals)} data points.")
    print(f"Redshift range: {z_vals.min():.3f} to {z_vals.max():.3f}\n")

    print("\n--- Best fit (global optimizer + multi-start cross-check) ---")
    best_x, chi2_best, converged = best_fit(z_vals, H_vals, sigma_vals)
    H0_fit, Om_fit, alpha_fit = best_x
    dof = len(z_vals) - 3
    print(f"  converged: {converged}")
    print(f"  H0    = {H0_fit:.4f}")
    print(f"  Om    = {Om_fit:.4f}")
    print(f"  alpha = {alpha_fit:.4f}")
    print(f"  chi^2 = {chi2_best:.4f}  (chi^2/dof = {chi2_best/dof:.4f}, dof={dof})")

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

    print("\n--- MCMC posterior (emcee, parallel) ---")
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
    fig_corner.savefig(os.path.join(outdir, 'corner_H_dot_alpha.png'), dpi=300, bbox_inches='tight')
    plt.close(fig_corner)

    print("\n--- Profile likelihood & confidence contours for alpha ---")
    plot_chi2_profile_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)
    
    print("  Generating contours (optimized)...")
    adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                               vary=('alpha', 'Om'), outdir=outdir)
    adaptive_contour_if_needed(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                               vary=('alpha', 'H0'), outdir=outdir)
    print("  Saved: chi2_profile_alpha.png, contour_alpha_Om.png, contour_alpha_H0.png")

    print("\n--- Confidence ellipses at best alpha value ---")
    H0_mesh, Om_mesh, delta_chi2 = plot_confidence_ellipses_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)
    print("  Saved: confidence_ellipses_alpha_best.png")

    # Save contour data for later comparison
    contour_data = {'X': H0_mesh, 'Y': Om_mesh, 'delta_chi2': delta_chi2}
    np.save(os.path.join(outdir, 'contour_H0_Om_hdot_alpha.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(outdir, 'contour_H0_Om_hdot_alpha.npy')}")  

    print("\n--- Hubble diagram ---")
    plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir=outdir)
    print("  Saved: hubble_diagram.png")

    print("\n--- Export best-fit data ---")
    export_best_fit_data(z_vals, H_vals, sigma_vals, best_x, outdir=outdir)

    consistency_check_alpha_small(best_x, z_vals, H_vals, sigma_vals)
    
    print("\n--- Model Comparison Table ---")
    table_data = create_model_comparison_table(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)

    print(f"\nDone. All figures and results saved to: {outdir}")

    if 'pcov' in locals():
        corr = pcov / np.outer(perr, perr)
        print("\nCorrelation matrix")
        print("--------------------------------")
        for row in corr:
            print(" ".join(f"{x:8.3f}" for x in row))
        np.savetxt(
            os.path.join(outdir, "correlation_matrix.txt"),
            corr,
            fmt="%.6f"
        )

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