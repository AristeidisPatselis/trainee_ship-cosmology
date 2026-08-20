"""
horndeski_full_dynamics_fit.py
================================
Full Horndeski Gravity fitting pipeline (arXiv:2110.01338)[cite: 4]
Includes:
  1. Gelman-Rubin convergence diagnostic (R-hat)
  2. Overlayed 2D posterior contours for shared parameters (H0, Om0)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.optimize import minimize
try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:
    from scipy.integrate import cumtrapz as cumulative_trapezoid
import emcee
import corner
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# 0. Data Loading & Settings
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/"

_BASE_DATA_DIRS = [DATA_DIR, os.environ.get("COSMO_DATA_DIR"), HERE, os.path.join(HERE, "data"), os.getcwd()]

def find_data_file(filename):
    for d in _BASE_DATA_DIRS:
        if d and os.path.isdir(d):
            path = os.path.join(d, filename)
            if os.path.isfile(path):
                return path
            for sub in sorted(os.listdir(d)):
                sub_path = os.path.join(d, sub, filename)
                if os.path.isfile(sub_path):
                    return sub_path
    raise FileNotFoundError(f"Could not locate '{filename}'.")

for pfile in ["Pantheon+SH0ES.dat", "pantheon_shoes.dat", "Pantheon+.dat", "Pantheon+SH0ES.txt"]:
    try:
        pantheon_path = find_data_file(pfile)
        break
    except FileNotFoundError:
        pantheon_path = None

df_pantheon = pd.read_csv(pantheon_path, sep=r'\s+', comment="#")
z_col = "zHD" if "zHD" in df_pantheon.columns else ("z" if "z" in df_pantheon.columns else df_pantheon.columns[0])
err_col = "m_b_corr_err_DIAG" if "m_b_corr_err_DIAG" in df_pantheon.columns else ("MU_ERR" if "MU_ERR" in df_pantheon.columns else df_pantheon.columns[2])

df_pantheon = df_pantheon[df_pantheon[z_col] > 0].copy()

if "MU_SH0ES" in df_pantheon.columns:
    mu_data = df_pantheon["MU_SH0ES"].values
elif "MU" in df_pantheon.columns:
    mu_data = df_pantheon["MU"].values
else:
    mu_data = df_pantheon["m_b_corr"].values - (-19.253)

z_data = df_pantheon[z_col].values
sigma_data = df_pantheon[err_col].values
Z_EVAL_GRID = np.concatenate(([0.0], np.geomspace(1e-5, max(z_data) * 1.05, 300)))

# ----------------------------------------------------------------------
# 1. Model Functions & Likelihood
# ----------------------------------------------------------------------

def solve_E_model1(z, Om0, alpha, n_iter=6):
    R = Om0 * (1.0 + z)**3 + (1.0 - Om0 - alpha)
    if np.any(R <= 0): return None
    E = np.sqrt(R)
    for _ in range(n_iter):
        f = alpha * E**3 - E**2 + R
        f_prime = 3.0 * alpha * E**2 - 2.0 * E
        E -= f / f_prime
    if np.any(np.isnan(E)) or np.any(E <= 0): return None
    return E

def solve_E_model2(z, Om0, beta):
    R = Om0 * (1.0 + z)**3 + (1.0 - Om0 - beta)
    if np.any(R <= 0): return None
    disc = 1.0 - 4.0 * beta * R
    if np.any(disc < 0): return None
    Y = (1.0 - np.sqrt(disc)) / (2.0 * beta) if abs(beta) >= 1e-8 else R + beta * R**2
    if np.any(Y <= 0): return None
    return np.sqrt(Y)

def mu_m1(z, theta):
    c_light = 299792.458
    H_vals = theta[0] * solve_E_model1(Z_EVAL_GRID, theta[1], theta[2]) if solve_E_model1(Z_EVAL_GRID, theta[1], theta[2]) is not None else None
    if H_vals is None or np.any(np.isnan(H_vals)) or np.any(H_vals <= 0): return None
    integ = cumulative_trapezoid(1.0 / H_vals, Z_EVAL_GRID, initial=0.0)
    dL = (1.0 + Z_EVAL_GRID) * c_light * integ
    mu_grid = np.zeros_like(Z_EVAL_GRID)
    mu_grid[1:] = 5.0 * np.log10(dL[1:]) + 25.0
    mu_grid[0] = mu_grid[1]
    return np.interp(z, Z_EVAL_GRID, mu_grid)

def mu_m2(z, theta):
    c_light = 299792.458
    H_vals = theta[0] * solve_E_model2(Z_EVAL_GRID, theta[1], theta[2]) if solve_E_model2(Z_EVAL_GRID, theta[1], theta[2]) is not None else None
    if H_vals is None or np.any(np.isnan(H_vals)) or np.any(H_vals <= 0): return None
    integ = cumulative_trapezoid(1.0 / H_vals, Z_EVAL_GRID, initial=0.0)
    dL = (1.0 + Z_EVAL_GRID) * c_light * integ
    mu_grid = np.zeros_like(Z_EVAL_GRID)
    mu_grid[1:] = 5.0 * np.log10(dL[1:]) + 25.0
    mu_grid[0] = mu_grid[1]
    return np.interp(z, Z_EVAL_GRID, mu_grid)

def log_prior(theta):
    H0, Om0, param = theta
    if not (60.0 < H0 < 85.0 and 0.15 < Om0 < 0.45 and -0.3 < param < 0.3):
        return -np.inf
    return -0.5 * ((Om0 - 0.315) / 0.030)**2

def log_prob_m1(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    mu_model = mu_m1(z_data, theta)
    if mu_model is None or np.any(np.isnan(mu_model)): return -np.inf
    return lp - 0.5 * np.sum(((mu_data - mu_model) / sigma_data)**2)

def log_prob_m2(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp): return -np.inf
    mu_model = mu_m2(z_data, theta)
    if mu_model is None or np.any(np.isnan(mu_model)): return -np.inf
    return lp - 0.5 * np.sum(((mu_data - mu_model) / sigma_data)**2)

# ----------------------------------------------------------------------
# 2. Gelman-Rubin Diagnostic Calculation
# ----------------------------------------------------------------------

def compute_gelman_rubin(chain_3d):
    """Calculates Gelman-Rubin R-hat statistic for 3D array (N_steps, N_walkers, N_dim)."""
    N, M, D = chain_3d.shape
    if M < 2:
        return np.full(D, np.nan)
    
    walker_means = np.mean(chain_3d, axis=0)
    grand_mean = np.mean(walker_means, axis=0)
    
    B = (N / (M - 1.0)) * np.sum((walker_means - grand_mean)**2, axis=0)
    W = np.mean(np.var(chain_3d, axis=0, ddof=1), axis=0)
    
    var_plus = ((N - 1.0) / N) * W + (1.0 / N) * B
    return np.sqrt(var_plus / W)

# ----------------------------------------------------------------------
# 3. Execution Pipeline
# ----------------------------------------------------------------------

if __name__ == "__main__":
    NWALKERS, NSTEPS, BURNIN = 32, 6000, 1500
    ncpu = max(1, cpu_count() - 1)
    rng = np.random.default_rng(42)
    
    out_dir = os.path.join(HERE, "outputs_horndeski_pantheon_comparison")
    os.makedirs(out_dir, exist_ok=True)

    # --- MODEL I ---
    res1 = minimize(lambda t: -log_prob_m1(t), [73.0, 0.31, 0.0], method="Nelder-Mead")
    p0_1 = np.array(res1.x) + np.array([0.5, 0.01, 0.005]) * rng.standard_normal((NWALKERS, 3))
    
    with Pool(processes=ncpu) as pool:
        sampler1 = emcee.EnsembleSampler(NWALKERS, 3, log_prob_m1, pool=pool)
        sampler1.run_mcmc(p0_1, NSTEPS, progress=True)

    chain_3d_1 = sampler1.get_chain(discard=BURNIN, flat=False)
    chain1_flat = sampler1.get_chain(discard=BURNIN, flat=True)
    r_hat_1 = compute_gelman_rubin(chain_3d_1)
    
    print("\n--- MODEL I DIAGNOSTICS ---")
    print(f"R-hat (H0, Om0, xi): {r_hat_1[0]:.4f}, {r_hat_1[1]:.4f}, {r_hat_1[2]:.4f}")

    # --- MODEL II ---
    res2 = minimize(lambda t: -log_prob_m2(t), [73.0, 0.31, 0.0], method="Nelder-Mead")
    p0_2 = np.array(res2.x) + np.array([0.5, 0.01, 0.005]) * rng.standard_normal((NWALKERS, 3))
    
    with Pool(processes=ncpu) as pool:
        sampler2 = emcee.EnsembleSampler(NWALKERS, 3, log_prob_m2, pool=pool)
        sampler2.run_mcmc(p0_2, NSTEPS, progress=True)

    chain_3d_2 = sampler2.get_chain(discard=BURNIN, flat=False)
    chain2_flat = sampler2.get_chain(discard=BURNIN, flat=True)
    r_hat_2 = compute_gelman_rubin(chain_3d_2)
    
    print("\n--- MODEL II DIAGNOSTICS ---")
    print(f"R-hat (H0, Om0, lambda): {r_hat_2[0]:.4f}, {r_hat_2[1]:.4f}, {r_hat_2[2]:.4f}")

    # --- OVERLAY CORNER PLOT FOR SHARED PARAMETERS (H0, Om0) ---
    # Base corner plot for Model I
    fig_overlay = corner.corner(
    chain1_flat[:, :2],
     labels=[r"$H_0 \ [\mathrm{km/s/Mpc}]$", r"$\Omega_{m0}$"],
     color="crimson",
     hist_kwargs={'density': True, 'linewidth': 1.5},
     plot_density=False,
     plot_datapoints=False,
     fill_contours=True,
     levels=(0.68, 0.95),
     contour_kwargs={'linewidths': 1.5, 'alpha': 0.8}
      )

 # Layer Model II on top
    corner.corner(
     chain2_flat[:, :2],
     fig=fig_overlay,
     color="navy",
     hist_kwargs={'density': True, 'linewidth': 1.5},
     plot_density=False,
     plot_datapoints=False,
     fill_contours=True,
     levels=(0.68, 0.95),
     contour_kwargs={'linewidths': 1.5, 'alpha': 0.8}
      ) 

 # Extract axes array from figure matrix
    axes = np.array(fig_overlay.axes).reshape((2, 2)) 

 # Utilize the empty top-right subplot (axes[0, 1]) for the legend
    ax_legend = axes[0, 1]
    ax_legend.set_visible(True)
    ax_legend.axis('off')

    line1 = mlines.Line2D([], [], color='crimson', lw=2, label=r'Model I ($G_5 = \xi X^2$)')
    line2 = mlines.Line2D([], [], color='navy', lw=2, label=r'Model II ($G_5 = \lambda X^4$)')

    ax_legend.legend(
     handles=[line1, line2],
     loc="center",
     fontsize=11,
     frameon=False
      )

 # Position title cleanly above subplots without colliding with the legend
    fig_overlay.suptitle(
     r"Posterior Overlay for Shared Parameters ($H_0, \Omega_{m0}$)",
     y=0.98,
     fontsize=13,
     fontweight='bold'
      )

 # Adjust plot boundaries to keep labels and title well-spaced
    fig_overlay.subplots_adjust(top=0.88, bottom=0.12, left=0.12, right=0.95, hspace=0.1, wspace=0.1)
 
    fig_overlay.savefig(
     os.path.join(out_dir, "shared_parameters_overlay.png"), 
     dpi=300, 
     bbox_inches="tight"
     )
    plt.close(fig_overlay) 
 
    print(f"\nOverlay plot saved to: {os.path.join(out_dir, 'shared_parameters_overlay.png')}")