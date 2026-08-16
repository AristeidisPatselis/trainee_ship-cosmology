"""
horndeski_cc_fit.py
======================
Sequential fitting script for Horndeski Gravity Models from arXiv:2110.01338 
(Petronikolou, Basilakos & Saridakis 2021).

Pipeline Flow:
    1. Loads Cosmic Chronometer (CC) dataset.
    2. SECTION 1: Model I (G5 ~ X, parameter alpha) -> Best-fit, emcee MCMC, Corner plot, H(z) plot.
    3. SECTION 2: Model II (G5 ~ X^2, parameter beta) -> Best-fit, emcee MCMC, Corner plot, H(z) plot.
    4. SECTION 3: Combined plot overlaying LCDM baseline, Model I, and Model II against CC data.
"""

import os
import numpy as np
from scipy.optimize import minimize
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ----------------------------------------------------------------------
# 0. Paths & Data Loading
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/"

_BASE_DATA_DIRS = [
    DATA_DIR,
    os.environ.get("COSMO_DATA_DIR"),
    HERE,
    os.path.join(HERE, "data"),
    os.getcwd(),
]

def find_data_file(filename):
    for d in _BASE_DATA_DIRS:
        if d and os.path.isdir(d):
            path = os.path.join(d, filename)
            if os.path.isfile(path):
                return path
            # search immediate subdirectories
            for sub in sorted(os.listdir(d)):
                sub_path = os.path.join(d, sub, filename)
                if os.path.isfile(sub_path):
                    return sub_path
    raise FileNotFoundError(f"Could not locate '{filename}'. Check data directories.")

z_data = np.loadtxt(find_data_file("c_z_vals.txt"))
H_data = np.loadtxt(find_data_file("c_H_vals.txt"))
sigma_data = np.loadtxt(find_data_file("c_sigma_vals.txt"))
N_DATA = len(z_data)
print(f"Loaded {N_DATA} Cosmic Chronometer data points.")

# Helper baseline LCDM
def H_lcdm(z, H0, Om0):
    return H0 * np.sqrt(Om0 * (1.0 + z)**3 + (1.0 - Om0))

# ----------------------------------------------------------------------
# Model Solver Definitions
# ----------------------------------------------------------------------
def solve_E_model1(z, Om0, alpha, n_iter=6):
    """E^2 - alpha * E^3 = Om0*(1+z)^3 + (1 - Om0 - alpha)"""
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
    """E^2 - beta * E^4 = Om0*(1+z)^3 + (1 - Om0 - beta)"""
    R = Om0 * (1.0 + z)**3 + (1.0 - Om0 - beta)
    if np.any(R <= 0): return None
    disc = 1.0 - 4.0 * beta * R
    if np.any(disc < 0): return None
    if abs(beta) < 1e-8:
        Y = R + beta * R**2
    else:
        Y = (1.0 - np.sqrt(disc)) / (2.0 * beta)
    if np.any(Y <= 0): return None
    return np.sqrt(Y)

def H_m1(z, theta):
    H0, Om0, alpha = theta
    E = solve_E_model1(z, Om0, alpha)
    return H0 * E if E is not None else None

def H_m2(z, theta):
    H0, Om0, beta = theta
    E = solve_E_model2(z, Om0, beta)
    return H0 * E if E is not None else None

# Likelihood wrappers
def chi2_m1(theta):
    H0, Om0, alpha = theta
    if not (30.0 < H0 < 150.0 and 0.0 < Om0 < 1.0 and -0.4 < alpha < 0.4): return np.inf
    Hm = H_m1(z_data, theta)
    return np.sum(((H_data - Hm) / sigma_data)**2) if Hm is not None else np.inf

def chi2_m2(theta):
    H0, Om0, beta = theta
    if not (30.0 < H0 < 150.0 and 0.0 < Om0 < 1.0 and -0.4 < beta < 0.4): return np.inf
    Hm = H_m2(z_data, theta)
    return np.sum(((H_data - Hm) / sigma_data)**2) if Hm is not None else np.inf

def log_prob_m1(theta):
    c2 = chi2_m1(theta)
    return -0.5 * c2 if np.isfinite(c2) else -np.inf

def log_prob_m2(theta):
    c2 = chi2_m2(theta)
    return -0.5 * c2 if np.isfinite(c2) else -np.inf


# ----------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    NWALKERS, NSTEPS, BURNIN = 32, 6000, 1500
    ncpu = max(1, cpu_count() - 1)
    rng = np.random.default_rng(42)
    z_grid = np.linspace(0, z_data.max() * 1.05, 400)

    # ==================================================================
    # SECTION 1: MODEL I FIT (G5 ~ X)
    # ==================================================================
    print("\n" + "="*60 + "\n   SECTION 1: RUNNING HORNDESKI MODEL I (G5 ~ X)\n" + "="*60)
    out1 = os.path.join(HERE, "outputs_cc_model1")
    os.makedirs(out1, exist_ok=True)

    res1 = minimize(chi2_m1, [70.0, 0.3, 0.01], method="Nelder-Mead")
    bf1 = res1.x
    chi2_min1 = res1.fun
    print(f"Model I Best Fit: H0={bf1[0]:.3f}, Om0={bf1[1]:.4f}, alpha={bf1[2]:.4f} | chi2={chi2_min1:.3f}")

    p0_1 = np.array(bf1) + 1e-3 * rng.standard_normal((NWALKERS, 3))
    with Pool(processes=ncpu) as pool:
        sampler1 = emcee.EnsembleSampler(NWALKERS, 3, log_prob_m1, pool=pool)
        sampler1.run_mcmc(p0_1, NSTEPS, progress=True)

    chain1 = sampler1.get_chain(discard=BURNIN, flat=True)
    np.save(os.path.join(out1, "model1_chain.npy"), chain1)

    # Corner Plot 1
    fig1 = corner.corner(
        chain1, labels=[r"$H_0$", r"$\Omega_{m0}$", r"$\alpha$"],
        quantiles=[0.16, 0.5, 0.84], show_titles=True, truths=bf1, truth_color="crimson"
    )
    fig1.suptitle("Posterior Distribution — Horndeski Model I", y=1.02)
    fig1.savefig(os.path.join(out1, "model1_corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # H(z) Plot 1
    idx1 = rng.choice(chain1.shape[0], size=min(1500, chain1.shape[0]), replace=False)
    H1_samples = np.array([H_m1(z_grid, chain1[i]) for i in idx1 if H_m1(z_grid, chain1[i]) is not None])
    H1_lo, H1_med, H1_hi = np.percentile(H1_samples, [16, 50, 84], axis=0)

    fig_h1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.errorbar(z_data, H_data, yerr=sigma_data, fmt="o", color="black", ecolor="gray", capsize=2, label="CC Data")
    ax1.plot(z_grid, H1_med, color="crimson", lw=2, label=rf"Model I Best Fit ($\alpha={bf1[2]:.3f}$)")
    ax1.fill_between(z_grid, H1_lo, H1_hi, color="crimson", alpha=0.2, label=r"$1\sigma$ band")
    ax1.set_xlabel(r"$z$"); ax1.set_ylabel(r"$H(z)$")
    ax1.legend(frameon=False)
    fig_h1.savefig(os.path.join(out1, "model1_Hz_fit.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_h1)

    # ==================================================================
    # SECTION 2: MODEL II FIT (G5 ~ X^2)
    # ==================================================================
    print("\n" + "="*60 + "\n   SECTION 2: RUNNING HORNDESKI MODEL II (G5 ~ X^2)\n" + "="*60)
    out2 = os.path.join(HERE, "outputs_cc_model2")
    os.makedirs(out2, exist_ok=True)

    res2 = minimize(chi2_m2, [70.0, 0.3, 0.01], method="Nelder-Mead")
    bf2 = res2.x
    chi2_min2 = res2.fun
    print(f"Model II Best Fit: H0={bf2[0]:.3f}, Om0={bf2[1]:.4f}, beta={bf2[2]:.4f} | chi2={chi2_min2:.3f}")

    p0_2 = np.array(bf2) + 1e-3 * rng.standard_normal((NWALKERS, 3))
    with Pool(processes=ncpu) as pool:
        sampler2 = emcee.EnsembleSampler(NWALKERS, 3, log_prob_m2, pool=pool)
        sampler2.run_mcmc(p0_2, NSTEPS, progress=True)

    chain2 = sampler2.get_chain(discard=BURNIN, flat=True)
    np.save(os.path.join(out2, "model2_chain.npy"), chain2)

    # Corner Plot 2
    fig2 = corner.corner(
        chain2, labels=[r"$H_0$", r"$\Omega_{m0}$", r"$\beta$"],
        quantiles=[0.16, 0.5, 0.84], show_titles=True, truths=bf2, truth_color="navy"
    )
    fig2.suptitle("Posterior Distribution — Horndeski Model II", y=1.02)
    fig2.savefig(os.path.join(out2, "model2_corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # H(z) Plot 2
    idx2 = rng.choice(chain2.shape[0], size=min(1500, chain2.shape[0]), replace=False)
    H2_samples = np.array([H_m2(z_grid, chain2[i]) for i in idx2 if H_m2(z_grid, chain2[i]) is not None])
    H2_lo, H2_med, H2_hi = np.percentile(H2_samples, [16, 50, 84], axis=0)

    fig_h2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.errorbar(z_data, H_data, yerr=sigma_data, fmt="o", color="black", ecolor="gray", capsize=2, label="CC Data")
    ax2.plot(z_grid, H2_med, color="navy", lw=2, label=rf"Model II Best Fit ($\beta={bf2[2]:.3f}$)")
    ax2.fill_between(z_grid, H2_lo, H2_hi, color="navy", alpha=0.2, label=r"$1\sigma$ band")
    ax2.set_xlabel(r"$z$"); ax2.set_ylabel(r"$H(z)$")
    ax2.legend(frameon=False)
    fig_h2.savefig(os.path.join(out2, "model2_Hz_fit.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_h2)

    # ==================================================================
    # SECTION 3: COMBINED COMPARISON PLOT (Model I vs Model II vs LCDM)
    # ==================================================================
    print("\n" + "="*60 + "\n   SECTION 3: GENERATING COMBINED MODEL COMPARISON\n" + "="*60)
    out_comp = os.path.join(HERE, "outputs_cc_comparison")
    os.makedirs(out_comp, exist_ok=True)

    fig_comp, ax_c = plt.subplots(figsize=(8, 5.5))
    ax_c.errorbar(z_data, H_data, yerr=sigma_data, fmt="o", ms=4, color="black",
                  ecolor="gray", elinewidth=1, capsize=2, label="CC Data")

    # Standard LCDM baseline reference (H0 = 68.0, Om0 = 0.3)
    H_lcdm_ref = H_lcdm(z_grid, 68.0, 0.3)
    ax_c.plot(z_grid, H_lcdm_ref, color="gray", linestyle="--", lw=1.8, label=r"$\Lambda$CDM ($H_0=68$)")

    # Horndeski Model I & II Best Fits
    ax_c.plot(z_grid, H1_med, color="crimson", lw=2, label=rf"Horndeski Model I ($\alpha={bf1[2]:.3f}$)")
    ax_c.plot(z_grid, H2_med, color="navy", lw=2, linestyle="-.", label=rf"Horndeski Model II ($\beta={bf2[2]:.3f}$)")

    ax_c.set_xlabel(r"Redshift $z$", fontsize=11)
    ax_c.set_ylabel(r"$H(z)\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$", fontsize=11)
    ax_c.set_title("Cosmic Chronometers: Horndeski Model I vs. Model II vs. $\Lambda$CDM", fontsize=12)
    ax_c.legend(frameon=False)
    fig_comp.tight_layout()
    fig_comp.savefig(os.path.join(out_comp, "horndeski_models_comparison.png"), dpi=250, bbox_inches="tight")
    plt.close(fig_comp)

    print(f"\nPipeline finished successfully. All outputs generated in:\n - {out1}\n - {out2}\n - {out_comp}")