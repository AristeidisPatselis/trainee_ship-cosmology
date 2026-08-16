"""
pantheon_horndeski_fit.py
========================
Sequential fitting script for Horndeski Gravity Models (Model I & Model II)
to the Pantheon+SH0ES SNIa compilation (Scolnic et al. 2022 / Brout et al. 2022).

Pipeline Flow:
    1. Load Pantheon+SH0ES catalog and full N x N STAT+SYS covariance.
    2. SECTION 1: Model I (G5 ~ X, params: H0, Om0, MB, alpha) -> Fit, MCMC, Corner & Hubble plots.
    3. SECTION 2: Model II (G5 ~ X^2, params: H0, Om0, MB, beta) -> Fit, MCMC, Corner & Hubble plots.
    4. SECTION 3: Combined Hubble diagram overlaying LCDM reference, Model I, and Model II.
"""

import os
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# ----------------------------------------------------------------------
# 0. Setup & Data Loading
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

def _expand_with_subdirs(base_dirs, max_depth=1):
    expanded = []
    seen = set()
    for d in base_dirs:
        if not d or not os.path.isdir(d):
            continue
        for path in (d,) + (
            tuple(os.path.join(d, e) for e in sorted(os.listdir(d))
                  if os.path.isdir(os.path.join(d, e))) if max_depth > 0 else ()
        ):
            if path not in seen:
                seen.add(path)
                expanded.append(path)
    return expanded

CANDIDATE_DATA_DIRS = _expand_with_subdirs(_BASE_DATA_DIRS)

def find_data_file(filename, candidate_dirs=CANDIDATE_DATA_DIRS):
    searched = []
    for d in candidate_dirs:
        if not d:
            continue
        path = os.path.join(d, filename)
        searched.append(path)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find '{filename}'.")

DAT_FILE = find_data_file("Pantheon+SH0ES.dat")
COV_FILE = find_data_file("Pantheon+SH0ES_STAT+SYS.cov")
print(f"Using data files from: {os.path.dirname(DAT_FILE)}")

C_LIGHT = 299792.458  # km/s
Z_CUT = 0.01          # Standard Pantheon+ cut

df = pd.read_csv(DAT_FILE, sep=r"\s+")
with open(COV_FILE) as f:
    n_cov = int(f.readline())
    cov_vals = np.fromfile(f, sep=" ")
assert n_cov == len(df), "Covariance dimension does not match SNe count."
C_full = cov_vals.reshape(n_cov, n_cov)

mask = (df["zHD"].values > Z_CUT) | (df["IS_CALIBRATOR"].values == 1)
idx = np.where(mask)[0]
sub = df.iloc[idx].reset_index(drop=True)
C = C_full[np.ix_(idx, idx)]

N_SN = len(sub)
N_CAL = int(sub["IS_CALIBRATOR"].sum())
print(f"Using {N_SN} SNe ({N_CAL} calibrators, {N_SN - N_CAL} Hubble-flow SNe with zHD > {Z_CUT}).")

zHD = sub["zHD"].values
zHEL = sub["zHEL"].values
m_b = sub["m_b_corr"].values
is_cal = sub["IS_CALIBRATOR"].values.astype(bool)
ceph_dist = sub["CEPH_DIST"].values

cho_fac = cho_factor(C, lower=True)
Z_GRID = np.linspace(0.0, zHD.max() + 0.01, 4000)

# ----------------------------------------------------------------------
# Model Solvers & Distance Modulus Functions
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

def mu_model_lcdm(H0, Om0):
    E = np.sqrt(Om0 * (1.0 + Z_GRID)**3 + (1.0 - Om0))
    cum_int = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
    Iz = np.interp(zHD, Z_GRID, cum_int)
    dL = (1.0 + zHEL) * (C_LIGHT / H0) * Iz
    return np.where(is_cal, ceph_dist, 5.0 * np.log10(dL) + 25.0)

def mu_model_m1(H0, Om0, alpha):
    E = solve_E_model1(Z_GRID, Om0, alpha)
    if E is None: return None
    cum_int = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
    Iz = np.interp(zHD, Z_GRID, cum_int)
    dL = (1.0 + zHEL) * (C_LIGHT / H0) * Iz
    return np.where(is_cal, ceph_dist, 5.0 * np.log10(dL) + 25.0)

def mu_model_m2(H0, Om0, beta):
    E = solve_E_model2(Z_GRID, Om0, beta)
    if E is None: return None
    cum_int = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
    Iz = np.interp(zHD, Z_GRID, cum_int)
    dL = (1.0 + zHEL) * (C_LIGHT / H0) * Iz
    return np.where(is_cal, ceph_dist, 5.0 * np.log10(dL) + 25.0)

def chi2_m1(theta):
    H0, Om0, MB, alpha = theta
    if not (30.0 < H0 < 150.0 and 0.0 < Om0 < 1.0 and -21.0 < MB < -17.0 and -0.4 < alpha < 0.4):
        return np.inf
    mu_mod = mu_model_m1(H0, Om0, alpha)
    if mu_mod is None: return np.inf
    r = m_b - MB - mu_mod
    return float(r @ cho_solve(cho_fac, r))

def chi2_m2(theta):
    H0, Om0, MB, beta = theta
    if not (30.0 < H0 < 150.0 and 0.0 < Om0 < 1.0 and -21.0 < MB < -17.0 and -0.4 < beta < 0.4):
        return np.inf
    mu_mod = mu_model_m2(H0, Om0, beta)
    if mu_mod is None: return np.inf
    r = m_b - MB - mu_mod
    return float(r @ cho_solve(cho_fac, r))

def log_prob_m1(theta):
    c2 = chi2_m1(theta)
    return -0.5 * c2 if np.isfinite(c2) else -np.inf

def log_prob_m2(theta):
    c2 = chi2_m2(theta)
    return -0.5 * c2 if np.isfinite(c2) else -np.inf

# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    NWALKERS, NSTEPS, BURNIN = 32, 4000, 1000
    ncpu = max(1, cpu_count() - 1)
    rng = np.random.default_rng(42)

    # ==================================================================
    # SECTION 1: MODEL I FIT (G5 ~ X)
    # ==================================================================
    print("\n" + "="*60 + "\n   SECTION 1: RUNNING PANTHEON MODEL I (G5 ~ X)\n" + "="*60)
    out1 = os.path.join(HERE, "outputs_sn_model1")
    os.makedirs(out1, exist_ok=True)

    res1 = minimize(chi2_m1, [70.0, 0.3, -19.3, 0.01], method="Nelder-Mead")
    bf1 = res1.x
    print(f"Model I Best Fit: H0={bf1[0]:.3f}, Om0={bf1[1]:.4f}, MB={bf1[2]:.4f}, alpha={bf1[3]:.4f} | chi2={res1.fun:.3f}")

    p0_1 = np.array(bf1) + np.array([1e-2, 1e-3, 1e-3, 1e-3]) * rng.standard_normal((NWALKERS, 4))
    with Pool(processes=ncpu) as pool:
        sampler1 = emcee.EnsembleSampler(NWALKERS, 4, log_prob_m1, pool=pool)
        sampler1.run_mcmc(p0_1, NSTEPS, progress=True)

    chain1 = sampler1.get_chain(discard=BURNIN, flat=True)
    np.save(os.path.join(out1, "pantheon_model1_chain.npy"), chain1)

    # Corner Plot 1
    fig1 = corner.corner(
        chain1, labels=[r"$H_0$", r"$\Omega_{m0}$", r"$M_B$", r"$\alpha$"],
        quantiles=[0.16, 0.5, 0.84], show_titles=True, truths=bf1, truth_color="crimson"
    )
    fig1.suptitle(r"Posterior Distribution — Horndeski Model I (Pantheon+)", y=1.02)
    fig1.savefig(os.path.join(out1, "pantheon_model1_corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Hubble Diagram 1
    mu_obs1 = m_b - bf1[2]
    mu_th1 = mu_model_m1(bf1[0], bf1[1], bf1[3])
    resid1 = mu_obs1 - mu_th1
    sigma_diag = np.sqrt(np.diag(C))

    fig_h1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    hf = ~is_cal
    ax1.errorbar(zHD[hf], mu_obs1[hf], yerr=sigma_diag[hf], fmt=".", ms=3, alpha=0.4, color="steelblue", label="Hubble-flow SNe")
    ax1.errorbar(zHD[is_cal], mu_obs1[is_cal], yerr=sigma_diag[is_cal], fmt="*", ms=7, color="darkorange", label="SH0ES calibrators")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\mu = m_B - M_B$")
    ax1.set_title(rf"Horndeski Model I Fit ($\alpha={bf1[3]:.3f}$)")
    ax1.legend(frameon=False)

    ax2.axhline(0, color="crimson", lw=1.5)
    ax2.scatter(zHD[hf], resid1[hf], s=6, alpha=0.4, color="steelblue")
    ax2.scatter(zHD[is_cal], resid1[is_cal], s=25, marker="*", color="darkorange")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$\mu_{\rm obs}-\mu_{\rm th}$")
    fig_h1.tight_layout()
    fig_h1.savefig(os.path.join(out1, "pantheon_model1_hubble.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_h1)

    # ==================================================================
    # SECTION 2: MODEL II FIT (G5 ~ X^2)
    # ==================================================================
    print("\n" + "="*60 + "\n   SECTION 2: RUNNING PANTHEON MODEL II (G5 ~ X^2)\n" + "="*60)
    out2 = os.path.join(HERE, "outputs_sn_model2")
    os.makedirs(out2, exist_ok=True)

    res2 = minimize(chi2_m2, [70.0, 0.3, -19.3, 0.01], method="Nelder-Mead")
    bf2 = res2.x
    print(f"Model II Best Fit: H0={bf2[0]:.3f}, Om0={bf2[1]:.4f}, MB={bf2[2]:.4f}, beta={bf2[3]:.4f} | chi2={res2.fun:.3f}")

    p0_2 = np.array(bf2) + np.array([1e-2, 1e-3, 1e-3, 1e-3]) * rng.standard_normal((NWALKERS, 4))
    with Pool(processes=ncpu) as pool:
        sampler2 = emcee.EnsembleSampler(NWALKERS, 4, log_prob_m2, pool=pool)
        sampler2.run_mcmc(p0_2, NSTEPS, progress=True)

    chain2 = sampler2.get_chain(discard=BURNIN, flat=True)
    np.save(os.path.join(out2, "pantheon_model2_chain.npy"), chain2)

    # Corner Plot 2
    fig2 = corner.corner(
        chain2, labels=[r"$H_0$", r"$\Omega_{m0}$", r"$M_B$", r"$\beta$"],
        quantiles=[0.16, 0.5, 0.84], show_titles=True, truths=bf2, truth_color="navy"
    )
    fig2.suptitle(r"Posterior Distribution — Horndeski Model II (Pantheon+)", y=1.02)
    fig2.savefig(os.path.join(out2, "pantheon_model2_corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Hubble Diagram 2
    mu_obs2 = m_b - bf2[2]
    mu_th2 = mu_model_m2(bf2[0], bf2[1], bf2[3])
    resid2 = mu_obs2 - mu_th2

    fig_h2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax1.errorbar(zHD[hf], mu_obs2[hf], yerr=sigma_diag[hf], fmt=".", ms=3, alpha=0.4, color="steelblue", label="Hubble-flow SNe")
    ax1.errorbar(zHD[is_cal], mu_obs2[is_cal], yerr=sigma_diag[is_cal], fmt="*", ms=7, color="darkorange", label="SH0ES calibrators")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\mu = m_B - M_B$")
    ax1.set_title(rf"Horndeski Model II Fit ($\beta={bf2[3]:.3f}$)")
    ax1.legend(frameon=False)

    ax2.axhline(0, color="navy", lw=1.5)
    ax2.scatter(zHD[hf], resid2[hf], s=6, alpha=0.4, color="steelblue")
    ax2.scatter(zHD[is_cal], resid2[is_cal], s=25, marker="*", color="darkorange")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$\mu_{\rm obs}-\mu_{\rm th}$")
    fig_h2.tight_layout()
    fig_h2.savefig(os.path.join(out2, "pantheon_model2_hubble.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_h2)

    # ==================================================================
    # SECTION 3: COMBINED COMPARISON PLOT
    # ==================================================================
    print("\n" + "="*60 + "\n   SECTION 3: GENERATING COMBINED MODEL COMPARISON\n" + "="*60)
    out_comp = os.path.join(HERE, "outputs_sn_comparison")
    os.makedirs(out_comp, exist_ok=True)

    z_grid_plot = np.geomspace(max(zHD.min(), 1e-3), zHD.max(), 400)

    # Curve calculation helpers
    def get_mu_curve_m1(H0, Om0, alpha):
        E = solve_E_model1(Z_GRID, Om0, alpha)
        cum = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
        Iz = np.interp(z_grid_plot, Z_GRID, cum)
        dL = (1.0 + z_grid_plot) * (C_LIGHT / H0) * Iz
        return 5.0 * np.log10(dL) + 25.0

    def get_mu_curve_m2(H0, Om0, beta):
        E = solve_E_model2(Z_GRID, Om0, beta)
        cum = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
        Iz = np.interp(z_grid_plot, Z_GRID, cum)
        dL = (1.0 + z_grid_plot) * (C_LIGHT / H0) * Iz
        return 5.0 * np.log10(dL) + 25.0

    def get_mu_curve_lcdm(H0, Om0):
        E = np.sqrt(Om0 * (1.0 + Z_GRID)**3 + (1.0 - Om0))
        cum = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
        Iz = np.interp(z_grid_plot, Z_GRID, cum)
        dL = (1.0 + z_grid_plot) * (C_LIGHT / H0) * Iz
        return 5.0 * np.log10(dL) + 25.0

    mu_ref_obs = m_b - bf1[2]

    fig_comp, ax_c = plt.subplots(figsize=(8, 5.5))
    ax_c.errorbar(zHD[hf], mu_ref_obs[hf], yerr=sigma_diag[hf], fmt=".", ms=3, alpha=0.3, color="gray", label="Hubble-flow SNe")
    ax_c.errorbar(zHD[is_cal], mu_ref_obs[is_cal], yerr=sigma_diag[is_cal], fmt="*", ms=6, color="darkorange", label="SH0ES calibrators")

    ax_c.plot(z_grid_plot, get_mu_curve_lcdm(70.0, 0.3), color="black", linestyle="--", lw=1.8, label=r"$\Lambda$CDM ($H_0=70$)")
    ax_c.plot(z_grid_plot, get_mu_curve_m1(bf1[0], bf1[1], bf1[3]), color="crimson", lw=2, label=rf"Horndeski Model I ($\alpha={bf1[3]:.3f}$)")
    ax_c.plot(z_grid_plot, get_mu_curve_m2(bf2[0], bf2[1], bf2[3]), color="navy", lw=2, linestyle="-.", label=rf"Horndeski Model II ($\beta={bf2[3]:.3f}$)")

    ax_c.set_xscale("log")
    ax_c.set_xlabel(r"Redshift $z$", fontsize=11)
    ax_c.set_ylabel(r"Distance Modulus $\mu$", fontsize=11)
    ax_c.set_title(r"Pantheon+ SNIa: Horndeski Model I vs. Model II vs. $\Lambda$CDM", fontsize=12)
    ax_c.legend(frameon=False)
    fig_comp.tight_layout()
    fig_comp.savefig(os.path.join(out_comp, "pantheon_horndeski_comparison.png"), dpi=250, bbox_inches="tight")
    plt.close(fig_comp)

    print(f"\nPipeline completed. Outputs generated in:\n - {out1}\n - {out2}\n - {out_comp}")