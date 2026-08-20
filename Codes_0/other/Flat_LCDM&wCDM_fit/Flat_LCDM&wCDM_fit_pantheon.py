"""
pantheon_lcdm_fit.py
=====================
Flat LCDM & wCDM fit to the Pantheon+SH0ES SNIa compilation (Scolnic et al. 2022 /
Brout et al. 2022), using the full STAT+SYS covariance matrix and the SH0ES
Cepheid-host calibrator sample to break the H0-M_B degeneracy.

Fixes applied:
    - Preserves clean LCDM / wCDM baseline cosmology without unintended model overrides.
    - Uses raw f-strings (rf"...") for all Matplotlib LaTeX labels to prevent escape sequence errors.
    - Ensures numerically robust Cholesky solves and positive-definite E(z) integration.
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
# 0. Setup & Data Search
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "outputs_sn")
os.makedirs(OUTDIR, exist_ok=True)

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
    raise FileNotFoundError(f"Could not find '{filename}'. Searched in:\n  " + "\n  ".join(searched))

DAT_FILE = find_data_file("Pantheon+SH0ES.dat")
COV_FILE = find_data_file("Pantheon+SH0ES_STAT+SYS.cov")
print(f"Using data files from: {os.path.dirname(DAT_FILE)}")

C_LIGHT = 299792.458  # km/s
Z_CUT = 0.01          # Standard Pantheon+ cosmology cut

# ----------------------------------------------------------------------
# 1. Load Data & Covariance
# ----------------------------------------------------------------------
df = pd.read_csv(DAT_FILE, sep=r"\s+")

with open(COV_FILE) as f:
    n_cov = int(f.readline())
    cov_vals = np.fromfile(f, sep=" ")
assert n_cov == len(df), "Covariance dimension does not match data table."
C_full = cov_vals.reshape(n_cov, n_cov)

mask = (df["zHD"].values > Z_CUT) | (df["IS_CALIBRATOR"].values == 1)
idx = np.where(mask)[0]
sub = df.iloc[idx].reset_index(drop=True)
C = C_full[np.ix_(idx, idx)]

N_SN = len(sub)
N_CAL = int(sub["IS_CALIBRATOR"].sum())
print(f"Using {N_SN} SNe ({N_CAL} SH0ES Cepheid calibrators, {N_SN - N_CAL} Hubble-flow SNe).")

zHD = sub["zHD"].values
zHEL = sub["zHEL"].values
m_b = sub["m_b_corr"].values
is_cal = sub["IS_CALIBRATOR"].values.astype(bool)
ceph_dist = sub["CEPH_DIST"].values

cho_fac = cho_factor(C, lower=True)
Z_GRID = np.linspace(0.0, zHD.max() + 0.01, 4000)

# ----------------------------------------------------------------------
# 2. Cosmological Distance Modulus Models
# ----------------------------------------------------------------------
def E_lcdm(z, Om0):
    """Flat LCDM dimensionless Hubble parameter."""
    return np.sqrt(Om0 * (1.0 + z)**3 + (1.0 - Om0))

def E_wcdm(z, Om0, w):
    """Flat wCDM dimensionless Hubble parameter."""
    arg = Om0 * (1.0 + z)**3 + (1.0 - Om0) * (1.0 + z)**(3.0 * (1.0 + w))
    if np.any(arg <= 0):
        return None
    return np.sqrt(arg)

def mu_model_lcdm(H0, Om0):
    E = E_lcdm(Z_GRID, Om0)
    cum_int = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
    Iz = np.interp(zHD, Z_GRID, cum_int)
    dL = (1.0 + zHEL) * (C_LIGHT / H0) * Iz
    mu_th = 5.0 * np.log10(dL) + 25.0
    return np.where(is_cal, ceph_dist, mu_th)

def mu_model_wcdm(H0, Om0, w):
    E = E_wcdm(Z_GRID, Om0, w)
    if E is None:
        return None
    cum_int = cumulative_trapezoid(1.0 / E, Z_GRID, initial=0.0)
    Iz = np.interp(zHD, Z_GRID, cum_int)
    dL = (1.0 + zHEL) * (C_LIGHT / H0) * Iz
    mu_th = 5.0 * np.log10(dL) + 25.0
    return np.where(is_cal, ceph_dist, mu_th)

# ----------------------------------------------------------------------
# 3. Likelihood & Optimization
# ----------------------------------------------------------------------
def chi2_pantheon_lcdm(theta):
    H0, Om0, M_B = theta
    if not (30.0 < H0 < 150.0 and 0.0 < Om0 < 1.0 and -21.0 < M_B < -17.0):
        return np.inf
    mu_model = mu_model_lcdm(H0, Om0)
    r = m_b - M_B - mu_model
    y = cho_solve(cho_fac, r)
    return float(r @ y)

def log_probability_lcdm(theta):
    c2 = chi2_pantheon_lcdm(theta)
    return -0.5 * c2 if np.isfinite(c2) else -np.inf

# ----------------------------------------------------------------------
# 4. Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    theta0 = [70.0, 0.3, -19.3]
    res = minimize(chi2_pantheon_lcdm, theta0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-5})
    H0_bf, Om0_bf, MB_bf = res.x
    chi2_min = res.fun
    dof = N_SN - 3
    
    print("\n--- Flat LCDM Best Fit ---")
    print(f"H0  = {H0_bf:.3f} km/s/Mpc")
    print(f"Om0 = {Om0_bf:.4f}")
    print(f"M_B = {MB_bf:.4f}")
    print(f"chi2 = {chi2_min:.3f} (dof = {dof}, chi2/dof = {chi2_min/dof:.3f})")

    # MCMC Sampling
    NDIM, NWALKERS, NSTEPS, BURNIN = 3, 32, 4000, 1000
    ncpu = max(1, cpu_count() - 1)
    rng = np.random.default_rng(42)
    p0 = np.array([H0_bf, Om0_bf, MB_bf]) + np.array([1e-2, 1e-3, 1e-3]) * rng.standard_normal((NWALKERS, NDIM))

    print(f"\nRunning MCMC with {NWALKERS} walkers and {NSTEPS} steps on {ncpu} cores...")
    with Pool(processes=ncpu) as pool:
        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_probability_lcdm, pool=pool)
        sampler.run_mcmc(p0, NSTEPS, progress=True)

    chain = sampler.get_chain(discard=BURNIN, flat=True)
    np.save(os.path.join(OUTDIR, "pantheon_chain.npy"), chain)

    # Corner Plot
    fig = corner.corner(
        chain,
        labels=[r"$H_0\ [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$", r"$\Omega_{m0}$", r"$M_B$"],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        truths=[H0_bf, Om0_bf, MB_bf],
        truth_color="crimson",
        color="steelblue",
    )
    fig.suptitle(r"$\Lambda$CDM Posterior — Pantheon+SH0ES SNIa", y=1.02)
    fig.savefig(os.path.join(OUTDIR, "pantheon_corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Hubble Diagram & Residuals
    mu_obs = m_b - MB_bf
    mu_th_bf = mu_model_lcdm(H0_bf, Om0_bf)
    resid = mu_obs - mu_th_bf
    sigma_diag = np.sqrt(np.diag(C))

    z_grid_plot = np.geomspace(max(zHD.min(), 1e-3), zHD.max(), 300)
    E_grid = E_lcdm(Z_GRID, Om0_bf)
    cum_int_plot = cumulative_trapezoid(1.0 / E_grid, Z_GRID, initial=0.0)
    Iz_plot = np.interp(z_grid_plot, Z_GRID, cum_int_plot)
    dL_plot = (1.0 + z_grid_plot) * (C_LIGHT / H0_bf) * Iz_plot
    mu_curve = 5.0 * np.log10(dL_plot) + 25.0

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    hf = ~is_cal

    ax1.errorbar(zHD[hf], mu_obs[hf], yerr=sigma_diag[hf], fmt=".", ms=3, alpha=0.4, color="steelblue", ecolor="steelblue", label="Hubble-flow SNe")
    ax1.errorbar(zHD[is_cal], mu_obs[is_cal], yerr=sigma_diag[is_cal], fmt="*", ms=7, color="darkorange", ecolor="darkorange", label="SH0ES calibrators")
    ax1.plot(z_grid_plot, mu_curve, color="crimson", lw=2, label=rf"$\Lambda$CDM Best Fit ($H_0={H0_bf:.1f}$, $\Omega_{{m0}}={Om0_bf:.3f}$)")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\mu = m_B - M_B$")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title(r"$\Lambda$CDM Fit to Pantheon+SH0ES Hubble Diagram")

    ax2.axhline(0, color="crimson", lw=1.5)
    ax2.scatter(zHD[hf], resid[hf], s=6, alpha=0.4, color="steelblue")
    ax2.scatter(zHD[is_cal], resid[is_cal], s=25, marker="*", color="darkorange")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$\mu_{\rm obs}-\mu_{\rm th}$")

    fig2.tight_layout()
    fig2.savefig(os.path.join(OUTDIR, "pantheon_hubble_fit.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nExecution complete. Outputs written to: {OUTDIR}")