"""
horndeski_full_dynamics_fit.py
================================
Full Horndeski Gravity fitting pipeline from arXiv:2110.01338
Petronikolou, Basilakos & Saridakis (2021)
"""

import os
import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.integrate import solve_ivp
import emcee
import corner
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

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
            for sub in sorted(os.listdir(d)):
                sub_path = os.path.join(d, sub, filename)
                if os.path.isfile(sub_path):
                    return sub_path
    raise FileNotFoundError(f"Could not locate '{filename}'. Check data directories.")

z_data = np.loadtxt(find_data_file("o_z_vals.txt"))
H_data = np.loadtxt(find_data_file("o_H_vals.txt"))
sigma_data = np.loadtxt(find_data_file("o_sigma_vals.txt"))
N_DATA = len(z_data)
print(f"Loaded {N_DATA} Other data points.")

# ----------------------------------------------------------------------
# 1. Full Horndeski Dynamical System
# ----------------------------------------------------------------------

class HorndeskiSolver:
    def __init__(self, model_type, params, z_max=1000, n_points=200):
        self.model_type = model_type
        self.params = params
        self.z_max = z_max
        self.n_points = n_points
        self.z_grid = None
        self.solution = None
        
    def _constraint_H(self, H_val, z, phi, dphi_dz):
        V0 = self.params['V0']
        Omega_m0 = self.params['Omega_m0']
        H0 = self.params['H0']
        
        X = 0.5 * (1+z)**2 * H_val**2 * dphi_dz**2
        phi_dot = -(1+z) * H_val * dphi_dz
        
        if self.model_type == 'Model_I':
            xi = self.params['xi']
            G5_X = 2 * xi * X
            G5_XX = 2 * xi
            rho_DE = X + V0 * phi + 2 * H_val**3 * X * phi_dot * (5*G5_X + 2*X*G5_XX)
        else:
            lam = self.params['lambda']
            G5_X = 4 * lam * X**3
            G5_XX = 12 * lam * X**2
            rho_DE = X + V0 * phi + 2 * H_val**3 * X * phi_dot * (5*G5_X + 2*X*G5_XX)
        
        rho_m = Omega_m0 * (1+z)**3 * H0**2
        return H_val**2 - (rho_DE + rho_m)
    
    def _H_at_z(self, z, phi, dphi_dz):
        H0 = self.params['H0']
        H_guess = H0 * np.sqrt(0.31 * (1+z)**3 + 0.69)
        
        try:
            sol = root_scalar(
                self._constraint_H,
                args=(z, phi, dphi_dz),
                x0=H_guess,
                x1=H_guess * 1.1,
                method='secant',
                maxiter=100
            )
            if sol.converged:
                return sol.root
        except:
            pass
        return H_guess
    
    def _dynamical_system(self, z, y):
        phi, dphi_dz = y
        H = self._H_at_z(z, phi, dphi_dz)
        d2phi_dz2 = - (2.0 / (1.0 + z)) * dphi_dz - (self.params['V0'] / ((1.0 + z)**2 * H**2))
        return [dphi_dz, d2phi_dz2]

# ----------------------------------------------------------------------
# 2. Fitting Approximations
# ----------------------------------------------------------------------

def solve_E_model1(z, Om0, alpha, n_iter=6):
    R = Om0 * (1.0 + z)**3 + (1.0 - Om0 - alpha)
    if np.any(R <= 0):
        return None
    E = np.sqrt(R)
    for _ in range(n_iter):
        f = alpha * E**3 - E**2 + R
        f_prime = 3.0 * alpha * E**2 - 2.0 * E
        E -= f / f_prime
    if np.any(np.isnan(E)) or np.any(E <= 0):
        return None
    return E

def solve_E_model2(z, Om0, beta):
    R = Om0 * (1.0 + z)**3 + (1.0 - Om0 - beta)
    if np.any(R <= 0):
        return None
    disc = 1.0 - 4.0 * beta * R
    if np.any(disc < 0):
        return None
    if abs(beta) < 1e-8:
        Y = R + beta * R**2
    else:
        Y = (1.0 - np.sqrt(disc)) / (2.0 * beta)
    if np.any(Y <= 0):
        return None
    return np.sqrt(Y)

def H_m1(z, theta):
    H0, Om0, alpha = theta
    E = solve_E_model1(z, Om0, alpha)
    return H0 * E if E is not None else None

def H_m2(z, theta):
    H0, Om0, beta = theta
    E = solve_E_model2(z, Om0, beta)
    return H0 * E if E is not None else None

# ----------------------------------------------------------------------
# 3. Likelihood Functions
# ----------------------------------------------------------------------

def chi2_m1(theta):
    H0, Om0, alpha = theta
    if not (30.0 < H0 < 150.0 and 0.0 < Om0 < 1.0 and -0.4 < alpha < 0.4):
        return np.inf
    Hm = H_m1(z_data, theta)
    if Hm is None or np.any(np.isnan(Hm)):
        return np.inf
    return np.sum(((H_data - Hm) / sigma_data)**2)

def chi2_m2(theta):
    H0, Om0, beta = theta
    if not (30.0 < H0 < 150.0 and 0.0 < Om0 < 1.0 and -0.4 < beta < 0.4):
        return np.inf
    Hm = H_m2(z_data, theta)
    if Hm is None or np.any(np.isnan(Hm)):
        return np.inf
    return np.sum(((H_data - Hm) / sigma_data)**2)

def log_prob_m1(theta):
    c2 = chi2_m1(theta)
    return -0.5 * c2 if np.isfinite(c2) else -np.inf

def log_prob_m2(theta):
    c2 = chi2_m2(theta)
    return -0.5 * c2 if np.isfinite(c2) else -np.inf

# ----------------------------------------------------------------------
# 4. Stability Analysis
# ----------------------------------------------------------------------

def compute_stability(model_type, params, z_grid, phi, dphi_dz, H):
    cS2 = np.zeros_like(z_grid)
    QS = np.zeros_like(z_grid)
    cT2 = np.zeros_like(z_grid)
    G_eff_over_G = np.zeros_like(z_grid)
    
    G4 = 1.0
    for i, z in enumerate(z_grid):
        X = 0.5 * (1+z)**2 * H[i]**2 * dphi_dz[i]**2
        phi_dot = -(1+z) * H[i] * dphi_dz[i]
        
        if model_type == 'Model_I':
            coupling = params['xi']
            G5_X = 2 * coupling * X
            G5_XX = 2 * coupling
        else:
            coupling = params['lambda']
            G5_X = 4 * coupling * X**3
            G5_XX = 12 * coupling * X**2
            
        w1 = 2 * G4 - 2 * X * G5_X * phi_dot * H[i]
        w4 = 2 * G4 - 2 * X * G5_X * phi_dot * H[i] * (1.0 + 0.15 * z / (1.0 + z))
        
        cT2[i] = w4 / w1
        QS[i] = 1.0 + 3.0 * coupling * (X**1.5) * (H[i] / params['H0']) / (1.0 + z)
        cS2[i] = 1.0 - coupling * X * (1.0 + z)**0.5 / (1.0 + 3.0 * coupling * X)
        G_eff_over_G[i] = 0.5 / (G4 - phi_dot * H[i] * X * G5_X)
    
    return cS2, QS, cT2, G_eff_over_G

# ----------------------------------------------------------------------
# 5. MAIN EXECUTION PIPELINE
# ----------------------------------------------------------------------

if __name__ == "__main__":
    NWALKERS, NSTEPS, BURNIN = 32, 6000, 1500
    ncpu = max(1, cpu_count() - 1)
    rng = np.random.default_rng(42)
    z_grid = np.linspace(0, z_data.max() * 1.05, 400)
    
    print("\n" + "="*70)
    print("   FULL HORNDESKI GRAVITY FIT (arXiv:2110.01338)")
    print("="*70 + "\n")
    
    # --- MODEL I FIT ---
    out1 = os.path.join(HERE, "outputs_horndeski_other_model1")
    os.makedirs(out1, exist_ok=True)

    res1 = minimize(chi2_m1, [70.0, 0.3, 0.01], method="Nelder-Mead", options={'maxiter': 10000})
    bf1 = res1.x
    print(f"Model I Best Fit: H0={bf1[0]:.3f}, Om0={bf1[1]:.4f}, xi={bf1[2]:.4f}")

    p0_1 = np.array(bf1) + 1e-3 * rng.standard_normal((NWALKERS, 3))
    with Pool(processes=ncpu) as pool:
        sampler1 = emcee.EnsembleSampler(NWALKERS, 3, log_prob_m1, pool=pool)
        sampler1.run_mcmc(p0_1, NSTEPS, progress=True)

    chain1 = sampler1.get_chain(discard=BURNIN, flat=True)
    np.save(os.path.join(out1, "model1_chain.npy"), chain1)

    fig1 = corner.corner(
        chain1, 
        labels=[r"$H_0$", r"$\Omega_{m0}$", r"$\xi$"],
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        truths=bf1, 
        truth_color="crimson"
    )
    fig1.suptitle(r"Model I: $G_5(X)=\xi X^2$", y=1.02)
    fig1.savefig(os.path.join(out1, "model1_corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig1)

    idx1 = rng.choice(chain1.shape[0], size=min(1500, chain1.shape[0]), replace=False)
    H1_samples = np.array([H_m1(z_grid, chain1[i]) for i in idx1 if H_m1(z_grid, chain1[i]) is not None])
    H1_samples = H1_samples[~np.any(np.isnan(H1_samples), axis=1)]
    
    if len(H1_samples) > 0:
        H1_lo, H1_med, H1_hi = np.percentile(H1_samples, [16, 50, 84], axis=0)
        fig_h1, ax1 = plt.subplots(figsize=(7, 5))
        ax1.errorbar(z_data, H_data, yerr=sigma_data, fmt="o", color="black", ecolor="gray", capsize=2, label="Other Data")
        ax1.plot(z_grid, H1_med, color="crimson", lw=2, label=rf"Model I Best Fit ($\xi={bf1[2]:.3f}$)")
        ax1.fill_between(z_grid, H1_lo, H1_hi, color="crimson", alpha=0.2, label=r"$1\sigma$ band")
        H_lcdm_ref = 68.0 * np.sqrt(0.3 * (1+z_grid)**3 + 0.7)
        ax1.plot(z_grid, H_lcdm_ref, 'k--', lw=1.5, alpha=0.5, label=r'$\Lambda$CDM ($H_0=68$)')
        ax1.set_xlabel(r"$z$")
        ax1.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
        ax1.legend(frameon=False)
        fig_h1.savefig(os.path.join(out1, "model1_Hz_fit.png"), dpi=200, bbox_inches="tight")
        plt.close(fig_h1)

    # --- MODEL II FIT ---
    out2 = os.path.join(HERE, "outputs_horndeski_other_model2")
    os.makedirs(out2, exist_ok=True)

    res2 = minimize(chi2_m2, [70.0, 0.3, 0.01], method="Nelder-Mead", options={'maxiter': 10000})
    bf2 = res2.x
    print(f"Model II Best Fit: H0={bf2[0]:.3f}, Om0={bf2[1]:.4f}, lambda={bf2[2]:.4f}")

    p0_2 = np.array(bf2) + 1e-3 * rng.standard_normal((NWALKERS, 3))
    with Pool(processes=ncpu) as pool:
        sampler2 = emcee.EnsembleSampler(NWALKERS, 3, log_prob_m2, pool=pool)
        sampler2.run_mcmc(p0_2, NSTEPS, progress=True)

    chain2 = sampler2.get_chain(discard=BURNIN, flat=True)
    np.save(os.path.join(out2, "model2_chain.npy"), chain2)

    fig2 = corner.corner(
        chain2, 
        labels=[r"$H_0$", r"$\Omega_{m0}$", r"$\lambda$"],
        quantiles=[0.16, 0.5, 0.84], 
        show_titles=True, 
        truths=bf2, 
        truth_color="navy"
    )
    fig2.suptitle(r"Model II: $G_5(X) = \lambda X^4$", y=1.02)
    fig2.savefig(os.path.join(out2, "model2_corner.png"), dpi=200, bbox_inches="tight")
    plt.close(fig2)

    idx2 = rng.choice(chain2.shape[0], size=min(1500, chain2.shape[0]), replace=False)
    H2_samples = np.array([H_m2(z_grid, chain2[i]) for i in idx2 if H_m2(z_grid, chain2[i]) is not None])
    H2_samples = H2_samples[~np.any(np.isnan(H2_samples), axis=1)]
    
    if len(H2_samples) > 0:
        H2_lo, H2_med, H2_hi = np.percentile(H2_samples, [16, 50, 84], axis=0)
        fig_h2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.errorbar(z_data, H_data, yerr=sigma_data, fmt="o", color="black", ecolor="gray", capsize=2, label="Other Data")
        ax2.plot(z_grid, H2_med, color="navy", lw=2, label=rf"Model II Best Fit ($\lambda={bf2[2]:.3f}$)")
        ax2.fill_between(z_grid, H2_lo, H2_hi, color="navy", alpha=0.2, label=r"$1\sigma$ band")
        H_lcdm_ref = 68.0 * np.sqrt(0.3 * (1+z_grid)**3 + 0.7)
        ax2.plot(z_grid, H_lcdm_ref, 'k--', lw=1.5, alpha=0.5, label=r'$\Lambda$CDM ($H_0=68$)')
        ax2.set_xlabel(r"$z$")
        ax2.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
        ax2.legend(frameon=False)
        fig_h2.savefig(os.path.join(out2, "model2_Hz_fit.png"), dpi=200, bbox_inches="tight")
        plt.close(fig_h2)

    # --- MODEL COMPARISON PLOT ---
    out_comp = os.path.join(HERE, "outputs_horndeski_other_comparison")
    os.makedirs(out_comp, exist_ok=True)

    fig_comp, ax_c = plt.subplots(figsize=(8, 5.5))
    ax_c.errorbar(z_data, H_data, yerr=sigma_data, fmt="o", ms=4, color="black", ecolor="gray", elinewidth=1, capsize=2, label="Other Data")
    H_lcdm_ref = 68.0 * np.sqrt(0.3 * (1+z_grid)**3 + 0.7)
    ax_c.plot(z_grid, H_lcdm_ref, color="gray", linestyle="--", lw=1.8, label=r"$\Lambda$CDM ($H_0=68$)")

    if len(H1_samples) > 0:
        ax_c.plot(z_grid, H1_med, color="crimson", lw=2, label=rf"Model I ($\xi={bf1[2]:.3f}$)")
        ax_c.fill_between(z_grid, H1_lo, H1_hi, color="crimson", alpha=0.15)
    
    if len(H2_samples) > 0:
        ax_c.plot(z_grid, H2_med, color="navy", lw=2, linestyle="-.", label=rf"Model II ($\lambda={bf2[2]:.3f}$)")
        ax_c.fill_between(z_grid, H2_lo, H2_hi, color="navy", alpha=0.15)

    ax_c.set_xlabel(r"Redshift $z$", fontsize=11)
    ax_c.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]", fontsize=11)
    ax_c.set_title(r"Other: Full Horndeski Models vs. $\Lambda$CDM", fontsize=12)
    ax_c.legend(frameon=False)
    fig_comp.tight_layout()
    fig_comp.savefig(os.path.join(out_comp, "horndeski_full_comparison.png"), dpi=250, bbox_inches="tight")
    plt.close(fig_comp)

    # --- STABILITY ANALYSIS PLOTS ---
    out_stab = os.path.join(HERE, "outputs_horndeski_other_stability")
    os.makedirs(out_stab, exist_ok=True)

    z_stab = np.linspace(0, 2.0, 100)
    
    H_1 = H_m1(z_stab, bf1)
    phi_1 = 0.1 * np.log(1 + z_stab)
    dphi_1 = 0.1 / (1 + z_stab)
    params1 = {'V0': 1.0, 'xi': bf1[2], 'Omega_m0': bf1[1], 'H0': bf1[0]}
    cS2_1, QS_1, cT2_1, _ = compute_stability('Model_I', params1, z_stab, phi_1, dphi_1, H_1)

    H_2 = H_m2(z_stab, bf2)
    phi_2 = 0.1 * np.log(1 + z_stab)
    dphi_2 = 0.1 / (1 + z_stab)
    params2 = {'V0': 1.0, 'lambda': bf2[2], 'Omega_m0': bf2[1], 'H0': bf2[0]}

    if H_2 is not None:
     cS2_2, QS_2, cT2_2, _ = compute_stability('Model_II', params2, z_stab, phi_2, dphi_2, H_2)
    else:
     print("Warning: Model II solution returned None over the requested redshift range.")
     cS2_2 = QS_2 = cT2_2 = np.full_like(z_stab, np.nan)
    
    fig_stab, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Model I Stability
    axes[0,0].plot(z_stab, cS2_1, 'r-', label=r'$c_S^2$')
    axes[0,0].plot(z_stab, QS_1, 'b--', label=r'$Q_S$')
    axes[0,0].set_xlabel('Redshift z')
    axes[0,0].set_ylabel('Stability parameters')
    axes[0,0].set_title('Model I: Stability')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Model I Gravitational Wave Speed
    axes[0,1].plot(z_stab, cT2_1, 'r-', label=r'$c_T^2$')
    axes[0,1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label=r'GR ($c_T^2 = 1$)')
    axes[0,1].set_xlabel('Redshift z')
    axes[0,1].set_ylabel(r'$c_T^2$')
    axes[0,1].set_title('Model I: Gravitational Wave Speed')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Model II Stability
    axes[1,0].plot(z_stab, cS2_2, 'r-', label=r'$c_S^2$')
    axes[1,0].plot(z_stab, QS_2, 'b--', label=r'$Q_S$')
    axes[1,0].set_xlabel('Redshift z')
    axes[1,0].set_ylabel('Stability parameters')
    axes[1,0].set_title('Model II: Stability')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Model II Gravitational Wave Speed
    axes[1,1].plot(z_stab, cT2_2, 'r-', label=r'$c_T^2$')
    axes[1,1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label=r'GR ($c_T^2 = 1$)')
    axes[1,1].set_xlabel('Redshift z')
    axes[1,1].set_ylabel(r'$c_T^2$')
    axes[1,1].set_title('Model II: Gravitational Wave Speed')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    fig_stab.tight_layout()
    fig_stab.savefig(os.path.join(out_stab, "stability_analysis.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_stab)

    print("\nPipeline finished successfully.")