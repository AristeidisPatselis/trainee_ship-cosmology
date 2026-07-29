"""
H_dot_lcdm_fit.py
=================
Modified Friedmann equation fit:

    H(z) = H0 * sqrt( Om*(1+z)^3 - a*H*(1+z)*dH_dz )
"""
import numpy as np
import scipy.optimize as opt
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib import rc
import emcee                              
import corner                             
from scipy.stats import norm 
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import os


# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================

def setup_matplotlib():
    """Enable LaTeX only if a real render actually succeeds on this machine.

    Why: `rc('text', usetex=True)` alone doesn't fail immediately if no LaTeX
    installation exists -- it only errors later, mid-plot, which is annoying
    when generating 6 figures in a row. So we do a throwaway test render here
    and silently fall back to matplotlib's built-in mathtext if it fails.
    """
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
        fig_test = plt.figure()
        plt.text(0.5, 0.5, r"$\alpha$")
        fig_test.canvas.draw()   # forces the actual LaTeX call to happen now, not later
        plt.close(fig_test)
    except Exception as e:
        print(f"Note: LaTeX rendering unavailable, using mathtext instead. ({e})")
        rc('text', usetex=False)
        rc('font', family='DejaVu Sans')


def load_clean_data(filename, script_dir):
    """Loads numeric data one value per line, tolerating stray bracket
    artifacts (e.g. leftover '...]' text) by keeping only what follows the
    last ']' on each line. Falls back gracefully for plain numeric files.

    This exists because the raw H(z)/sigma files (as exported from whatever
    upstream tool/notebook produced them) sometimes have a leading fragment
    like "[12] 67.3" per line -- e.g. copy-pasted output with array indices
    still attached. Splitting on ']' and taking the last piece strips that
    off; if there's no ']' at all, split(']')[-1] just returns the whole
    line unchanged, so plain numeric files still work fine.
    """
    filepath = os.path.join(script_dir, filename)
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.split(']')[-1].strip()
            if clean_line:          # skip blank lines
                data.append(float(clean_line))
    return np.array(data)


def load_all_data():
    """Loads z, H(z), and sigma_H arrays from disk.

    Assumes the three files are already aligned row-by-row (z_vals.txt[i],
    H_vals.txt[i], sigma_vals.txt[i] all refer to the same cosmic-chronometer
    data point i) -- no re-sorting or matching is done here.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    z_vals = load_clean_data('z_vals.txt', script_dir)
    H_vals = load_clean_data('H_vals.txt', script_dir)
    sigma_vals = load_clean_data('sigma_vals.txt', script_dir)
    return z_vals, H_vals, sigma_vals

def dHdz(z, H, H0, Omega0, alpha):
    x = 1 + z
    dH = (Omega0 * x**3 - H[0]**2 / H0**2) / (alpha * H[0] * x)
    return np.array([dH])

def model_H(z_eval, H0, Omega0, alpha):
    """Solve ODE, return H(z) at requested redshifts."""
    z_sorted = np.sort(z_eval)
    sol = solve_ivp(
        dHdz, (0, z_sorted.max()), [H0],
        args=(H0, Omega0, alpha),
        t_eval=z_sorted,
        method='Radau',        # robust choice; try RK45 too
        rtol=1e-8, atol=1e-10
    )
    if not sol.success:
        return None
    return np.interp(z_eval, sol.t, sol.y[0])

def chi2(params, z_vals, H_vals, sigma_vals):
    H0, Omega0, alpha = params
    if H0 <= 0 or Omega0 <= 0 or alpha == 0:
        return 1e10
    H_model = model_H(z_vals, H0, Omega0, alpha)
    if H_model is None or np.any(~np.isfinite(H_model)):
        return 1e10
    return np.sum(((H_vals - H_model) / sigma_vals)**2)


def chi2_min(x0=None):
    """Run the 3-parameter (H0, Omega0, alpha) best fit once and return
    everything the downstream plotting functions need.
    """
    z_vals, H_vals, sigma_vals = load_all_data()
    if x0 is None:
        x0 = [70.0, 0.3, 1.0]
    result = minimize(chi2, x0, args=(z_vals, H_vals, sigma_vals), method='Nelder-Mead')
    H0_fit, Omega0_fit, alpha_fit = result.x
    chi2_best = result.fun
    return H0_fit, Omega0_fit, alpha_fit, chi2_best, z_vals, H_vals, sigma_vals


def plot_chi2_profile_alpha(alpha_range=None, n_points=60):
    """1D profile: chi^2(alpha), refitting (H0, Omega0) at each alpha so this
    is a true *profile* likelihood, not a slice through the best-fit point.
    Marks 1sigma/2sigma/3sigma thresholds for 1 dof (Delta chi^2 = 1, 4, 9).
    """
    H0_fit, Omega0_fit, alpha_fit, chi2_best, z_vals, H_vals, sigma_vals = chi2_min()

    if alpha_range is None:
        alpha_range = (alpha_fit * 0.3, alpha_fit * 2.0)
    alphas = np.linspace(*alpha_range, n_points)

    chi2_vals = np.empty(n_points)
    for i, a in enumerate(alphas):
        # profile out H0, Omega0 at this fixed alpha
        def chi2_reduced(p2):
            H0, Om = p2
            return chi2([H0, Om, a], z_vals, H_vals, sigma_vals)
        res = minimize(chi2_reduced, [H0_fit, Omega0_fit], method='Nelder-Mead')
        chi2_vals[i] = res.fun

    delta_chi2 = chi2_vals - chi2_best

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, delta_chi2, color='navy', lw=2)
    ax.axvline(alpha_fit, color='gray', ls=':', lw=1)
    for level, label in [(1, r'1$\sigma$'), (4, r'2$\sigma$'), (9, r'3$\sigma$')]:
        ax.axhline(level, color='gray', ls='--', lw=0.8)
        ax.text(alphas[-1], level, label, va='bottom', ha='right', fontsize=9, color='gray')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\Delta\chi^2(\alpha)$')
    ax.set_title(r'Profile likelihood: $\Delta\chi^2$ vs $\alpha$')
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig('chi2_profile_alpha.png', dpi=150)
    plt.close(fig)
    return alphas, chi2_vals, chi2_best


def plot_contour_alpha_omega0(n_grid=40, fix_H0=True):
    """2D contour: alpha vs Omega0, chi^2 as the contoured quantity.
    If fix_H0=True, H0 is held at its best-fit value (fast, standard first pass).
    If fix_H0=False, H0 is profiled out at every grid point (slower, more rigorous).
    """
    H0_fit, Omega0_fit, alpha_fit, chi2_best, z_vals, H_vals, sigma_vals = chi2_min()

    alpha_grid = np.linspace(alpha_fit * 0.3, alpha_fit * 2.0, n_grid)
    omega0_grid = np.linspace(max(0.05, Omega0_fit * 0.5), Omega0_fit * 1.5, n_grid)
    A, O = np.meshgrid(alpha_grid, omega0_grid)
    CHI2 = np.empty_like(A)

    for i in range(n_grid):
        for j in range(n_grid):
            a, om = A[i, j], O[i, j]
            if fix_H0:
                CHI2[i, j] = chi2([H0_fit, om, a], z_vals, H_vals, sigma_vals)
            else:
                def chi2_H0(H0_arr):
                    return chi2([H0_arr[0], om, a], z_vals, H_vals, sigma_vals)
                res = minimize(chi2_H0, [H0_fit], method='Nelder-Mead')
                CHI2[i, j] = res.fun

    delta_chi2 = CHI2 - chi2_best  # note: chi2_best from the *full* 3-param fit

    fig, ax = plt.subplots(figsize=(7, 6))
    levels = [2.30, 6.18, 11.83]  # 1sigma, 2sigma, 3sigma for 2 dof
    cs = ax.contour(A, O, delta_chi2, levels=levels, colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'})
    ax.contourf(A, O, delta_chi2, levels=[0, *levels, delta_chi2.max()],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    ax.plot(alpha_fit, Omega0_fit, 'k*', ms=14, label='best fit')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\Omega_0$')
    ax.set_title(r'$\Delta\chi^2$ confidence contours: $\alpha$ vs $\Omega_0$'
                 + ('  (H0 fixed)' if fix_H0 else '  (H0 profiled)'))
    ax.legend()
    fig.tight_layout()
    fig.savefig('contour_alpha_omega0.png', dpi=150)
    plt.close(fig)
    return A, O, delta_chi2

def main():
    setup_matplotlib()

    H0_fit, Omega0_fit, alpha_fit, chi2_best, z_vals, H_vals, sigma_vals = chi2_min()
    print(f"H0={H0_fit:.3f}, Omega0={Omega0_fit:.4f}, alpha={alpha_fit:.4f}, chi2={chi2_best:.2f}")

    plot_chi2_profile_alpha()
    plot_contour_alpha_omega0(fix_H0=True)
    plot_contour_alpha_omega0()

if __name__ == "__main__":
    main()