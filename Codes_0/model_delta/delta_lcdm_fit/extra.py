"""
test.py
=======
Single-parameter companion to delta_lcdm_fit.py.

delta_lcdm_fit.py jointly fits all three parameters (Om, H0, delta) with
Nelder-Mead + emcee. This script asks a narrower question instead: for
FIXED, fiducial Om and H0 (the standard textbook values, 0.3 and 70
km/s/Mpc, matching the original quick-look version of this script), what
is the best-fit delta alone, and its 1-sigma interval from a simple 1D
chi-squared profile?

Rather than re-implement data loading and the implicit H(z) solver here
(as the old standalone version of this script did), everything is
imported from delta_lcdm_fit.py, so:

  - the data-loading convention (z_vals.txt / H_vals.txt / sigma_vals.txt,
    with stray "]" artifacts stripped) stays in one place, and
  - the implicit equation H = H0*sqrt(Om*(1+z)^3 + (1-Om)*(H/H0)^delta) is
    solved with the robust auto-expanding bracket + delta=0 fast path from
    H_single/H_model, instead of the old fixed 0.1x-10x bracket, which
    could fail to bracket a root (brentq needs a sign change) for large
    |delta| or Om values far from the LCDM guess.

Run this from the same directory as delta_lcdm_fit.py and the three
*_vals.txt data files.
"""

import numpy as np
import matplotlib.pyplot as plt

import delta_lcdm_fit as dlf

# ------------------------------------------------------------------
# Fixed fiducial values for this single-parameter test
# (delta_lcdm_fit.py instead fits these jointly with delta)
# ------------------------------------------------------------------
H0_FIXED = 70.0
OM_FIXED = 0.3


# ------------------------------------------------------------------
# Chi-squared as a function of delta only, Om and H0 held fixed.
# Reuses dlf.H_model (robust auto-expanding-bracket solver).
# ------------------------------------------------------------------
def chi2_delta(delta, z_data, H_data, sigma_H, H0, Om):
    H_theory = dlf.H_model(z_data, Om, H0, delta)
    if np.any(np.isnan(H_theory)) or np.any(H_theory <= 0):
        return np.inf
    return np.sum(((H_data - H_theory) / sigma_H) ** 2)


def find_crossing(delta_grid, above_min, mask, sign):
    """Linear-interpolate where chi2_grid - min_chi2 - 1 crosses zero,
    restricted to `mask` (left or right of the best-fit delta).

    sign = -1 picks the crossing closest to best_delta on the left branch
    (the last sign change walking left-to-right); sign = +1 picks the
    closest crossing on the right branch (the first sign change).
    """
    d_sub = delta_grid[mask]
    a_sub = above_min[mask]
    if len(d_sub) < 2:
        return None
    sign_changes = np.where(np.diff(np.sign(a_sub)) != 0)[0]
    if len(sign_changes) == 0:
        return None
    i = sign_changes[-1] if sign == -1 else sign_changes[0]
    x0, x1 = d_sub[i], d_sub[i + 1]
    y0, y1 = a_sub[i], a_sub[i + 1]
    return x0 - y0 * (x1 - x0) / (y1 - y0)


def main():
    dlf.setup_matplotlib()

    # ------------------------------------------------------------------
    # 1. Load data (shared loader from delta_lcdm_fit.py)
    # ------------------------------------------------------------------
    z_vals, H_vals, sigma_vals = dlf.load_all_data()
    print(f"Loaded {len(z_vals)} H(z) data points.")
    print(f"Fixed fiducial values: H0 = {H0_FIXED}, Om = {OM_FIXED}\n")

    # ------------------------------------------------------------------
    # 2. Best-fit delta (Om, H0 fixed)
    # ------------------------------------------------------------------
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        chi2_delta, bounds=(-5, 5), method="bounded",
        args=(z_vals, H_vals, sigma_vals, H0_FIXED, OM_FIXED),
    )
    best_delta = result.x
    min_chi2 = result.fun

    print(f"Best-fit delta = {best_delta:.4f}")
    print(f"Minimum chi2   = {min_chi2:.4f}")

    # ------------------------------------------------------------------
    # 3. Grid scan for Delta chi^2 = 1 uncertainty (1-parameter 1-sigma)
    # ------------------------------------------------------------------
    delta_grid = np.linspace(best_delta - 2, best_delta + 2, 200)
    chi2_grid = np.array([
        chi2_delta(d, z_vals, H_vals, sigma_vals, H0_FIXED, OM_FIXED)
        for d in delta_grid
    ])

    above_min = chi2_grid - min_chi2 - 1.0
    left_mask = delta_grid < best_delta
    right_mask = delta_grid > best_delta

    delta_minus = find_crossing(delta_grid, above_min, left_mask, sign=-1)
    delta_plus = find_crossing(delta_grid, above_min, right_mask, sign=+1)

    if delta_minus is None or delta_plus is None:
        print("Warning: could not bracket a Delta-chi^2=1 crossing on the "
              "grid; try widening delta_grid.")
    else:
        print(f"1-sigma interval: [{delta_minus:.4f}, {delta_plus:.4f}]")

    # ------------------------------------------------------------------
    # 4. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Data vs best-fit model
    z_fine = np.linspace(min(z_vals), max(z_vals), 200)
    H_fine = dlf.H_model(z_fine, OM_FIXED, H0_FIXED, best_delta)

    axes[0].errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', label='Data')
    axes[0].plot(z_fine, H_fine, 'r-', label=f'Best fit (delta={best_delta:.3f})')
    axes[0].set_xlabel('z')
    axes[0].set_ylabel('H(z)')
    axes[0].legend()

    # Chi2 profile
    axes[1].plot(delta_grid, chi2_grid - min_chi2)
    axes[1].axhline(1.0, color='gray', linestyle='--', label=r'$\Delta\chi^2=1$')
    axes[1].axvline(best_delta, color='r', linestyle='--')
    axes[1].set_xlabel('delta')
    axes[1].set_ylabel(r'$\Delta\chi^2$')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig("delta_only_profile.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()