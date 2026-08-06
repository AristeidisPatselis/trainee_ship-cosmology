import os
import numpy as np
import scipy.optimize as opt
from scipy.optimize import brentq
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import emcee
import corner

# =============================================================================
# 1. SETUP
# =============================================================================

def setup_matplotlib():
    """Enables LaTeX formatting for professional plots, but only if a real
    LaTeX render actually succeeds on this machine (rc() alone doesn't fail
    even when the LaTeX installation is broken/incomplete)."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
        fig_test = plt.figure()
        plt.text(0.5, 0.5, r"$\alpha$")
        fig_test.canvas.draw()
        plt.close(fig_test)
    except Exception as e:
        print(f"Note: LaTeX rendering unavailable, using standard mathtext fonts instead. ({e})")
        rc('text', usetex=False)
        rc('font', family='DejaVu Sans')


def load_data():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    z = np.loadtxt(os.path.join(script_dir, "z_vals.txt"))
    H = np.loadtxt(os.path.join(script_dir, "H_vals.txt"))
    sigma = np.loadtxt(os.path.join(script_dir, "sigma_vals.txt"))
    return z, H, sigma


# =============================================================================
# 2. MODIFIED FRIEDMANN EQUATION
#
#   H(z) = H0 * sqrt( Om*(1+z)^3 + (1-Om)*(H(z)/H0)^delta )
#
# delta = 0 reproduces standard flat LCDM exactly (the (1-Om)(H/H0)^0 term
# becomes the ordinary constant dark-energy density parameter). delta is
# therefore a direct measure of how much the data prefer a dark-energy
# component whose effective density evolves with H away from a pure
# cosmological constant. H appears on both sides, so for every trial
# (Om, H0, delta) each H(z_i) must be solved for implicitly.
# =============================================================================

def H_single(z, Om, H0, delta):
    """
    Solve H - H0*sqrt(Om*(1+z)^3 + (1-Om)*(H/H0)^delta) = 0 for a single z.
    Uses bracketed root-finding (brentq), which is more robust than a
    derivative-based solver here because the bracket is expanded
    automatically until a sign change is found.
    """
    def eq(H):
        inside = Om * (1 + z)**3 + (1 - Om) * (H / H0)**delta
        if inside <= 0 or not np.isfinite(inside):
            return 1e10
        return H - H0 * np.sqrt(inside)

    # LCDM value as the starting bracket center
    guess = H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))
    lo, hi = 0.05 * guess, 20 * guess

    try:
        flo, fhi = eq(lo), eq(hi)
        tries = 0
        while flo * fhi > 0 and tries < 40:
            lo *= 0.7
            hi *= 1.4
            flo, fhi = eq(lo), eq(hi)
            tries += 1
        if flo * fhi > 0:
            return np.nan
        return brentq(eq, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200)
    except Exception:
        return np.nan


def H_model(z_array, Om, H0, delta):
    """Vectorized wrapper: solves H_single for every z in z_array."""
    z_array = np.atleast_1d(z_array)
    return np.array([H_single(z, Om, H0, delta) for z in z_array])


# =============================================================================
# 3. CHI-SQUARED
# =============================================================================

def chi2(params, z_vals, H_vals, sigma_vals):
    Om, H0, delta = params
    if not (0.01 < Om < 0.99) or H0 <= 0:
        return np.inf
    H_theory = H_model(z_vals, Om, H0, delta)
    if np.any(np.isnan(H_theory)):
        return np.inf
    return np.sum(((H_vals - H_theory) / sigma_vals)**2)


# =============================================================================
# 4. THE NAIVE "SINGLE-POINT" METHOD (as in idk2.py), kept here only so it
#    can be printed alongside the correct result for comparison.
# =============================================================================

def naive_mean_delta(z_vals, H_vals, sigma_vals, Om0=0.3, H0_fixed=70.0):
    weights = 1.0 / sigma_vals**2
    H_mean = np.sum(weights * H_vals) / np.sum(weights)
    sigma_H_mean = np.sqrt(1.0 / np.sum(weights))
    z_mean = np.mean(z_vals)

    def delta_of_H(H):
        return np.log(
            ((H / H0_fixed)**2 - Om0 * (1 + z_mean)**3) / (1 - Om0)
        ) / np.log(H / H0_fixed)

    delta = delta_of_H(H_mean)
    eps = 1e-5
    d_delta_dH = (delta_of_H(H_mean + eps) - delta_of_H(H_mean - eps)) / (2 * eps)
    sigma_delta = abs(d_delta_dH) * sigma_H_mean
    return delta, sigma_delta


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    setup_matplotlib()
    z_vals, H_vals, sigma_vals = load_data()
    print(f"Loaded {len(z_vals)} H(z) data points.\n")

    # -------------------------------------------------------------------
    # Method 1 (for comparison only): naive single-point analytic estimate
    # -------------------------------------------------------------------
    naive_delta, naive_err = naive_mean_delta(z_vals, H_vals, sigma_vals)
    print("--- Method 1: naive mean-point analytic estimate (fixed Om, H0) ---")
    print(f"delta = {naive_delta:.4f} +/- {naive_err:.4f}\n")

    # -------------------------------------------------------------------
    # Method 2: frequentist joint chi-squared minimization over (Om,H0,delta)
    # -------------------------------------------------------------------
    print("--- Method 2: joint chi-squared minimization (Om, H0, delta) ---")

    def neg_obj(p):
        return chi2(p, z_vals, H_vals, sigma_vals)

    bounds = [(0.01, 0.99), (40, 100), (-3, 3)]
    starts = [
        [0.30, 70.0, 0.0],
        [0.25, 68.0, -1.0],
        [0.35, 72.0, 1.0],
    ]
    best = None
    for s0 in starts:
        r = opt.minimize(neg_obj, s0, bounds=bounds, method="Nelder-Mead",
                          options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 20000,
                                   "maxfev": 20000})
        if best is None or r.fun < best.fun:
            best = r

    best_Om, best_H0, best_delta = best.x
    dof = len(z_vals) - 3
    print(f"Omega_m0 = {best_Om:.5f}")
    print(f"H0       = {best_H0:.5f}")
    print(f"delta    = {best_delta:.5f}")
    print(f"chi2     = {best.fun:.4f}   (dof = {dof}, chi2/dof = {best.fun/dof:.3f})\n")

    # -------------------------------------------------------------------
    # Method 3: Bayesian MCMC posterior (the recommended method)
    # -------------------------------------------------------------------
    print("--- Method 3: MCMC posterior sampling (recommended) ---")

    def log_prior(theta):
        Om, H0, delta = theta
        if 0.01 < Om < 0.99 and 40 < H0 < 100 and -3 < delta < 3:
            return 0.0
        return -np.inf

    def log_likelihood(theta):
        Om, H0, delta = theta
        H_theory = H_model(z_vals, Om, H0, delta)
        if np.any(np.isnan(H_theory)):
            return -np.inf
        return -0.5 * np.sum(((H_vals - H_theory) / sigma_vals)**2)

    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    ndim, nwalkers, nsteps = 3, 32, 4000
    initial = np.array([best_Om, best_H0, best_delta])
    pos = initial + np.array([0.003, 0.3, 0.03]) * np.random.randn(nwalkers, ndim)
    # clip to bounds so no walker starts outside the prior
    for i, (lo, hi) in enumerate(bounds):
        pos[:, i] = np.clip(pos[:, i], lo + 1e-4, hi - 1e-4)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

    Om_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    H0_mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])
    delta_mcmc = np.percentile(flat_samples[:, 2], [16, 50, 84])

    print(f"Omega_m0 = {Om_mcmc[1]:.5f} (+{Om_mcmc[2]-Om_mcmc[1]:.5f} / -{Om_mcmc[1]-Om_mcmc[0]:.5f})")
    print(f"H0       = {H0_mcmc[1]:.4f} (+{H0_mcmc[2]-H0_mcmc[1]:.4f} / -{H0_mcmc[1]-H0_mcmc[0]:.4f})")
    print(f"delta    = {delta_mcmc[1]:.4f} (+{delta_mcmc[2]-delta_mcmc[1]:.4f} / -{delta_mcmc[1]-delta_mcmc[0]:.4f})")
    print(f"\n[delta = 0 is the standard LCDM limit -> "
          f"data pull delta away from 0 at the "
          f"{abs(delta_mcmc[1])/max(delta_mcmc[2]-delta_mcmc[1], delta_mcmc[1]-delta_mcmc[0]):.1f} sigma level]\n")

    # -------------------------------------------------------------------
    # Summary comparison table
    # -------------------------------------------------------------------
    print("=" * 60)
    print("SUMMARY: delta from the three methods")
    print("=" * 60)
    print(f"{'Method':45s} delta")
    print(f"{'1. Naive single mean-point (fixed Om,H0)':45s} {naive_delta:+.3f} +/- {naive_err:.3f}")
    print(f"{'2. Joint chi2 minimization (best fit)':45s} {best_delta:+.3f}  (point estimate only)")
    print(f"{'3. MCMC posterior (Om,H0,delta jointly fit)':45s} "
          f"{delta_mcmc[1]:+.3f} (+{delta_mcmc[2]-delta_mcmc[1]:.3f}/-{delta_mcmc[1]-delta_mcmc[0]:.3f})")
    print("=" * 60 + "\n")

    # =====================================================================
    # PLOT 1: Delta chi-squared contour in the (Omega_m, delta) plane
    #         (H0 fixed at its best-fit value -- a profile slice through
    #          the 3D chi2 surface, shown here because it's fast and
    #          intuitive; the MCMC corner plot below gives the fully
    #          marginalized, statistically correct version.)
    # =====================================================================
    print("--- Generating chi2 contour map in (Omega_m0, delta) ---")
    n_grid = 150
    Om_space = np.linspace(0.05, 0.95, n_grid)
    delta_space = np.linspace(-3, 3, n_grid)
    OM, DE = np.meshgrid(Om_space, delta_space)

    Z = np.empty_like(OM)
    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = chi2((OM[i, j], best_H0, DE[i, j]), z_vals, H_vals, sigma_vals)

    delta_chisq = Z - best.fun
    levels = [2.30, 6.18, 11.83]  # 1,2,3-sigma for 2 dof

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(OM, DE, delta_chisq, levels=[0] + levels, cmap="viridis_r", extend="max")
    cs = ax.contour(OM, DE, delta_chisq, levels=levels, colors="white", linewidths=1)
    ax.clabel(cs, inline=True, fontsize=9,
              fmt={levels[0]: r"$1\sigma$", levels[1]: r"$2\sigma$", levels[2]: r"$3\sigma$"})
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.2, label=r"$\Lambda$CDM ($\delta=0$)")
    ax.plot(best_Om, best_delta, "r*", markersize=15, label="Best fit")
    ax.set_xlabel(r"$\Omega_{m,0}$")
    ax.set_ylabel(r"$\delta$")
    ax.set_title(r"$\Delta\chi^2$ confidence contours ($H_0$ fixed at best fit)")
    ax.legend()
    fig.colorbar(cf, ax=ax, label=r"$\Delta\chi^2$")
    fig.tight_layout()
    fig.savefig("attemt3_contour_Om_delta.png", dpi=150)
    plt.show()

    # =====================================================================
    # PLOT 2: Delta chi-squared contour in the (H0, delta) plane
    #         (Omega_m fixed at its best-fit value)
    # =====================================================================
    print("--- Generating chi2 contour map in (H0, delta) ---")
    H0_space = np.linspace(55, 85, n_grid)
    HH, DE2 = np.meshgrid(H0_space, delta_space)

    Z2 = np.empty_like(HH)
    for i in range(n_grid):
        for j in range(n_grid):
            Z2[i, j] = chi2((best_Om, HH[i, j], DE2[i, j]), z_vals, H_vals, sigma_vals)

    delta_chisq2 = Z2 - best.fun

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    cf2 = ax2.contourf(HH, DE2, delta_chisq2, levels=[0] + levels, cmap="viridis_r", extend="max")
    cs2 = ax2.contour(HH, DE2, delta_chisq2, levels=levels, colors="white", linewidths=1)
    ax2.clabel(cs2, inline=True, fontsize=9,
               fmt={levels[0]: r"$1\sigma$", levels[1]: r"$2\sigma$", levels[2]: r"$3\sigma$"})
    ax2.axhline(0, color="crimson", linestyle="--", linewidth=1.2, label=r"$\Lambda$CDM ($\delta=0$)")
    ax2.plot(best_H0, best_delta, "r*", markersize=15, label="Best fit")
    ax2.set_xlabel(r"$H_0$")
    ax2.set_ylabel(r"$\delta$")
    ax2.set_title(r"$\Delta\chi^2$ confidence contours ($\Omega_{m,0}$ fixed at best fit)")
    ax2.legend()
    fig2.colorbar(cf2, ax=ax2, label=r"$\Delta\chi^2$")
    fig2.tight_layout()
    fig2.savefig("attemt3_contour_H0_delta.png", dpi=150)
    plt.show()

    # =====================================================================
    # PLOT 3: MCMC corner plot -- the statistically correct, fully
    #         marginalized joint and 1D posterior for delta.
    # =====================================================================
    print("--- Generating MCMC corner plot ---")
    fig3 = corner.corner(
        flat_samples,
        labels=[r"$\Omega_{m,0}$", r"$H_0$", r"$\delta$"],
        truths=[best_Om, best_H0, best_delta],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig3.savefig("attemt3_corner_Om_H0_delta.png", dpi=150)
    plt.show()

    print("\nSaved figures: contour_Om_delta.png, contour_H0_delta.png, corner_Om_H0_delta.png")


if __name__ == "__main__":
    main()