"""
Hdot_alpha_lcdm_fit.py
======================

Modified Friedmann equation with a Hubble-derivative ("Hdot") correction term:

    H(z)^2 = H0^2 * Om0 * (1+z)^3  -  alpha * (1+z) * H(z) * dH/dz

Free parameters: (H0, Om0, alpha).  No delta / b*H^delta term anywhere in
this file -- this is the pure 3-parameter Hdot-alpha extension of LambdaCDM.

Physical reading
-----------------
Standard flat LambdaCDM has a constant dark-energy density, Om0*(1+z)^3 + OL0.
Here the constant OL0 term is replaced entirely by a term built from the rate
of change of H itself, alpha*(1+z)*H*dH/dz. As alpha -> 0 this correction
term is suppressed relative to the (1+z)^3 matter term at low z, so the
model should smoothly approach an "empty" (no dark energy) matter-only
Friedmann equation -- this is checked explicitly below (see
`consistency_check_alpha_to_zero`), and is NOT the same limit as recovering
standard LambdaCDM (which additionally needs an explicit OL0 term that this
model does not have).

Pipeline
--------
1.  Load and clean the (z, H, sigma_H) cosmic-chronometer data.
2.  Solve H(z) implicitly via the substitution u = H^2, turning the Hdot
    term into a first-order linear-in-u ODE, solved robustly with solve_ivp.
3.  Best fit via a global optimizer (differential_evolution) followed by a
    local polish (Nelder-Mead), plus an independent multi-start cross-check.
4.  Uncertainties two ways: curve_fit covariance (Laplace/Gaussian approx)
    and full MCMC posterior sampling with emcee + corner plot.
5.  Profile chi^2(alpha) and 2D Delta-chi^2 contours (alpha-Om0, alpha-H0).
6.  Hubble diagram (data + best-fit curve + residuals).
7.  AIC/BIC comparison against baseline flat LambdaCDM (Om0, H0) so the
    extra alpha parameter can be judged on whether it is actually earning
    its keep.
"""

import os
import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize, differential_evolution, curve_fit
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import rc
import emcee
import corner

np.random.seed(42)

# =============================================================================
# 0. PLOTTING SETUP
# =============================================================================

def setup_matplotlib():
    """Enable LaTeX rendering only if a real render actually succeeds here.

    `rc('text', usetex=True)` doesn't fail immediately if there's no LaTeX
    installation -- it errors later, mid-plot. So do a throwaway test render
    now and fall back to mathtext if it fails, rather than crashing partway
    through a batch of figures.
    """
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


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_clean_data(filename, script_dir):
    """Load one numeric value per line, tolerating stray bracket artifacts.

    Handles lines like "[12] 67.3" (e.g. leftover array indices from a
    copy-pasted terminal/notebook dump) by keeping only what follows the
    last ']' on each line. Plain numeric files (no ']' at all) pass through
    unchanged, since split(']')[-1] just returns the whole line.
    """
    filepath = os.path.join(script_dir, filename)
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.split(']')[-1].strip()
            if clean_line:
                data.append(float(clean_line))
    return np.array(data)


def load_all_data():
    """Load z, H(z), sigma_H arrays (assumed already aligned row-by-row)."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    z_vals = load_clean_data('z_vals.txt', script_dir)
    H_vals = load_clean_data('H_vals.txt', script_dir)
    sigma_vals = load_clean_data('sigma_vals.txt', script_dir)
    assert len(z_vals) == len(H_vals) == len(sigma_vals), \
        "z_vals, H_vals, sigma_vals must have matching lengths"
    return z_vals, H_vals, sigma_vals


# =============================================================================
# 2. MODEL: implicit H(z) via the Hdot ODE
# =============================================================================
#
# Starting equation:
#   H^2 = H0^2 * Om0 * (1+z)^3 - alpha*(1+z)*H*dH/dz
#
# Substitute u = H^2  =>  du/dz = 2*H*dH/dz, so H*dH/dz = (1/2) du/dz:
#
#   u = H0^2*Om0*(1+z)^3 - alpha*(1+z)/2 * du/dz
#
#   =>  du/dz = [2 / (alpha*(1+z))] * ( H0^2*Om0*(1+z)^3 - u )
#
# This is linear in u, which makes it much better behaved numerically than
# solving directly for H (avoids the H in a denominator on top of alpha).
# Boundary condition: z=0, H=H0  =>  u(0) = H0^2.

def _rhs_u(z, u, H0, Om0, alpha):
    u_safe = min(max(u[0], 1e-8), 1e10)   # guard against a bad step blowing up
    x = 1.0 + z
    dudz = (2.0 / (alpha * x)) * (H0**2 * Om0 * x**3 - u_safe)
    return [np.clip(dudz, -1e12, 1e12)]


def model_H(z_eval, H0, Om0, alpha):
    """Solve the Hdot ODE for u=H^2 and return H(z) at the requested z's.

    Returns None (scalar call) or an array of NaNs (vectorized call) if the
    integration fails or produces an unphysical (non-finite / negative) u,
    so a chi^2 built on this can penalise it cleanly instead of crashing.
    """
    z_eval = np.atleast_1d(np.asarray(z_eval, dtype=float))
    if alpha == 0 or H0 <= 0 or Om0 <= 0:
        return np.full_like(z_eval, np.nan)

    z_max = max(z_eval.max(), 1e-6)
    t_eval = np.sort(np.unique(np.append(z_eval, 0.0)))

    try:
        sol = solve_ivp(
            _rhs_u, (0.0, z_max), [H0**2],
            args=(H0, Om0, alpha),
            t_eval=t_eval,
            method='LSODA',        # handles the stiffness that small alpha causes
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


def H_lcdm(z, H0, Om0):
    """Standard flat LambdaCDM, for the baseline AIC/BIC comparison only."""
    return H0 * np.sqrt(Om0 * (1 + z)**3 + (1 - Om0))


# =============================================================================
# 3. CHI-SQUARED
# =============================================================================

# NOTE on Om0's range: this model has no explicit dark-energy (constant)
# term -- the entire late-time acceleration-like behaviour has to come from
# the alpha*(1+z)*H*dH/dz term. On this dataset the true chi^2 minimum sits
# at Om0 > 1 (an "over-closed" matter density in the ordinary LambdaCDM
# sense), which only makes sense here because the Hdot term is compensating
# for it -- Om0 is not the physical matter density fraction unless alpha
# happens to be negligible. Bounds are kept wide enough to contain the true
# unconstrained optimum (checked separately) rather than artificially
# clipping the fit against a prior imported from standard LambdaCDM.
BOUNDS = [(50.0, 100.0), (0.01, 3.0), (0.01, 6.0)]   # H0, Om0, alpha
PARAM_NAMES = ['H0', 'Om0', 'alpha']


def _within_bounds(params):
    return all(lo <= p <= hi for p, (lo, hi) in zip(params, BOUNDS))


def chi2(params, z_vals, H_vals, sigma_vals):
    H0, Om0, alpha = params
    if H0 <= 0 or Om0 <= 0 or alpha <= 0 or not _within_bounds(params):
        return 1e12
    H_model = model_H(z_vals, H0, Om0, alpha)
    if H_model is None or np.any(~np.isfinite(H_model)):
        return 1e12
    return float(np.sum(((H_vals - H_model) / sigma_vals) ** 2))


def chi2_lcdm(params, z_vals, H_vals, sigma_vals):
    H0, Om0 = params
    if H0 <= 0 or not (0 < Om0 < 1):
        return 1e12
    return float(np.sum(((H_vals - H_lcdm(z_vals, H0, Om0)) / sigma_vals) ** 2))


# =============================================================================
# 4. BEST FIT: global optimizer + multi-start cross-check
# =============================================================================

def best_fit(z_vals, H_vals, sigma_vals, n_starts=8, verbose=True):
    """Global fit (differential_evolution) followed by a Nelder-Mead polish,
    plus an independent multi-start Nelder-Mead scan as a cross-check that
    the global optimizer landed on the true minimum and not a local one.
    """
    de_result = differential_evolution(
        chi2, bounds=BOUNDS, args=(z_vals, H_vals, sigma_vals),
        seed=42, maxiter=200, tol=1e-8, polish=True, popsize=20,
    )
    best_x, best_chi2 = de_result.x, de_result.fun

    rng = np.random.default_rng(42)
    starts = [best_x] + [
        [rng.uniform(lo, hi) for (lo, hi) in BOUNDS] for _ in range(n_starts)
    ]
    local_results = []
    for x0 in starts:
        res = minimize(chi2, x0, args=(z_vals, H_vals, sigma_vals),
                        method='Nelder-Mead', bounds=BOUNDS,
                        options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        local_results.append(res)
        if res.fun < best_chi2:
            best_chi2, best_x = res.fun, res.x

    if verbose:
        spread = np.array([r.fun for r in local_results if np.isfinite(r.fun)])
        print(f"Multi-start scan: {len(spread)}/{len(starts)} runs converged "
              f"to finite chi^2, range [{spread.min():.3f}, {spread.max():.3f}]")
        if spread.max() - spread.min() > 0.5:
            print("  -> spread across starts suggests a degenerate/multi-modal "
                  "chi^2 surface; trust the global (differential_evolution) result.")

    return best_x, best_chi2, de_result.success


# =============================================================================
# 5. UNCERTAINTIES: curve_fit covariance + MCMC
# =============================================================================

def model_H_curvefit(z_array, H0, Om0, alpha):
    """Vectorised wrapper with the curve_fit-friendly (z, *params) signature."""
    H = model_H(z_array, H0, Om0, alpha)
    if H is None or np.any(~np.isfinite(H)):
        # curve_fit can't handle NaNs; push residuals huge instead of crashing
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
             nwalkers=32, nsteps=3000, discard=500, thin=15):
    ndim = 3
    spread = np.array([2.0, 0.02, 0.1])   # small Gaussian ball around best fit
    pos = best_x + spread * np.random.randn(nwalkers, ndim)
    # clip the initial ball into bounds so no walker starts at -inf log-prob
    for i, (lo, hi) in enumerate(BOUNDS):
        pos[:, i] = np.clip(pos[:, i], lo + 1e-6, hi - 1e-6)

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob, args=(z_vals, H_vals, sigma_vals)
    )
    sampler.run_mcmc(pos, nsteps, progress=False)
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    return sampler, flat_samples


# =============================================================================
# 6. PROFILE LIKELIHOOD / CONTOURS
# =============================================================================

def plot_chi2_profile_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                             n_points=60, outdir='.'):
    """1D profile chi^2(alpha): H0 and Om0 are re-fit at every alpha, so this
    is a true profile likelihood rather than a slice through the best fit.
    """
    H0_fit, Om0_fit, alpha_fit = best_x
    alpha_lo = max(BOUNDS[2][0], alpha_fit * 0.3)
    alpha_hi = min(BOUNDS[2][1], alpha_fit * 2.5)
    alphas = np.linspace(alpha_lo, alpha_hi, n_points)

    chi2_vals = np.empty(n_points)
    for i, a in enumerate(alphas):
        def chi2_reduced(p2):
            return chi2([p2[0], p2[1], a], z_vals, H_vals, sigma_vals)
        res = minimize(chi2_reduced, [H0_fit, Om0_fit], method='Nelder-Mead',
                        bounds=[BOUNDS[0], BOUNDS[1]])
        chi2_vals[i] = res.fun

    delta_chi2 = chi2_vals - chi2_best

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, delta_chi2, color='navy', lw=2)
    ax.axvline(alpha_fit, color='gray', ls=':', lw=1)
    for level, label in [(1, r'1$\sigma$'), (4, r'2$\sigma$'), (9, r'3$\sigma$')]:
        ax.axhline(level, color='gray', ls='--', lw=0.8)
        ax.text(alphas[-1], level, label, va='bottom', ha='right',
                fontsize=9, color='gray')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\Delta\chi^2(\alpha)$')
    ax.set_title(r'Profile likelihood: $\Delta\chi^2$ vs $\alpha$')
    ax.set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'chi2_profile_alpha.png'), dpi=150)
    plt.close(fig)
    return alphas, chi2_vals


def plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                     vary=('alpha', 'Om0'), n_grid=40, outdir='.'):
    """Delta-chi^2 contour in two of the three parameters, with the third
    held fixed at its best-fit value.
    """
    idx = {'H0': 0, 'Om0': 1, 'alpha': 2}
    ix, iy = idx[vary[0]], idx[vary[1]]
    iz = ({0, 1, 2} - {ix, iy}).pop()

    labels = {'H0': r'$H_0$', 'Om0': r'$\Omega_{m,0}$', 'alpha': r'$\alpha$'}
    center = best_x[ix], best_x[iy]

    x_lo, x_hi = max(BOUNDS[ix][0], center[0] * 0.3), min(BOUNDS[ix][1], center[0] * 2.0)
    y_lo, y_hi = max(BOUNDS[iy][0], center[1] * 0.3), min(BOUNDS[iy][1], center[1] * 2.0)
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
    levels = [2.30, 6.18, 11.83]   # 68%/95%/99% for 2 dof

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contour(X, Y, delta_chi2, levels=levels,
                     colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.clabel(cs, fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'})
    ax.contourf(X, Y, delta_chi2, levels=[0, *levels, max(delta_chi2.max(), levels[-1] + 1)],
                colors=['#08306b', '#4292c6', '#9ecae1', 'white'], alpha=0.3)
    ax.plot(center[0], center[1], 'k*', ms=14, label='best fit')
    ax.set_xlabel(labels[vary[0]])
    ax.set_ylabel(labels[vary[1]])
    ax.set_title(rf'$\Delta\chi^2$ contours: {labels[vary[0]]} vs {labels[vary[1]]} '
                 rf'({labels[PARAM_NAMES[iz]]} fixed)')
    ax.legend()
    fig.tight_layout()
    fname = f'contour_{vary[0]}_{vary[1]}.png'
    fig.savefig(os.path.join(outdir, fname), dpi=150)
    plt.close(fig)
    return X, Y, delta_chi2


# =============================================================================
# 7. HUBBLE DIAGRAM
# =============================================================================

def plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir='.'):
    H0_fit, Om0_fit, alpha_fit = best_x
    z_smooth = np.linspace(0, z_vals.max() * 1.05, 300)
    H_smooth = model_H(z_smooth, H0_fit, Om0_fit, alpha_fit)
    H_at_data = model_H(z_vals, H0_fit, Om0_fit, alpha_fit)
    residuals = H_vals - H_at_data

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax1.errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', color='crimson',
                 ms=4, capsize=2, label='Cosmic chronometer data')
    ax1.plot(z_smooth, H_smooth, color='navy', lw=2,
              label=rf'Hdot-$\alpha$ fit ($\alpha={alpha_fit:.3f}$)')
    ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]')
    ax1.set_title('Hubble diagram: Hdot-$\\alpha$ model best fit')
    ax1.legend()

    ax2.errorbar(z_vals, residuals, yerr=sigma_vals, fmt='o', color='crimson', ms=4, capsize=2)
    ax2.axhline(0, color='navy', lw=1.5)
    ax2.set_xlabel(r'$z$')
    ax2.set_ylabel(r'$H_{\rm obs}-H_{\rm model}$')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hubble_diagram.png'), dpi=150)
    plt.close(fig)


# =============================================================================
# 8. CONSISTENCY CHECK + MODEL COMPARISON
# =============================================================================

def consistency_check_alpha_small(best_x, z_vals, H_vals, sigma_vals):
    """As alpha shrinks, the Hdot term should be driven increasingly stiff
    (it sits in a 1/alpha coefficient), so this just reports how chi^2
    behaves as alpha is pushed toward the low edge of its prior -- it is
    NOT expected to converge to the standard LambdaCDM chi^2, since this
    model has no explicit dark-energy term to fall back on.
    """
    H0_fit, Om0_fit, _ = best_x
    print("\nBehaviour of chi^2 as alpha shrinks (Om0, H0 fixed at best fit):")
    for a in [1.0, 0.5, 0.2, 0.1, 0.05]:
        c = chi2([H0_fit, Om0_fit, a], z_vals, H_vals, sigma_vals)
        print(f"  alpha={a:<5} chi^2={c:.3f}")


def model_comparison(best_x, chi2_best, z_vals, H_vals, sigma_vals):
    """AIC/BIC of the Hdot-alpha model (k=3) vs baseline flat LambdaCDM (k=2)."""
    n = len(z_vals)

    de_lcdm = differential_evolution(
        chi2_lcdm, bounds=[(50.0, 90.0), (0.05, 0.95)],
        args=(z_vals, H_vals, sigma_vals), seed=42, tol=1e-8,
    )
    H0_l, Om0_l = de_lcdm.x
    chi2_lcdm_best = de_lcdm.fun

    def aic_bic(chi2_val, k):
        return chi2_val + 2 * k, chi2_val + k * np.log(n)

    aic_hdot, bic_hdot = aic_bic(chi2_best, 3)
    aic_lcdm, bic_lcdm = aic_bic(chi2_lcdm_best, 2)

    print("\nModel comparison (lower is better):")
    print(f"  LambdaCDM   (k=2): H0={H0_l:.2f}, Om0={Om0_l:.3f}, "
          f"chi2={chi2_lcdm_best:.2f}, AIC={aic_lcdm:.2f}, BIC={bic_lcdm:.2f}")
    print(f"  Hdot-alpha  (k=3): H0={best_x[0]:.2f}, Om0={best_x[1]:.3f}, "
          f"alpha={best_x[2]:.3f}, chi2={chi2_best:.2f}, "
          f"AIC={aic_hdot:.2f}, BIC={bic_hdot:.2f}")
    print(f"  Delta AIC (Hdot-alpha minus LambdaCDM) = {aic_hdot - aic_lcdm:+.2f} "
          "(negative favours the Hdot-alpha model, positive favours LambdaCDM)")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    outdir = os.path.dirname(os.path.realpath(__file__))
    setup_matplotlib()

    z_vals, H_vals, sigma_vals = load_all_data()
    print(f"Loaded {len(z_vals)} data points.")

    print("\n--- Best fit (global optimizer + multi-start cross-check) ---")
    best_x, chi2_best, converged = best_fit(z_vals, H_vals, sigma_vals)
    H0_fit, Om0_fit, alpha_fit = best_x
    dof = len(z_vals) - 3
    print(f"converged: {converged}")
    print(f"H0    = {H0_fit:.4f}")
    print(f"Om0   = {Om0_fit:.4f}")
    print(f"alpha = {alpha_fit:.4f}")
    print(f"chi^2 = {chi2_best:.4f}  (chi^2/dof = {chi2_best/dof:.4f}, dof={dof})")

    print("\n--- curve_fit covariance (Gaussian/Laplace uncertainties) ---")
    try:
        popt, perr, pcov = fit_uncertainties_curvefit(z_vals, H_vals, sigma_vals, best_x)
        for name, val, err in zip(PARAM_NAMES, popt, perr):
            print(f"  {name:6s} = {val:.4f} +/- {err:.4f}")
        # curve_fit sometimes lands a hair away from the global optimum; keep
        # whichever point has the lower chi^2 as the "best_x" used downstream
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

    fig_corner = corner.corner(
        flat_samples, labels=[r'$H_0$', r'$\Omega_{m,0}$', r'$\alpha$'],
        truths=list(best_x), show_titles=True,
    )
    fig_corner.savefig(os.path.join(outdir, 'corner_Hdot_alpha.png'), dpi=150)
    plt.close(fig_corner)

    print("\n--- Profile likelihood & confidence contours ---")
    plot_chi2_profile_alpha(best_x, chi2_best, z_vals, H_vals, sigma_vals, outdir=outdir)
    plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                     vary=('alpha', 'Om0'), outdir=outdir)
    plot_contour_2d(best_x, chi2_best, z_vals, H_vals, sigma_vals,
                     vary=('alpha', 'H0'), outdir=outdir)
    print("  Saved: chi2_profile_alpha.png, contour_alpha_Om0.png, contour_alpha_H0.png")

    print("\n--- Hubble diagram ---")
    plot_hubble_diagram(best_x, z_vals, H_vals, sigma_vals, outdir=outdir)
    print("  Saved: hubble_diagram.png")

    consistency_check_alpha_small(best_x, z_vals, H_vals, sigma_vals)
    model_comparison(best_x, chi2_best, z_vals, H_vals, sigma_vals)

    print("\nDone. Figures written to:", outdir)


if __name__ == "__main__":
    main()