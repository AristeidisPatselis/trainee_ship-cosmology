import os
import numpy as np
import scipy.optimize as opt
from scipy.optimize import root
import matplotlib.pyplot as plt
import emcee
import corner
from matplotlib import rc

# =============================================================================
# 1. SETUP & UTILITIES
# =============================================================================

def setup_matplotlib():
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
    except Exception as e:
        print(f"Warning: LaTeX rendering not enabled. Error: {e}")

def load_clean_data(filename):
    """Safely loads numeric data, stripping prompt artifacts."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(script_dir, filename)
    
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            clean_line = line.split(']')[-1].strip()
            if clean_line:
                data.append(float(clean_line))
    return np.array(data)

# =============================================================================
# 2. COSMOLOGICAL MODEL & STATISTICS
# =============================================================================

def H_model_vectorized(z_array, Om, H0, delta):
    """
    Vectorized implicit solver for the modified Friedmann equation.
    Solves H(z) for all redshifts simultaneously for massive speedup.
    """
    def equations(H_array):
        # E(z)^2 = Om*(1+z)^3 + (1-Om)*E(z)^delta, where E(z) = H/H0
        inside = Om * (1 + z_array)**3 + (1 - Om) * (H_array / H0)**delta
        
        # Penalize unphysical (negative) values inside the square root
        if np.any(inside <= 0):
            return np.full_like(H_array, 1e9)
            
        return H_array - H0 * np.sqrt(inside)

    # Initial guess based on standard Lambda-CDM
    guess = H0 * np.sqrt(Om * (1 + z_array)**3 + (1 - Om))
    
    try:
        sol = root(equations, guess, method='hybr')
        if sol.success:
            return sol.x
        return np.full_like(z_array, np.nan)
    except Exception:
        return np.full_like(z_array, np.nan)

def calc_chi2(params, z_vals, H_vals, sigma_vals):
    """Calculates chi-squared for a given parameter set."""
    Om, H0, delta = params

    # Strict physical and mathematical boundaries
    if not (0.01 < Om < 0.99 and 40 < H0 < 100 and -4 < delta < 4):
        return np.inf

    H_th = H_model_vectorized(z_vals, Om, H0, delta)
    
    if np.any(np.isnan(H_th)):
        return np.inf

    return np.sum(((H_vals - H_th) / sigma_vals)**2)

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

def main():
    setup_matplotlib()

    # --- Step 1: Load Data ---
    print("--- Loading Data ---")
    try:
        z_vals = load_clean_data('z_vals.txt')
        H_vals = load_clean_data('H_vals.txt')
        sigma_vals = load_clean_data('sigma_vals.txt')
        print(f"Successfully loaded {len(z_vals)} data points.\n")
    except FileNotFoundError as e:
        print(f"Critical Error: Data files missing. {e}")
        return

    # --- Step 2: Frequentist Optimization ---
    print("--- Finding Initial Best Fit (L-BFGS-B) ---")
    initial_guess = [0.3, 70.0, 0.0] # Start near Lambda-CDM (delta = 0)
    bounds = [(0.01, 0.99), (50, 90), (-3, 3)]

    result = opt.minimize(
        calc_chi2, initial_guess, args=(z_vals, H_vals, sigma_vals),
        bounds=bounds, method="L-BFGS-B"
    )

    best_Om, best_H0, best_delta = result.x
    min_chisq = result.fun
    dof = len(z_vals) - len(result.x)

    print(f"Omega_m0 = {best_Om:.4f}")
    print(f"H0       = {best_H0:.4f}")
    print(f"delta    = {best_delta:.4f}")
    print(f"chi^2/dof = {min_chisq:.2f}/{dof} = {min_chisq/dof:.3f}\n")

    # --- Step 3: Delta vs Omega_m Contour Plot ---
    print("--- Generating Delta Contour Map ---")
    # We fix H0 to its best fit value to visualize the correlation between Om and delta
    grid_size = 40
    Om_space = np.linspace(0.15, 0.45, grid_size)
    delta_space = np.linspace(-1.5, 1.5, grid_size)
    xx, yy = np.meshgrid(Om_space, delta_space)
    Z = np.zeros_like(xx)

    for i in range(grid_size):
        for j in range(grid_size):
            Z[i, j] = calc_chi2([xx[i, j], best_H0, yy[i, j]], z_vals, H_vals, sigma_vals)

    delta_chisq = Z - min_chisq
    confidence_levels = [2.30, 6.18, 11.83] # 1-sigma, 2-sigma, 3-sigma for 2 DOF

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx, yy, delta_chisq, levels=[0] + confidence_levels, cmap='viridis_r', extend='max')
    cs_lines = ax.contour(xx, yy, delta_chisq, levels=confidence_levels, colors='white', linewidths=1.5)
    
    ax.clabel(cs_lines, inline=True, fontsize=10, fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'})
    ax.plot(best_Om, best_delta, 'r*', markersize=15, label='Best fit')
    
    ax.set_xlabel(r'$\Omega_{m,0}$')
    ax.set_ylabel(r'$\delta$')
    ax.set_title(r'$\Delta\chi^2$ Contours (Fixed $H_0 = {:.2f}$)'.format(best_H0))
    ax.legend()
    fig.colorbar(cf, ax=ax, label=r'$\Delta\chi^2$')
    plt.savefig('attemt1_contours_plot')
    plt.show()

    # --- Step 4: Bayesian MCMC Sampling ---
    print("\n--- Running MCMC Sampling ---")
    def log_probability(theta):
        Om, H0, delta = theta
        if not (0.01 < Om < 0.99 and 40 < H0 < 100 and -4 < delta < 4):
            return -np.inf
        
        H_th = H_model_vectorized(z_vals, Om, H0, delta)
        if np.any(np.isnan(H_th)):
            return -np.inf
            
        return -0.5 * np.sum(((H_vals - H_th) / sigma_vals)**2)

    ndim, nwalkers, nsteps = 3, 50, 3000
    pos = result.x + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([1, 10, 1])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=800, thin=15, flat=True)

    Om_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    H0_mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])
    delta_mcmc = np.percentile(flat_samples[:, 2], [16, 50, 84])

    print("\n--- Final MCMC Constraints ---")
    print(f"Omega_m0 = {Om_mcmc[1]:.4f} (+{Om_mcmc[2]-Om_mcmc[1]:.4f} / -{Om_mcmc[1]-Om_mcmc[0]:.4f})")
    print(f"H0       = {H0_mcmc[1]:.2f} (+{H0_mcmc[2]-H0_mcmc[1]:.2f} / -{H0_mcmc[1]-H0_mcmc[0]:.2f})")
    print(f"delta    = {delta_mcmc[1]:.4f} (+{delta_mcmc[2]-delta_mcmc[1]:.4f} / -{delta_mcmc[1]-delta_mcmc[0]:.4f})\n")

    # --- Step 5: MCMC Corner Plot ---
    fig_corner = corner.corner(
        flat_samples, 
        labels=[r"$\Omega_{m,0}$", r"$H_0$", r"$\delta$"],
        truths=[best_Om, best_H0, best_delta],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    plt.savefig('attemt1_corner_plot.png')
    plt.show()

if __name__ == "__main__":
    main()