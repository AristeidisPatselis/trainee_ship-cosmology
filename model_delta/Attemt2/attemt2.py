import os
import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize, root
import emcee
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import corner
from matplotlib import rc
from scipy.ndimage import gaussian_filter
from scipy.optimize import brentq

# ============================================================
# Setup and data loading
# ============================================================

def setup_matplotlib():
    """Setup plotting style."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
    except:
        print("LaTeX not available, using standard fonts")

def load_data(filename):
    """Load data from file."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(script_dir, filename)
    return np.loadtxt(filepath)

# Load data
z_vals = load_data("z_vals.txt")
H_vals = load_data("H_vals.txt")
sigma_vals = load_data("sigma_vals.txt")

print(f"Loaded {len(z_vals)} data points")
print(f"Redshift range: {z_vals.min():.3f} to {z_vals.max():.3f}")
print(f"Hubble range: {H_vals.min():.1f} to {H_vals.max():.1f} km/s/Mpc")

# ============================================================
# Modified Friedmann equation (implicit)
# H/H0 = sqrt( Omega_m(1+z)^3 + (1-Omega_m)(H/H0)^delta )
# ============================================================

def H_single(z, Om, H0, delta):
    """
    Solve the implicit Friedmann equation for a single redshift.
    Uses an efficient numerical solver.
    """
    def equation(H):
        x = H / H0
        inside = Om * (1 + z)**3 + (1 - Om) * x**delta
        if inside <= 0:
            return 1e10 * (H - H0)  # Penalty for invalid
        return H - H0 * np.sqrt(inside)
    
    # Initial guess: ΛCDM solution
    guess = H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))
    
    try:
        # Bracket the root
        H_low = H0 * 0.01
        H_high = H0 * 10.0
        
        # Check if signs differ
        f_low = equation(H_low)
        f_high = equation(H_high)
        
        if f_low * f_high < 0:
            # Use Brent's method
            try:
                return brentq(equation, H_low, H_high)
            except:
                pass
        
        # Fallback to root solver
        sol = root(equation, guess, method='hybr')
        if sol.success:
            return sol.x[0]
        return np.nan
            
    except:
        return np.nan

def H_model(z_array, Om, H0, delta):
    """
    Calculate theoretical H(z) for all redshifts.
    """
    results = []
    for z in z_array:
        H_val = H_single(z, Om, H0, delta)
        results.append(H_val)
    return np.array(results)

# ============================================================
# Efficient approximation for initial parameter estimation
# ============================================================

def estimate_delta_from_data(Om0=0.3, H0=70.0):
    """
    Quick estimate of delta using weighted mean approach.
    Useful for initial guess in MCMC.
    """
    weights = 1.0 / sigma_vals**2
    
    # Weighted mean of H
    H_mean = np.sum(weights * H_vals) / np.sum(weights)
    sigma_H_mean = np.sqrt(1.0 / np.sum(weights))
    
    # Weighted mean of z
    z_mean = np.sum(weights * z_vals) / np.sum(weights)
    
    # For a given H0 and Om, solve for delta
    def delta_func(H, Om, H0):
        log_term = np.log(((H/H0)**2 - Om*(1+z_mean)**3) / (1 - Om))
        return log_term / np.log(H/H0)
    
    return delta_func(H_mean, Om0, H0)

# ============================================================
# Chi-square function
# ============================================================

def chi2(params):
    """Chi-square for a given set of parameters."""
    Om, H0, delta = params
    
    # Physical limits
    if Om <= 0.01 or Om >= 0.99:
        return np.inf
    if H0 <= 30 or H0 >= 120:
        return np.inf
    if delta < -5 or delta > 5:
        return np.inf
    
    # Calculate theoretical H values
    H_theory = H_model(z_vals, Om, H0, delta)
    
    # Reject failed solutions
    if np.any(np.isnan(H_theory)) or np.any(H_theory <= 0):
        return np.inf
    
    # Calculate chi-square
    return np.sum(((H_vals - H_theory) / sigma_vals)**2)

# ============================================================
# Parameter estimation with multiple methods
# ============================================================

print("\n" + "="*60)
print("PARAMETER ESTIMATION")
print("="*60)

# Method 1: Estimate delta from mean values
delta_approx = estimate_delta_from_data()
print(f"\nMethod 1 (Approximation): delta = {delta_approx:.4f}")

# Method 2: Optimization (best-fit)
print("\nOptimizing full model...")
initial_guess = [0.3, 70.0, delta_approx]
bounds = [(0.01, 0.99), (30, 120), (-5, 5)]

result = minimize(chi2, initial_guess, bounds=bounds, method='L-BFGS-B')
best_params = result.x
best_Om, best_H0, best_delta = best_params
min_chi2 = result.fun
dof = len(z_vals) - 3  # 3 parameters

print(f"\nMethod 2 (MLE Best-fit):")
print(f"  Omega_m0 = {best_Om:.5f}")
print(f"  H0       = {best_H0:.2f}")
print(f"  delta    = {best_delta:.5f}")
print(f"  chi2     = {min_chi2:.2f}")
print(f"  chi2/dof = {min_chi2/dof:.3f}")

# ============================================================
# MCMC for uncertainty estimation
# ============================================================

print("\nRunning MCMC for error estimation...")

def log_prior(params):
    Om, H0, delta = params
    if 0.01 < Om < 0.99 and 30 < H0 < 120 and -5 < delta < 5:
        return 0.0
    return -np.inf

def log_likelihood(params):
    Om, H0, delta = params
    H_theory = H_model(z_vals, Om, H0, delta)
    if np.any(np.isnan(H_theory)) or np.any(H_theory <= 0):
        return -np.inf
    return -0.5 * np.sum(((H_vals - H_theory) / sigma_vals)**2)

def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# Setup MCMC
ndim = 3
nwalkers = 32
nsteps = 2000
burnin = 500
thin = 10

# Initialize walkers near best fit with small scatter
pos = best_params + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([0.1, 1, 0.1])
pos = np.clip(pos, [0.01, 30, -5], [0.99, 120, 5])

# Run MCMC
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, nsteps, progress=True)

# Extract samples (discard burn-in, thin)
samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

# Parameter statistics
Om_samples = samples[:, 0]
H0_samples = samples[:, 1]
delta_samples = samples[:, 2]

Om_result = np.percentile(Om_samples, [16, 50, 84])
H0_result = np.percentile(H0_samples, [16, 50, 84])
delta_result = np.percentile(delta_samples, [16, 50, 84])

print(f"\nMethod 3 (MCMC Results):")
print(f"  Omega_m0 = {Om_result[1]:.5f} +{Om_result[2]-Om_result[1]:.5f} -{Om_result[1]-Om_result[0]:.5f}")
print(f"  H0       = {H0_result[1]:.2f} +{H0_result[2]-H0_result[1]:.2f} -{H0_result[1]-H0_result[0]:.2f}")
print(f"  delta    = {delta_result[1]:.5f} +{delta_result[2]-delta_result[1]:.5f} -{delta_result[1]-delta_result[0]:.5f}")

# ============================================================
# Create contour plots
# ============================================================

print("\nGenerating plots...")
setup_matplotlib()

# Figure 1: Corner plot showing all parameters
fig = corner.corner(
    samples, 
    labels=[r"$\Omega_{m,0}$", r"$H_0$", r"$\delta$"],
    truths=best_params,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
    color='darkblue'
)
plt.suptitle("MCMC Parameter Constraints", fontsize=14, y=0.98)
plt.savefig('attemt2_corner_plot_delta.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 2: 2D contour for delta vs H0 (showing tension)
fig, ax = plt.subplots(figsize=(10, 8))

# Create 2D histogram
H_bins = 40
delta_bins = 40
H_range = (55, 85)
delta_range = (-0.5, 1.5)

H_hist, H_edges, delta_edges = np.histogram2d(
    H0_samples, delta_samples,
    bins=[H_bins, delta_bins],
    range=[H_range, delta_range]
)

# Find levels for confidence contours
smoothed = gaussian_filter(H_hist, sigma=1.5)
max_val = smoothed.max()
levels = [max_val * 0.68, max_val * 0.95, max_val * 0.997]

# Plot
contour_plot = ax.contour(H_edges[:-1], delta_edges[:-1], smoothed.T, 
                         levels=levels, colors='darkblue', linewidths=2)
ax.contourf(H_edges[:-1], delta_edges[:-1], smoothed.T,
            levels=[0, max_val*0.68], colors='lightblue', alpha=0.5)

# Best fit point
ax.plot(best_H0, best_delta, 'r*', markersize=15, label='Best fit')

# Mean from MCMC
ax.plot(H0_result[1], delta_result[1], 'bo', markersize=10, label='MCMC mean')

# Literature values - FIXED: Use proper LaTeX for Greek letters
literature = {
    r"Planck $\Lambda$CDM": (67.4, 0.0),
    "SH0ES": (73.04, 0.0),
}

for name, (H0_val, delta_val) in literature.items():
    ax.axvline(H0_val, color='gray', linestyle='--', alpha=0.5)
    ax.text(H0_val + 0.5, delta_range[0] + 0.1, name, rotation=90, fontsize=9, alpha=0.7)

ax.set_xlabel(r"$H_0$ [km/s/Mpc]", fontsize=14)
ax.set_ylabel(r"$\delta$", fontsize=14)
ax.set_title(r"$\delta$ vs $H_0$ Contours", fontsize=14)
ax.legend()
ax.grid(alpha=0.3)
ax.set_xlim(H_range)
ax.set_ylim(delta_range)

# Add text with results
textstr = f'$\\delta = {delta_result[1]:.4f}^{{+{delta_result[2]-delta_result[1]:.4f}}}_{{-{delta_result[1]-delta_result[0]:.4f}}}$'
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('attemt2_delta_vs_H0_contours.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 3: Hubble diagram with data and best-fit model
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                               gridspec_kw={'height_ratios': [3, 1]})

# Main plot
z_fine = np.linspace(0, max(z_vals)*1.1, 100)

# ΛCDM model (delta=0)
H_LCDM = H_model(z_fine, best_Om, best_H0, 0)
H_best = H_model(z_fine, best_Om, best_H0, best_delta)

ax1.errorbar(z_vals, H_vals, yerr=sigma_vals, fmt='o', 
             color='darkblue', capsize=2, label='Data', alpha=0.6)
ax1.plot(z_fine, H_best, 'r-', linewidth=2, label=f'Best fit ($\\delta={best_delta:.4f}$)')
ax1.plot(z_fine, H_LCDM, 'g--', linewidth=2, label=r'$\Lambda$CDM ($\delta=0$)')

ax1.set_xlabel(r'$z$', fontsize=14)
ax1.set_ylabel(r'$H(z)$ [km/s/Mpc]', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(alpha=0.3)

# Residuals
residuals = H_vals - H_model(z_vals, best_Om, best_H0, best_delta)
ax2.errorbar(z_vals, residuals, yerr=sigma_vals, fmt='o', 
             color='darkblue', capsize=2, alpha=0.6)
ax2.axhline(0, color='red', linestyle='--', linewidth=1)
ax2.set_xlabel(r'$z$', fontsize=14)
ax2.set_ylabel(r'Residuals [km/s/Mpc]', fontsize=14)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('attemt2_hubble_diagram_delta.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 4: Delta posterior distribution
fig, ax = plt.subplots(figsize=(10, 6))

# Histogram of delta samples
n, bins, patches = ax.hist(delta_samples, bins=50, density=True, 
                           alpha=0.7, color='darkblue', edgecolor='black')

# Add Gaussian fit
from scipy.stats import norm
mu, sigma = norm.fit(delta_samples)
x = np.linspace(delta_samples.min(), delta_samples.max(), 100)
ax.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
        label=f'Gaussian fit: $\\mu={mu:.4f}, \\sigma={sigma:.4f}$')

# Mark best fit and confidence intervals
ax.axvline(best_delta, color='red', linestyle='--', linewidth=2, label='Best fit')
ax.axvline(delta_result[1], color='green', linestyle='-', linewidth=2, label='Median')
ax.axvline(delta_result[0], color='green', linestyle=':', linewidth=1, alpha=0.7)
ax.axvline(delta_result[2], color='green', linestyle=':', linewidth=1, alpha=0.7)

# Add shaded region for 68% CL
ax.axvspan(delta_result[0], delta_result[2], alpha=0.2, color='green', 
           label='68% CL')

ax.set_xlabel(r'$\delta$', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)
ax.set_title('Posterior Distribution of $\\delta$', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('attemt2_delta_posterior.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# Summary of results
# ============================================================

print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print(f"\nBest-fit parameters (MLE):")
print(f"  Ω_m,0 = {best_Om:.4f}")
print(f"  H_0   = {best_H0:.2f} km/s/Mpc")
print(f"  δ     = {best_delta:.4f}")

print(f"\nMCMC constraints (68% CL):")
print(f"  Ω_m,0 = {Om_result[1]:.4f} ± {max(Om_result[2]-Om_result[1], Om_result[1]-Om_result[0]):.4f}")
print(f"  H_0   = {H0_result[1]:.2f} ± {max(H0_result[2]-H0_result[1], H0_result[1]-H0_result[0]):.2f} km/s/Mpc")
print(f"  δ     = {delta_result[1]:.4f} ± {max(delta_result[2]-delta_result[1], delta_result[1]-delta_result[0]):.4f}")

print(f"\nGoodness of fit:")
print(f"  χ²_min  = {min_chi2:.2f}")
print(f"  χ²/dof  = {min_chi2/dof:.3f}")

# Interpretation of delta
if abs(best_delta) < 0.1:
    print(f"\nInterpretation: δ ≈ 0, consistent with ΛCDM model")
elif best_delta > 0:
    print(f"\nInterpretation: δ > 0 suggests modified gravity/DE at z~{np.mean(z_vals):.2f}")
else:
    print(f"\nInterpretation: δ < 0 suggests tension with ΛCDM at z~{np.mean(z_vals):.2f}")

print("\n" + "="*60)