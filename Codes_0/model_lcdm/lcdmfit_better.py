import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import emcee
import corner
from matplotlib import rc
from tkinter import Tk, filedialog

# =============================================================================
# CONFIG
# =============================================================================
# Change these to use different data files or directories
# The directory is relative to the script location
DATA_DIR = '/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0/Data_Sets/'  # Directory containing the data files
# Alternatively, use an absolute path:
# DATA_DIR = '/home/aristeidismp/Desktop/MyData/'

# Data file names - change these to use different datasets
Z_FILE = 'c_z_vals.txt'           # Redshift values
H_FILE = 'c_H_vals.txt'           # H(z) measurements
SIGMA_FILE = 'c_sigma_vals.txt'   # Errors on H(z)

# =============================================================================
# SETUP
# =============================================================================

def setup_matplotlib():
    """Attempts to enable LaTeX formatting for professional plots."""
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
    except Exception as e:
        print(f"Warning: LaTeX rendering not enabled. Error: {e}")

def load_clean_data(filename, data_dir):
    """
    Safely loads numeric data from a text file with recursive search.
    Strips out potential prompt artifacts before converting to float.
    """
    # Find the file path recursively
    filepath = find_file_recursively(filename, data_dir)
    print(f"  Found: {filepath}")
    
    data = []
    with open(filepath, "r") as f:
        for line in f:
            clean_line = line.split("]")[-1].strip()
            if clean_line:
                data.append(float(clean_line))
    return np.array(data)

def find_file_recursively(filename, data_dir):
    """
    Search for a file recursively in data_dir and its subdirectories.
    Returns the full path if found, raises FileNotFoundError otherwise.
    """
    # First check in the main directory
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return filepath
    
    # If not found, search recursively in subdirectories
    for root, dirs, files in os.walk(data_dir):
        if filename in files:
            return os.path.join(root, filename)
    
    # If still not found, raise an error with helpful message
    available_files = list_available_files(data_dir)
    raise FileNotFoundError(
        f"Could not find '{filename}' in '{data_dir}' or its subdirectories.\n"
        f"Available .txt files in {data_dir} and subdirectories:\n{available_files}"
    )

def list_available_files(data_dir, max_files=30):
    """List available .txt files in data_dir and subdirectories."""
    files = []
    for root, dirs, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.txt'):
                rel_path = os.path.relpath(os.path.join(root, f), data_dir)
                files.append(rel_path)
                if len(files) >= max_files:
                    files.append(f"... and more")
                    return "\n".join(files)
    return "\n".join(files) if files else "No .txt files found"

def load_all_data():
    """Load z, H(z), sigma_H arrays from the configured DATA_DIR.
    
    Searches recursively through all subdirectories.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # If DATA_DIR is a relative path, make it absolute
    if not os.path.isabs(DATA_DIR):
        data_dir = os.path.join(script_dir, DATA_DIR)
    else:
        data_dir = DATA_DIR
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    print(f"Loading data from: {data_dir}")
    print(f"  Searching for: {Z_FILE}, {H_FILE}, {SIGMA_FILE}")
    
    z_vals = load_clean_data(Z_FILE, data_dir)
    H_vals = load_clean_data(H_FILE, data_dir)
    sigma_vals = load_clean_data(SIGMA_FILE, data_dir)
    
    if len(z_vals) != len(H_vals) or len(H_vals) != len(sigma_vals):
        raise ValueError(
            f"Data files have mismatched lengths: "
            f"z={len(z_vals)}, H={len(H_vals)}, sigma={len(sigma_vals)}"
        )
    
    return z_vals, H_vals, sigma_vals

def select_data_folder():
    """Open a dialog to select the data folder."""
    root = Tk()
    root.withdraw()  # Hide tkinter window
    
    folder = filedialog.askdirectory(
        title="Select the folder containing z_vals.txt, H_vals.txt, sigma_vals.txt"
    )
    
    root.destroy()
    return folder

# =============================================================================
# 2. COSMOLOGICAL MODEL & STATISTICS
# =============================================================================

def H_model(z, Om_m0, H_0):
    """
    Theoretical Hubble parameter for a flat Lambda-CDM model.
    Formula: H(z) = H0 * sqrt(Omega_m * (1+z)^3 + (1 - Omega_m))
    """
    E_z = np.sqrt(Om_m0 * (1 + z)**3 + (1 - Om_m0))
    return E_z * H_0

def calc_chisq(pars, z_vals, H_vals, sigma_vals):
    """
    Computes the Chi-squared statistic for given cosmological parameters.
    Fully vectorized to support either 1D scalar inputs or 2D parameter grids.
    """
    Om_m0 = np.atleast_1d(pars[0])
    H_0   = np.atleast_1d(pars[1])

    z_expand = z_vals[:, None]      
    Om_expand = Om_m0[None, :]      
    H0_expand = H_0[None, :]        

    theorH = H_model(z_expand, Om_expand, H0_expand)
    
    chi2_components = (H_vals[:, None] - theorH)**2 / sigma_vals[:, None]**2
    return np.sum(chi2_components, axis=0)

# =============================================================================
# 3. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    setup_matplotlib()

    # Create output directory for saving plots and results
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}\n")

    # --- Step 1: Select Folder and Load Data ---
    print("--- Select Data Folder ---")
    
    # Option 1: Use GUI dialog to select folder
    # folder = select_data_folder()
    # if not folder:
    #     print("No folder selected. Exiting.")
    #     return
    # print(f"\nData folder: {folder}")
    
    # Option 2: Use the configured DATA_DIR (recommended)
    # Uncomment the GUI version above and comment this out if you prefer the dialog
    
    print("--- Loading Data ---")
    
    try:
        # Use the new load_all_data function with configurable DATA_DIR
        z_vals, H_vals, sigma_vals = load_all_data()
        print(f"\nSuccessfully loaded {len(z_vals)} data points.")
        print(f"Redshift range: {z_vals.min():.3f} to {z_vals.max():.3f}\n")
        
    except FileNotFoundError as e:
        print(f"Critical Error: Data files missing. {e}")
        return
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return

    # --- Step 2: Frequentist Optimization (Curve Fitting) ---
    print("--- Optimization Results ---")
    
    p0 = [0.3, 70.0] 
    bounds = ([0.0, 50.0], [1.0, 100.0]) 

    popt, pcov = opt.curve_fit(
        H_model, z_vals, H_vals,
        p0=p0, sigma=sigma_vals, absolute_sigma=True, bounds=bounds
    )

    best_Om, best_H0 = popt
    Om_err, H0_err = np.sqrt(np.diag(pcov))

    min_chisq = calc_chisq(popt, z_vals, H_vals, sigma_vals)[0]
    dof = len(z_vals) - len(popt)

    print(f"Omega_m       = {best_Om:.4f} +/- {Om_err:.4f}")
    print(f"Omega_Lambda  = {1 - best_Om:.4f} +/- {Om_err:.4f}")
    print(f"H_0           = {best_H0:.4f} +/- {H0_err:.4f}")
    print(f"chi^2_reduced = {min_chisq:.2f}/{dof} = {min_chisq/dof:.3f}\n")

    # --- Step 3: Chi-Squared Grid & Contour Plotting ---
    print("--- Generating Chi-Squared Maps ---")
    sample_rate = 100
    Om_space = np.linspace(0.0, 1.0, sample_rate)
    H0_space = np.linspace(55, 80, sample_rate)

    xx, yy = np.meshgrid(Om_space, H0_space)

    Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T
    result = calc_chisq([Xgrid[:, 0], Xgrid[:, 1]], z_vals, H_vals, sigma_vals)

    Z = result.reshape(xx.shape)
    delta_chisq = Z - min_chisq 

    confidence_levels = [2.30, 6.18, 11.83]

    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(xx, yy, delta_chisq, levels=[0] + confidence_levels, cmap='viridis_r', extend='max')
    cs_lines = ax.contour(xx, yy, delta_chisq, levels=confidence_levels, colors='white', linewidths=1)
    
    ax.clabel(cs_lines, inline=True, fontsize=10, fmt={2.30: r'1$\sigma$', 6.18: r'2$\sigma$', 11.83: r'3$\sigma$'})
    ax.plot(best_Om, best_H0, 'r*', markersize=15, label='Best fit')
    
    ax.set_xlabel(r'$\Omega_{m,0}$')
    ax.set_ylabel(r'$H_0$')
    ax.set_title(r'$\Delta\chi^2$ Confidence Contours')
    ax.legend()
    fig.colorbar(cf, ax=ax, label=r'$\Delta\chi^2$')
    
    # Save before showing
    plt.savefig(os.path.join(output_dir, "DeltaChi2_Contour.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # Close the figure to free memory

    # --- Step 4: Bayesian MCMC Sampling ---
    print("\n--- Running MCMC Sampling ---")

    def log_prior(theta):
        Om, H0 = theta
        if 0.0 < Om < 1.0 and 40.0 < H0 < 100.0:
            return 0.0
        return -np.inf

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = H_model(z_vals, theta[0], theta[1])
        log_likelihood = -0.5 * np.sum((H_vals - model)**2 / sigma_vals**2)
        return lp + log_likelihood

    ndim, nwalkers, nsteps = 2, 32, 3000

    pos = popt + 1e-3 * np.random.randn(nwalkers, ndim) * np.array([1, 10])

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=True)

    flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)

    Om_mcmc = np.percentile(flat_samples[:, 0], [16, 50, 84])
    H0_mcmc = np.percentile(flat_samples[:, 1], [16, 50, 84])

    print(f"\nMCMC Omega_m = {Om_mcmc[1]:.4f} (+{Om_mcmc[2]-Om_mcmc[1]:.4f} / -{Om_mcmc[1]-Om_mcmc[0]:.4f})")
    print(f"MCMC H_0     = {H0_mcmc[1]:.2f} (+{H0_mcmc[2]-H0_mcmc[1]:.2f} / -{H0_mcmc[1]-H0_mcmc[0]:.2f})\n")

    fig_corner = corner.corner(
        flat_samples, 
        labels=[r"$\Omega_{m,0}$", r"$H_0$"],
        truths=[best_Om, best_H0],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12}
    )
    
    # Save before showing
    plt.savefig(os.path.join(output_dir, "MCMC_H_VS_MCMC_Omega_m.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig_corner)  # Close the figure to free memory

    # --- Step 5: Hubble Tension Visualization ---
    literature = {
        "This Work (CC)":     (best_H0, H0_err,   'crimson'),
        "Planck 2018 (CMB)":   (67.4,  0.5,  'steelblue'),
        "SH0ES 2022 (Local)":  (73.04, 1.04, 'darkorange'),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, (val, err, color)) in enumerate(literature.items()):
        ax.errorbar(val, i, xerr=err, fmt='o', color=color, capsize=4, markersize=9)

    ax.set_yticks(range(len(literature)))
    ax.set_yticklabels(literature.keys())
    ax.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax.set_title(r"Hubble Parameter: Current Fit vs. The $H_0$ Tension")

    ax.axvspan(67.4-0.5, 67.4+0.5, color='steelblue', alpha=0.15)
    ax.axvspan(73.04-1.04, 73.04+1.04, color='darkorange', alpha=0.15)

    plt.tight_layout()
    
    # Save before showing
    plt.savefig(os.path.join(output_dir, "Hubble_Parameter.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)  # Close the figure to free memory

    # --- Step 6: Export Results (Optional) ---
    print("\n--- Exporting Results ---")
    
    # Export best-fit parameters
    with open(os.path.join(output_dir, "lcdm_fit_results.txt"), "w") as f:
        f.write("# LCDM Fit Results\n")
        f.write("# =================\n")
        f.write(f"Omega_m       = {best_Om:.6f} +/- {Om_err:.6f}\n")
        f.write(f"Omega_Lambda  = {1 - best_Om:.6f} +/- {Om_err:.6f}\n")
        f.write(f"H_0           = {best_H0:.6f} +/- {H0_err:.6f}\n")
        f.write(f"chi^2         = {min_chisq:.6f}\n")
        f.write(f"dof           = {dof}\n")
        f.write(f"chi^2_reduced = {min_chisq/dof:.6f}\n")
        f.write(f"MCMC Omega_m  = {Om_mcmc[1]:.6f} (+{Om_mcmc[2]-Om_mcmc[1]:.6f} / -{Om_mcmc[1]-Om_mcmc[0]:.6f})\n")
        f.write(f"MCMC H_0      = {H0_mcmc[1]:.6f} (+{H0_mcmc[2]-H0_mcmc[1]:.6f} / -{H0_mcmc[1]-H0_mcmc[0]:.6f})\n")
    
    print(f"  Results exported to: {os.path.join(output_dir, 'lcdm_fit_results.txt')}")
    print(f"  Plots saved in: {output_dir}")

    # Save contour data for later comparison 
    # Note: lcdmfit_better.py uses xx (Om) and yy (H0) for the grid
    contour_data = {'X': xx, 'Y': yy, 'delta_chi2': delta_chisq}
    np.save(os.path.join(output_dir, 'contour_H0_Om_lcdm.npy'), contour_data)
    print(f"  Saved contour data to: {os.path.join(output_dir, 'contour_H0_Om_lcdm.npy')}")

if __name__ == "__main__":
    main()