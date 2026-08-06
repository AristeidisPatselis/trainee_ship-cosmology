#!/usr/bin/env python3
"""
combine_and_check.py
====================
Combined script that:
1. Checks directory structure and finds all model folders (from check_folders.py)
2. Combines confidence contours from all models into a single plot matching the reference image style.

Usage:
    python combine_and_check.py [--base-dir PATH] [--check-only]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import re
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG - MATCH YOUR ACTUAL DIRECTORY STRUCTURE
# =============================================================================

DEFAULT_BASE_DIR = (
    "/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/Academia/Patra-Physics/Traineeship/Codes_0"
)


def get_base_dir():
    """Resolve BASE_DIR: CLI arg > env var > hardcoded default."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-dir", default=None)
    args, _ = parser.parse_known_args()
    return (
        args.base_dir
        or os.environ.get("COSMO_BASE_DIR")
        or DEFAULT_BASE_DIR
    )


BASE_DIR = get_base_dir()
OUTPUT_DIR = os.path.join(BASE_DIR, "comparison", "results_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model configurations — updated to match the reference image
MODELS = [
    {
        'name': 'lcdm',
        'folder': 'model_lcdm',
        'subfolder': 'results',
        'label': r'$\Lambda$CDM',
        'color': '#DC143C',  # Crimson red
        'marker': '*',
        'contour_file': 'contour_H0_Om_lcdm.npy',
        'summary_file': 'lcdm_fit_results.txt',
    },
    {
        'name': 'lcdm_hdot',
        'folder': 'model_a',
        'subfolder': 'results',
        'label': r'$\Lambda$CDM$+\dot{H}$',
        'color': '#FF8C00',  # Dark orange
        'marker': '*',
        'contour_file': 'contour_H0_Om_hdot_alpha.npy',
        'summary_file': 'fit_summary.txt',
    },
    {
        'name': 'bh4_hdot',
        'folder': 'model_delta',
        'subfolder': 'delta_lcdm_fit/results',
        'label': r'$bH^4+\dot{H}$',
        'color': '#2E8B57',  # Sea green
        'marker': '*',
        'contour_file': 'contour_H0_Om_delta_free.npy',
        'summary_file': 'fit_summary.txt',
    },
    {
        'name': 'bhdelta',
        'folder': 'model_delta4',
        'subfolder': 'results_delta4',
        'label': r'$bH^\delta$',
        'color': '#4169E1',  # Royal blue
        'marker': '*',
        'contour_file': 'contour_H0_Om_delta4.npy',
        'summary_file': 'fit_summary_delta4.txt',
    },
    {
        'name': 'bhdelta_hdot',
        'folder': 'model_delta4_a',
        'subfolder': 'results_delta4_alpha',
        'label': r'$bH^\delta+\dot{H}$',
        'color': '#800080',  # Purple
        'marker': '*',
        'contour_file': 'contour_H0_Om_delta4_alpha.npy',
        'summary_file': 'fit_summary_delta4_alpha.txt',
    },
]

# Shared Delta-chi2 levels / styling (module level so they're defined once)
LEVELS = [0, 2.30, 6.18, 11.83]
TIER_ALPHAS = [0.35, 0.20, 0.10]  # 1σ, 2σ, 3σ fill transparency

# =============================================================================
# CHECK FOLDERS FUNCTIONALITY (from check_folders.py)
# =============================================================================

def check_folders():
    """Check the directory structure and find all model folders and their results."""
    print("=" * 70)
    print("DIRECTORY STRUCTURE CHECK")
    print("=" * 70)
    print(f"Base directory: {BASE_DIR}")
    print()

    if not os.path.exists(BASE_DIR):
        print(f"ERROR: Base directory does not exist: {BASE_DIR}")
        return False

    # List all directories
    print("All directories in BASE_DIR:")
    for item in sorted(os.listdir(BASE_DIR)):
        full_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(full_path):
            print(f"  📁 {item}")
            # Check for result files
            result_files = []
            for root, dirs, files in os.walk(full_path):
                for f in files:
                    if f.endswith('.txt') or f.endswith('.npy'):
                        if 'summary' in f.lower() or 'fit' in f.lower() or 'contour' in f.lower():
                            rel_path = os.path.relpath(os.path.join(root, f), full_path)
                            result_files.append(rel_path)
                if len(result_files) > 5:
                    break

            if result_files:
                print(f"     Found result files:")
                for rf in result_files[:5]:
                    print(f"       - {rf}")
                if len(result_files) > 5:
                    print(f"       ... and {len(result_files) - 5} more")
            else:
                print(f"     No result files found")

    print()
    print("=" * 70)

    # Find all fit_summary files
    print("\nAll fit_summary files found:")
    summary_count = 0
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if 'summary' in f.lower() and f.endswith('.txt'):
                rel_path = os.path.relpath(os.path.join(root, f), BASE_DIR)
                print(f"  - {rel_path}")
                summary_count += 1
    if summary_count == 0:
        print("  (none found)")

    print()
    print("=" * 70)

    # Find all contour files
    print("\nAll contour .npy files found:")
    contour_count = 0
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if 'contour' in f.lower() and f.endswith('.npy'):
                rel_path = os.path.relpath(os.path.join(root, f), BASE_DIR)
                print(f"  - {rel_path}")
                contour_count += 1
    if contour_count == 0:
        print("  (none found)")

    print()
    print("=" * 70)
    return True


# =============================================================================
# PLOTTING FUNCTIONS (from combine_contours.py, restyled)
# =============================================================================

def setup_matplotlib():
    """Configure matplotlib for publication-quality plots."""
    common = dict(labelsize=14, titlesize=16)
    try:
        rc('text', usetex=True)
        rc('font', family='serif')
        rc('font', size=12)
        rc('axes', labelsize=common['labelsize'], titlesize=common['titlesize'])
        rc('legend', fontsize=10)
        rc('figure', dpi=300)
        print("  Using LaTeX rendering")
        return True
    except Exception:
        rc('text', usetex=False)
        rc('font', family='DejaVu Sans')
        rc('font', size=12)
        rc('axes', labelsize=common['labelsize'], titlesize=common['titlesize'])
        rc('legend', fontsize=10)
        rc('figure', dpi=300)
        print("  Using mathtext rendering")
        return False


def get_full_path(model_config):
    """Get the full path to the model's results folder."""
    folder_path = os.path.join(BASE_DIR, model_config['folder'])
    subfolder = model_config.get('subfolder', '')
    return os.path.join(folder_path, subfolder) if subfolder else folder_path


def _find_file_by_hint(folder_path, hint, ext='.txt'):
    """Walk folder_path looking for a file containing `hint` in its name."""
    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            if hint in f.lower() and f.endswith(ext):
                return os.path.join(root, f)
    return None


def extract_best_fit_from_summary(model_config):
    """Extract H0 and Om from a model's summary file."""
    folder_path = get_full_path(model_config)
    if not os.path.exists(folder_path):
        return None, None

    summary_file = model_config.get('summary_file', 'fit_summary.txt')
    filepath = os.path.join(folder_path, summary_file)

    if not os.path.exists(filepath):
        filepath = _find_file_by_hint(folder_path, 'summary') or filepath

    if not os.path.exists(filepath):
        return None, None

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except OSError:
        return None, None

    h0 = om = None
    for p in (r'H0\s*=\s*([0-9.]+)', r'H_0\s*=\s*([0-9.]+)', r'best_H0\s*=\s*([0-9.]+)'):
        m = re.search(p, content)
        if m:
            h0 = float(m.group(1))
            break

    for p in (r'Om\s*=\s*([0-9.]+)', r'Omega_m\s*=\s*([0-9.]+)',
              r'best_Om\s*=\s*([0-9.]+)', r'Omega_m\s+([0-9.]+)'):
        m = re.search(p, content)
        if m:
            om = float(m.group(1))
            break

    if h0 is not None and om is not None:
        return h0, om
    return None, None


def load_contour_data(model_config):
    """Load the raw Delta-chi2 contour grid dict for a model."""
    folder_path = get_full_path(model_config)
    if not os.path.exists(folder_path):
        return None

    contour_file = model_config.get('contour_file')
    if contour_file:
        filepath = os.path.join(folder_path, contour_file)
        if os.path.exists(filepath):
            try:
                return np.load(filepath, allow_pickle=True).item()
            except Exception:
                pass

    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            if f.startswith('contour_') and f.endswith('.npy'):
                try:
                    return np.load(os.path.join(root, f), allow_pickle=True).item()
                except Exception:
                    continue
    return None


def _orient_grid(X, Y, Z, best_H0, best_Om):
    """
    Ensure Om is on the X axis and H0 is on the Y axis, regardless of how
    the grid was stored. H0 ~ O(50-100), Om ~ O(0-1), so we use the
    magnitude of the grid means (nan-safe) to detect orientation.
    """
    x_mean = np.nanmean(X)
    y_mean = np.nanmean(Y)

    # X currently looks like H0 (large) and Y looks like Om (small) -> swap
    if x_mean > 10 and y_mean <= 10:
        plot_X, plot_Y = Y, X
        plot_Z = Z.T if Z.shape[0] != Z.shape[1] else Z
    else:
        plot_X, plot_Y = X, Y
        plot_Z = Z

    return plot_X, plot_Y, plot_Z, best_Om, best_H0


def create_comparison_plot(model_data_list, save_path=None):
    """
    Create a combined contour plot styled to match the reference image.

    Features:
    - Filled contours with tiered alpha transparency
    - Dashed contour borders (matching the image style)
    - Star markers for best-fit points
    - Clean legend in upper right
    - No literature bands
    - Title: "Comparison of all cosmological models"
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for model_data in model_data_list:
        Z = model_data.get('grid_Z')
        if Z is None:
            continue

        X, Y = model_data['grid_X'], model_data['grid_Y']
        best_H0, best_Om = model_data.get('best_fit', (None, None))

        plot_X, plot_Y, plot_Z, plot_best_X, plot_best_Y = _orient_grid(
            X, Y, Z, best_H0, best_Om
        )
        plot_Z = np.ma.masked_invalid(plot_Z)

        color = model_data.get('color', 'gray')
        label = model_data['label']

        # Filled contours, drawn tier-by-tier to control alpha blending
        for i in range(3):
            ax.contourf(
                plot_X, plot_Y, plot_Z,
                levels=[LEVELS[i], LEVELS[i + 1]],
                colors=[color], alpha=TIER_ALPHAS[i], zorder=10 - i,
            )

        # Dashed outline borders — matching the reference image style
        ax.contour(
            plot_X, plot_Y, plot_Z, levels=LEVELS[1:],
            colors=[color], linewidths=1.5, linestyles='dashed', zorder=15,
        )

        # Bounds come from the GRID itself (nan-safe)
        min_x = min(min_x, np.nanmin(plot_X))
        max_x = max(max_x, np.nanmax(plot_X))
        min_y = min(min_y, np.nanmin(plot_Y))
        max_y = max(max_y, np.nanmax(plot_Y))

        # Best-fit marker (star, matching the image)
        if plot_best_X is not None and plot_best_Y is not None:
            marker = model_data.get('marker', '*')
            ax.plot(
                plot_best_X, plot_best_Y, marker=marker, markersize=12,
                color=color, markeredgecolor='black', markeredgewidth=0.8,
                zorder=20,
            )
            min_x, max_x = min(min_x, plot_best_X), max(max_x, plot_best_X)
            min_y, max_y = min(min_y, plot_best_Y), max(max_y, plot_best_Y)

        # Proxy artist for the legend
        ax.plot([], [], color=color, linewidth=2.5, label=label)

    # Apply padded limits
    if min_x != float('inf'):
        x_pad = (max_x - min_x) * 0.08
        y_pad = (max_y - min_y) * 0.08
        ax.set_xlim(min_x - x_pad, max_x + x_pad)
        ax.set_ylim(min_y - y_pad, max_y + y_pad)
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(55, 85)

    # Formatting — matching the reference image
    ax.set_xlabel(r'$\Omega_{m,0}$', fontsize=14)
    ax.set_ylabel(r'$H_0$ [km/s/Mpc]', fontsize=14)
    ax.set_title('Comparison of all cosmological models', fontsize=16)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='gray')

    # Clean spines
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'Comparison_all_cosmological_models.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {save_path}")
    if min_x != float('inf'):
        print(
            f"  > Axis limits: "
            f"x=({ax.get_xlim()[0]:.2f}, {ax.get_xlim()[1]:.2f}), "
            f"y=({ax.get_ylim()[0]:.2f}, {ax.get_ylim()[1]:.2f})"
        )
    plt.close(fig)


def load_model_data(model_config):
    """Load best-fit + contour grid data for one model config."""
    H0, Om = extract_best_fit_from_summary(model_config)
    best_fit = (H0, Om) if H0 is not None and Om is not None else None
    if best_fit:
        print(f"  ✓ Best fit: H0={H0:.2f}, Om={Om:.4f}")
    else:
        print("  ! No best fit found")

    data = load_contour_data(model_config)
    grid_X = grid_Y = grid_Z = None

    if data is not None:
        if all(k in data for k in ('X', 'Y', 'delta_chi2')):
            grid_X, grid_Y, grid_Z = data['X'], data['Y'], data['delta_chi2']
            print("  ✓ Extracted grid data")
        elif all(k in data for k in ('xx', 'yy', 'delta_chisq')):
            grid_X, grid_Y, grid_Z = data['xx'], data['yy'], data['delta_chisq']
            print("  ✓ Extracted grid data")
        else:
            print(f"  ! Data keys: {list(data.keys())}")
    else:
        print("  ! No contour data found")

    return {
        'name': model_config['name'],
        'label': model_config['label'],
        'color': model_config.get('color', 'gray'),
        'marker': model_config.get('marker', '*'),
        'best_fit': best_fit,
        'grid_X': grid_X,
        'grid_Y': grid_Y,
        'grid_Z': grid_Z,
    }


def print_summary(model_data_list):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'H0':<12} {'Ωm':<12} {'Grid Status':<10}")
    print("-" * 60)
    for model_data in model_data_list:
        H0, Om = model_data['best_fit'] if model_data['best_fit'] else ('N/A', 'N/A')
        grid_status = "OK" if model_data.get('grid_Z') is not None else "Missing"
        if isinstance(H0, float):
            print(f"{model_data['label']:<25} {H0:<12.2f} {Om:<12.4f} {grid_status:<10}")
        else:
            print(f"{model_data['label']:<25} {H0:<12} {Om:<12} {grid_status:<10}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Check folders and create combined cosmological contour plot."
    )
    parser.add_argument(
        "--base-dir", default=None,
        help="Override the base directory path."
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only run the folder check, skip plotting."
    )
    args = parser.parse_args()

    global BASE_DIR
    if args.base_dir:
        BASE_DIR = args.base_dir
        print(f"Using base directory from CLI: {BASE_DIR}")

    print("=" * 70)
    print("MODEL CONTOUR COMPARISON AND VISUALIZATION")
    print("=" * 70)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Step 1: Check folders (from check_folders.py)
    print("\n>>> STEP 1: Checking directory structure...")
    check_folders()

    if args.check_only:
        print("\n--check-only flag set. Skipping plot generation.")
        return

    # Step 2: Setup matplotlib and create plot (from combine_contours.py)
    print("\n>>> STEP 2: Setting up matplotlib...")
    setup_matplotlib()

    print("\n>>> STEP 3: Loading model data...")
    model_data_list = []
    for model_config in MODELS:
        print(f"\nProcessing: {model_config['label']}")
        model_data_list.append(load_model_data(model_config))

    print_summary(model_data_list)

    print("\n" + "=" * 70)
    print("Creating comparison plot...")
    create_comparison_plot(model_data_list)

    print("\n" + "=" * 70)
    print("Done! Results saved in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
