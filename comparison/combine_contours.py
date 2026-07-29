"""
combine_contours.py
====================
Combine confidence contours from all models into a single plot,
using raw Delta-chi2 grids for accurate, overlapping filled contours.
Automatically scales the axes to ensure ALL contours (and best-fit points)
are fully visible, by deriving bounds from the parameter grids themselves
rather than from the drawn contour line geometry.
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
    "/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/"
    "Academia/Patra-Physics/Traineeship/Codes"
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

print(f"Base directory: {BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Model configurations
MODELS = [
    {
        'name': 'lcdm',
        'folder': 'model_lcdm',
        'subfolder': 'results',
        'label': r'$\Lambda$CDM',
        'color': '#1f77b4',
        'marker': 's',
        'contour_file': 'contour_H0_Om_lcdm.npy',
        'summary_file': 'lcdm_fit_results.txt',
    },
    {
        'name': 'hdot_alpha',
        'folder': 'model_a',
        'subfolder': 'results',
        'label': r'$\dot{H}$-$\alpha$',
        'color': '#ff7f0e',
        'marker': 'o',
        'contour_file': 'contour_H0_Om_hdot_alpha.npy',
        'summary_file': 'fit_summary.txt',
    },
    {
        'name': 'delta_free',
        'folder': 'model_delta',
        'subfolder': 'delta_lcdm_fit/results',
        'label': r'$\delta$-LCDM',
        'color': '#2ca02c',
        'marker': '^',
        'contour_file': 'contour_H0_Om_delta_free.npy',
        'summary_file': 'fit_summary.txt',
    },
    {
        'name': 'delta4',
        'folder': 'model_delta4',
        'subfolder': 'results_delta4',
        'label': r'$\delta=4$',
        'color': '#d62728',
        'marker': 'D',
        'contour_file': 'contour_H0_Om_delta4.npy',
        'summary_file': 'fit_summary_delta4.txt',
    },
    {
        'name': 'delta4_alpha',
        'folder': 'model_delta4_a',
        'subfolder': 'results_delta4_alpha',
        'label': r'$\delta=4, \alpha$',
        'color': '#9467bd',
        'marker': 'v',
        'contour_file': 'contour_H0_Om_delta4_alpha.npy',
        'summary_file': 'fit_summary_delta4_alpha.txt',
    },
    {
        'name': 'delta_alpha',
        'folder': 'model_delta_a',
        'subfolder': 'results_delta_alpha',
        'label': r'$\delta, \alpha$ free',
        'color': '#8c564b',
        'marker': '*',
        'contour_file': 'contour_H0_Om_delta_alpha_free.npy',
        'summary_file': 'fit_summary_delta_alpha.txt',
    },
]

LITERATURE = {
    "Planck 2018": (67.4, 0.5),
    "SH0ES 2022": (73.04, 1.04),
}

# Shared Delta-chi2 levels / styling (module level so they're defined once)
LEVELS = [0, 2.30, 6.18, 11.83]
LEVEL_LABELS = {2.30: r'$1\sigma$', 6.18: r'$2\sigma$', 11.83: r'$3\sigma$'}
TIER_ALPHAS = [0.4, 0.2, 0.08]  # 1σ, 2σ, 3σ fill transparency

# =============================================================================
# FUNCTIONS
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


def create_overlapping_deltachi2_plot(model_data_list, save_path=None):
    """
    Overlay Delta-chi2 grid contours from all models onto a single axis.
    Axis limits are computed from the parameter GRIDS (and best-fit points),
    not from the drawn contour geometry, so every model's full contour set
    is guaranteed to be visible even if a contour doesn't close or a level
    produces no line segments.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

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

        # Filled contours, drawn tier-by-tier to control alpha blending precisely
        for i in range(3):
            ax.contourf(
                plot_X, plot_Y, plot_Z,
                levels=[LEVELS[i], LEVELS[i + 1]],
                colors=[color], alpha=TIER_ALPHAS[i], zorder=10 - i,
            )

        # Solid outline borders + inline sigma labels
        cs = ax.contour(
            plot_X, plot_Y, plot_Z, levels=LEVELS[1:],
            colors=[color], linewidths=1.5, zorder=15,
        )
        ax.clabel(cs, cs.levels, inline=True, fmt=LEVEL_LABELS, fontsize=10)

        # Bounds come from the GRID itself (nan-safe), not the contour paths,
        # so the full contour set stays inside the final view regardless of
        # whether every level actually closes within this grid.
        min_x = min(min_x, np.nanmin(plot_X))
        max_x = max(max_x, np.nanmax(plot_X))
        min_y = min(min_y, np.nanmin(plot_Y))
        max_y = max(max_y, np.nanmax(plot_Y))

        # Best-fit marker
        if plot_best_X is not None and plot_best_Y is not None:
            marker = model_data.get('marker', '*')
            ax.plot(
                plot_best_X, plot_best_Y, marker=marker, markersize=14,
                color=color, markeredgecolor='white', markeredgewidth=1.5,
                zorder=20,
            )
            min_x, max_x = min(min_x, plot_best_X), max(max_x, plot_best_X)
            min_y, max_y = min(min_y, plot_best_Y), max(max_y, plot_best_Y)

        # Proxy artist for the legend
        ax.plot([], [], color=color, linewidth=2, label=label)

    # Literature reference bands; also fold into the y-bounds
    for name, (H0, err) in LITERATURE.items():
        ax.axhline(H0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, zorder=1)
        ax.axhspan(H0 - err, H0 + err, color='gray', alpha=0.08, zorder=0)
        min_y = min(min_y, H0 - err)
        max_y = max(max_y, H0 + err)

    # Apply padded limits
    if min_x != float('inf'):
        x_pad = (max_x - min_x) * 0.08
        y_pad = (max_y - min_y) * 0.08
        ax.set_xlim(min_x - x_pad, max_x + x_pad)
        ax.set_ylim(min_y - y_pad, max_y + y_pad)
    else:
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(55, 80)

    # Place literature labels using the FINAL (padded) x-limit, so they sit
    # correctly regardless of how the bounds were derived
    x_lo, x_hi = ax.get_xlim()
    for name, (H0, _err) in LITERATURE.items():
        ax.text(x_hi, H0, f" {name}", va='center', ha='right', fontsize=10, alpha=0.7)

    # Formatting
    ax.set_xlabel(r'$\Omega_{m,0}$', fontsize=16)
    ax.set_ylabel(r'$H_0$', fontsize=16)
    ax.set_title(r'Shared $\Delta\chi^2$ Confidence Contours Comparison', fontsize=18)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9, edgecolor='black')
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'Overlapping_DeltaChi2_Contours.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved overlapping grid-based contour plot to: {save_path}")
    if min_x != float('inf'):
        print(
            f"  > Axis scaled to full grids: "
            f"x=({x_lo:.2f}, {x_hi:.2f}), y=({ax.get_ylim()[0]:.2f}, {ax.get_ylim()[1]:.2f})"
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
    print(f"{'Model':<20} {'H0':<12} {'Ωm':<12} {'Grid Status':<10}")
    print("-" * 56)
    for model_data in model_data_list:
        H0, Om = model_data['best_fit'] if model_data['best_fit'] else ('N/A', 'N/A')
        grid_status = "OK" if model_data.get('grid_Z') is not None else "Missing"
        if isinstance(H0, float):
            print(f"{model_data['label']:<20} {H0:<12.2f} {Om:<12.4f} {grid_status:<10}")
        else:
            print(f"{model_data['label']:<20} {H0:<12} {Om:<12} {grid_status:<10}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("MODEL CONTOUR COMPARISON AND VISUALIZATION")
    print("=" * 70)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    setup_matplotlib()

    model_data_list = []
    for model_config in MODELS:
        print(f"\nProcessing: {model_config['label']}")
        model_data_list.append(load_model_data(model_config))

    print_summary(model_data_list)

    print("\n" + "=" * 70)
    print("Creating comparison plot...")
    create_overlapping_deltachi2_plot(model_data_list)

    print("\n" + "=" * 70)
    print("Done! Results saved in:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()