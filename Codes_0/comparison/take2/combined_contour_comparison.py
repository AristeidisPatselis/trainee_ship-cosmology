#!/usr/bin/env python3
"""
combined_contour_comparison.py
==============================
Single script that:
  1. Inspects the directory structure and locates model results / contour files.
  2. Loads raw Δχ² grids + best-fit points for every model.
  3. Produces a publication-style overlapping contour plot that matches the
     visual style of the reference figure:
       - Title: "Comparison of all cosmological models"
       - Matching model labels & colours
       - Filled 1σ / 2σ regions with solid (1σ) + dashed (2σ) outlines
       - Star markers for best-fit points
       - No literature bands
       - Automatic axis limits that keep every contour fully visible
"""

import os
import sys
import argparse
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_BASE_DIR = (
    "/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/"
    "Academia/Patra-Physics/Traineeship/Codes_0"
)


def get_base_dir():
    """Resolve BASE_DIR: CLI --base-dir > env COSMO_BASE_DIR > hardcoded default."""
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

# Model configurations – labels & colours chosen to reproduce the reference plot.
# Folder / file names keep the original layout so the script still finds the data.
MODELS = [
    {
        "name": "lcdm",
        "folder": "model_lcdm",
        "subfolder": "results",
        "label": r"$\Lambda$CDM",
        "color": "#d62728",          # red
        "marker": "*",
        "contour_file": "contour_H0_Om_lcdm.npy",
        "summary_file": "lcdm_fit_results.txt",
    },
    {
        "name": "hdot_alpha",
        "folder": "model_a",
        "subfolder": "results",
        "label": r"$\Lambda$CDM+$\dot{H}$",
        "color": "#ff7f0e",          # orange
        "marker": "*",
        "contour_file": "contour_H0_Om_hdot_alpha.npy",
        "summary_file": "fit_summary.txt",
    },
    {
        "name": "delta4_alpha",
        "folder": "model_delta4_a",
        "subfolder": "results_delta4_alpha",
        "label": r"$bH^{4}+\dot{H}$",
        "color": "#2ca02c",          # green
        "marker": "*",
        "contour_file": "contour_H0_Om_delta4_alpha.npy",
        "summary_file": "fit_summary_delta4_alpha.txt",
    },
    {
        "name": "delta4",
        "folder": "model_delta4",
        "subfolder": "results_delta4",
        "label": r"$bH^{5}$",
        "color": "#1f77b4",          # blue
        "marker": "*",
        "contour_file": "contour_H0_Om_delta4.npy",
        "summary_file": "fit_summary_delta4.txt",
    },
    {
        "name": "delta_alpha",
        "folder": "model_delta_a",
        "subfolder": "results_delta_alpha",
        "label": r"$bH^{5}+\dot{H}$",
        "color": "#9467bd",          # purple
        "marker": "*",
        "contour_file": "contour_H0_Om_delta_alpha_free.npy",
        "summary_file": "fit_summary_delta_alpha.txt",
    },
]

# Δχ² levels for 1σ / 2σ (we omit 3σ to keep the plot clean like the reference)
LEVELS = [0.0, 2.30, 6.18]
LEVEL_LABELS = {2.30: r"$1\sigma$", 6.18: r"$2\sigma$"}
# Fill transparencies: stronger for 1σ, lighter for 2σ
TIER_ALPHAS = [0.35, 0.18]

# =============================================================================
# DIRECTORY INSPECTION (from check_folders.py)
# =============================================================================

def check_directory_structure(base_dir):
    """Print a readable overview of model folders and result files."""
    print("=" * 70)
    print("DIRECTORY STRUCTURE CHECK")
    print("=" * 70)
    print(f"Base directory: {base_dir}")
    print()

    if not os.path.isdir(base_dir):
        print(f"  ERROR: Base directory does not exist: {base_dir}")
        print("  Use --base-dir /path/to/Codes_0 or set COSMO_BASE_DIR.")
        return

    print("All directories in BASE_DIR:")
    for item in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, item)
        if not os.path.isdir(full_path):
            continue
        print(f"  📁 {item}")
        result_files = []
        for root, _dirs, files in os.walk(full_path):
            for f in files:
                if f.endswith((".txt", ".npy")) and any(
                    k in f.lower() for k in ("summary", "fit", "contour")
                ):
                    rel = os.path.relpath(os.path.join(root, f), full_path)
                    result_files.append(rel)
            if len(result_files) > 8:
                break
        if result_files:
            print("     Found result files:")
            for rf in result_files[:5]:
                print(f"       - {rf}")
            if len(result_files) > 5:
                print(f"       ... and {len(result_files) - 5} more")
        else:
            print("     No result files found")

    print()
    print("=" * 70)
    print("\nAll fit_summary / *summary*.txt files:")
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if "summary" in f.lower() and f.endswith(".txt"):
                print(f"  - {os.path.relpath(os.path.join(root, f), base_dir)}")

    print()
    print("=" * 70)
    print("\nAll contour_*.npy files:")
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if "contour" in f.lower() and f.endswith(".npy"):
                print(f"  - {os.path.relpath(os.path.join(root, f), base_dir)}")
    print("=" * 70)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_matplotlib():
    """Publication-quality defaults; fall back to mathtext if LaTeX is unavailable."""
    try:
        rc("text", usetex=True)
        rc("font", family="serif")
        print("  Using LaTeX rendering")
        return True
    except Exception:
        rc("text", usetex=False)
        rc("font", family="DejaVu Sans")
        print("  Using mathtext rendering")
        return False
    finally:
        rc("font", size=12)
        rc("axes", labelsize=14, titlesize=16)
        rc("legend", fontsize=11)
        rc("figure", dpi=300)


def get_full_path(model_config):
    folder_path = os.path.join(BASE_DIR, model_config["folder"])
    sub = model_config.get("subfolder", "")
    return os.path.join(folder_path, sub) if sub else folder_path


def _find_file_by_hint(folder_path, hint, ext=".txt"):
    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            if hint in f.lower() and f.endswith(ext):
                return os.path.join(root, f)
    return None


def extract_best_fit_from_summary(model_config):
    folder_path = get_full_path(model_config)
    if not os.path.exists(folder_path):
        return None, None

    summary_file = model_config.get("summary_file", "fit_summary.txt")
    filepath = os.path.join(folder_path, summary_file)
    if not os.path.exists(filepath):
        filepath = _find_file_by_hint(folder_path, "summary") or filepath
    if not os.path.exists(filepath):
        return None, None

    try:
        with open(filepath, "r") as fh:
            content = fh.read()
    except OSError:
        return None, None

    h0 = om = None
    for pat in (r"H0\s*=\s*([0-9.]+)", r"H_0\s*=\s*([0-9.]+)", r"best_H0\s*=\s*([0-9.]+)"):
        m = re.search(pat, content)
        if m:
            h0 = float(m.group(1))
            break
    for pat in (
        r"Om\s*=\s*([0-9.]+)",
        r"Omega_m\s*=\s*([0-9.]+)",
        r"best_Om\s*=\s*([0-9.]+)",
        r"Omega_m\s+([0-9.]+)",
    ):
        m = re.search(pat, content)
        if m:
            om = float(m.group(1))
            break
    return (h0, om) if (h0 is not None and om is not None) else (None, None)


def load_contour_data(model_config):
    folder_path = get_full_path(model_config)
    if not os.path.exists(folder_path):
        return None

    contour_file = model_config.get("contour_file")
    if contour_file:
        fp = os.path.join(folder_path, contour_file)
        if os.path.exists(fp):
            try:
                return np.load(fp, allow_pickle=True).item()
            except Exception:
                pass

    for root, _dirs, files in os.walk(folder_path):
        for f in files:
            if f.startswith("contour_") and f.endswith(".npy"):
                try:
                    return np.load(os.path.join(root, f), allow_pickle=True).item()
                except Exception:
                    continue
    return None


def _orient_grid(X, Y, Z, best_H0, best_Om):
    """
    Guarantee that Ωm is the horizontal axis and H0 the vertical axis.
    H0 is O(50-100), Ωm is O(0-1); we use the means to decide orientation.
    """
    x_mean = np.nanmean(X)
    y_mean = np.nanmean(Y)
    if x_mean > 10 and y_mean <= 10:  # currently swapped
        plot_X, plot_Y = Y, X
        plot_Z = Z.T if Z.shape[0] != Z.shape[1] else Z
    else:
        plot_X, plot_Y = X, Y
        plot_Z = Z
    return plot_X, plot_Y, plot_Z, best_Om, best_H0


# =============================================================================
# PLOTTING – styled to match the reference image
# =============================================================================

def create_overlapping_deltachi2_plot(model_data_list, save_path=None):
    """
    Overlay Δχ² contours for all models.
    Visual style follows the supplied reference figure:
      - solid outlines for 1σ, dashed for 2σ
      - translucent filled regions
      - star markers coloured like the contours
      - title and axis labels matching the photo
      - automatic limits derived from the grids themselves
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for model_data in model_data_list:
        Z = model_data.get("grid_Z")
        if Z is None:
            continue

        X, Y = model_data["grid_X"], model_data["grid_Y"]
        best_H0, best_Om = model_data.get("best_fit", (None, None))

        plot_X, plot_Y, plot_Z, plot_best_X, plot_best_Y = _orient_grid(
            X, Y, Z, best_H0, best_Om
        )
        plot_Z = np.ma.masked_invalid(plot_Z)

        color = model_data.get("color", "gray")
        label = model_data["label"]

        # --- filled regions (1σ stronger, 2σ lighter) ---
        for i in range(2):
            ax.contourf(
                plot_X,
                plot_Y,
                plot_Z,
                levels=[LEVELS[i], LEVELS[i + 1]],
                colors=[color],
                alpha=TIER_ALPHAS[i],
                zorder=10 - i,
            )

        # --- solid 1σ outline ---
        cs1 = ax.contour(
            plot_X,
            plot_Y,
            plot_Z,
            levels=[LEVELS[1]],
            colors=[color],
            linewidths=1.8,
            linestyles="solid",
            zorder=15,
        )

        # --- dashed 2σ outline ---
        cs2 = ax.contour(
            plot_X,
            plot_Y,
            plot_Z,
            levels=[LEVELS[2]],
            colors=[color],
            linewidths=1.4,
            linestyles="dashed",
            zorder=14,
        )

        # optional sigma labels (kept subtle)
        try:
            ax.clabel(cs1, cs1.levels, inline=True, fmt=LEVEL_LABELS, fontsize=9)
        except Exception:
            pass

        # grid bounds (guarantees every contour stays inside the final view)
        min_x = min(min_x, np.nanmin(plot_X))
        max_x = max(max_x, np.nanmax(plot_X))
        min_y = min(min_y, np.nanmin(plot_Y))
        max_y = max(max_y, np.nanmax(plot_Y))

        # best-fit star
        if plot_best_X is not None and plot_best_Y is not None:
            ax.plot(
                plot_best_X,
                plot_best_Y,
                marker="*",
                markersize=16,
                color=color,
                markeredgecolor="black",
                markeredgewidth=0.9,
                zorder=25,
            )
            min_x = min(min_x, plot_best_X)
            max_x = max(max_x, plot_best_X)
            min_y = min(min_y, plot_best_Y)
            max_y = max(max_y, plot_best_Y)

        # legend proxy (solid line)
        ax.plot([], [], color=color, linewidth=2.2, label=label)

    # padded limits – keep a little extra room like the reference figure
    if min_x != float("inf"):
        x_pad = (max_x - min_x) * 0.07
        y_pad = (max_y - min_y) * 0.08
        # enforce a sensible floor so the plot never looks cramped
        x_lo = max(0.05, min_x - x_pad)
        x_hi = min(1.05, max_x + x_pad)
        y_lo = max(55.0, min_y - y_pad)
        y_hi = min(90.0, max_y + y_pad)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
    else:
        ax.set_xlim(0.1, 1.0)
        ax.set_ylim(58, 87)

    # formatting that matches the reference image
    ax.set_xlabel(r"$\Omega_{m,0}$", fontsize=15)
    ax.set_ylabel(r"$H_0$ [km/s/Mpc]", fontsize=15)
    ax.set_title("Comparison of all cosmological models", fontsize=16, pad=12)
    ax.legend(
        loc="upper right",
        fontsize=11,
        framealpha=0.92,
        edgecolor="0.3",
        fancybox=True,
    )
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.grid(False)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "Comparison_all_cosmological_models.png")

    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"✓ Saved plot → {save_path}")
    if min_x != float("inf"):
        print(
            f"  Axis limits: Ωm ∈ [{ax.get_xlim()[0]:.3f}, {ax.get_xlim()[1]:.3f}], "
            f"H0 ∈ [{ax.get_ylim()[0]:.1f}, {ax.get_ylim()[1]:.1f}]"
        )
    plt.close(fig)
    return save_path


def load_model_data(model_config):
    print(f"\nProcessing: {model_config['label']}")
    H0, Om = extract_best_fit_from_summary(model_config)
    best_fit = (H0, Om) if H0 is not None and Om is not None else None
    if best_fit:
        print(f"  ✓ Best fit: H0 = {H0:.2f}, Ωm = {Om:.4f}")
    else:
        print("  ! No best-fit values found in summary")

    data = load_contour_data(model_config)
    grid_X = grid_Y = grid_Z = None
    if data is not None:
        if all(k in data for k in ("X", "Y", "delta_chi2")):
            grid_X, grid_Y, grid_Z = data["X"], data["Y"], data["delta_chi2"]
            print("  ✓ Extracted grid (X, Y, delta_chi2)")
        elif all(k in data for k in ("xx", "yy", "delta_chisq")):
            grid_X, grid_Y, grid_Z = data["xx"], data["yy"], data["delta_chisq"]
            print("  ✓ Extracted grid (xx, yy, delta_chisq)")
        else:
            print(f"  ! Unexpected keys: {list(data.keys())}")
    else:
        print("  ! No contour .npy found")

    return {
        "name": model_config["name"],
        "label": model_config["label"],
        "color": model_config.get("color", "gray"),
        "marker": model_config.get("marker", "*"),
        "best_fit": best_fit,
        "grid_X": grid_X,
        "grid_Y": grid_Y,
        "grid_Z": grid_Z,
    }


def print_summary(model_data_list):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<28} {'H0':<12} {'Ωm':<12} {'Grid':<8}")
    print("-" * 60)
    for md in model_data_list:
        H0, Om = md["best_fit"] if md["best_fit"] else ("N/A", "N/A")
        status = "OK" if md.get("grid_Z") is not None else "Missing"
        if isinstance(H0, float):
            print(f"{md['label']:<28} {H0:<12.2f} {Om:<12.4f} {status:<8}")
        else:
            print(f"{md['label']:<28} {H0:<12} {Om:<12} {status:<8}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("COMBINED COSMOLOGICAL MODEL CONTOUR COMPARISON")
    print("=" * 70)
    print(f"Base directory : {BASE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # 1. Directory inspection
    check_directory_structure(BASE_DIR)

    # 2. Matplotlib setup
    setup_matplotlib()

    # 3. Load every model
    model_data_list = [load_model_data(cfg) for cfg in MODELS]
    print_summary(model_data_list)

    # 4. Produce the comparison plot
    print("\n" + "=" * 70)
    print("Creating comparison plot (style matched to reference figure)...")
    out = create_overlapping_deltachi2_plot(model_data_list)

    print("\n" + "=" * 70)
    print("Done.")
    print(f"  Plot saved to: {out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
