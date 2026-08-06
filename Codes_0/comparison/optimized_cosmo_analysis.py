#!/usr/bin/env python3
"""
optimized_cosmo_analysis.py
===========================
Merged and optimized script combining the best features of:
1. Directory structure inspection and diagnostic checking.
2. Robust multi-pattern parameter extraction (Regex for H0 and Om).
3. Grid orientation auto-detection and masked grid processing.
4. Publication-quality overlapping contour visualization (1σ, 2σ, 3σ),
   with optional Literature Reference Bands (Planck 2018, SH0ES 2022).
5. A separate, styled Parameter Summary Table figure.

Two output figures are produced, kept fully independent so either can be
regenerated, resized, or dropped into a paper/slide without dragging the
other along:
  - Comparison_all_cosmological_models.png  (contours only)
  - Parameter_Summary_Table.png             (best-fit table only)

Usage:
    python optimized_cosmo_analysis.py [--base-dir PATH] [--check-only] [--no-literature] [--jobs N]
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

warnings.filterwarnings("ignore")

log = logging.getLogger("cosmo_analysis")

# =============================================================================
# GLOBAL CONFIGURATION & DEFAULTS
# =============================================================================

DEFAULT_BASE_DIR = (
    "/home/aristeidismp/Desktop/Aristeidis_Michailis_Patselis/"
    "Academia/Patra-Physics/Traineeship/Codes_0"
)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    folder: str
    label: str
    color: str
    marker: str
    contour_file: str
    summary_file: str
    subfolder: str = ""


MODELS: tuple[ModelConfig, ...] = (
    ModelConfig(
        name="lcdm", folder="model_lcdm", subfolder="results",
        label=r"$\Lambda$CDM", color="#DC143C", marker="*",
        contour_file="contour_H0_Om_lcdm.npy", summary_file="lcdm_fit_results.txt",
    ),
    ModelConfig(
        name="lcdm_hdot", folder="model_a", subfolder="results",
        label=r"$\Lambda$CDM$+\dot{H}$", color="#FF8C00", marker="o",
        contour_file="contour_H0_Om_hdot_alpha.npy", summary_file="fit_summary.txt",
    ),
    ModelConfig(
        name="bh4_hdot", folder="model_delta", subfolder="delta_lcdm_fit/results",
        label=r"$bH^4+\dot{H}$", color="#2E8B57", marker="^",
        contour_file="contour_H0_Om_delta_free.npy", summary_file="fit_summary.txt",
    ),
    ModelConfig(
        name="bhdelta", folder="model_delta4", subfolder="results_delta4",
        label=r"$bH^\delta$", color="#4169E1", marker="s",
        contour_file="contour_H0_Om_delta4.npy", summary_file="fit_summary_delta4.txt",
    ),
    ModelConfig(
        name="bhdelta_hdot", folder="model_delta4_a", subfolder="results_delta4_alpha",
        label=r"$bH^\delta+\dot{H}$", color="#800080", marker="D",
        contour_file="contour_H0_Om_delta4_alpha.npy", summary_file="fit_summary_delta4_alpha.txt",
    ),
    ModelConfig(
        name="delta_alpha", folder="model_delta_a", subfolder="results_delta_alpha",
        label=r"$\delta, \alpha$ free", color="#8C564B", marker="v",
        contour_file="contour_H0_Om_delta_alpha_free.npy", summary_file="fit_summary_delta_alpha.txt",
    ),
)


@dataclass(frozen=True)
class LitValue:
    mean: float
    err: float


LITERATURE: dict[str, LitValue] = {
    "Planck 2018": LitValue(67.4, 0.5),
    "SH0ES 2022": LitValue(73.04, 1.04),
}

# Delta-chi2 confidence levels and transparency tiers
LEVELS = [0.0, 2.30, 6.18, 11.83]
LEVEL_LABELS = {2.30: r"$1\sigma$", 6.18: r"$2\sigma$", 11.83: r"$3\sigma$"}
TIER_ALPHAS = [0.35, 0.20, 0.08]
CONTOUR_LINESTYLES = ["solid", "dashed", "dotted"]

# Precompiled regex patterns (built once at import time, not per-call).
H0_PATTERNS = [
    re.compile(p) for p in (
        r"H0\s*=\s*([0-9.]+)", r"H_0\s*=\s*([0-9.]+)",
        r"best_H0\s*=\s*([0-9.]+)", r"h0\s*=\s*([0-9.]+)", r"H0\s+([0-9.]+)",
    )
]
OM_PATTERNS = [
    re.compile(p) for p in (
        r"Om\s*=\s*([0-9.]+)", r"Omega_m\s*=\s*([0-9.]+)",
        r"best_Om\s*=\s*([0-9.]+)", r"Omega_m\s+([0-9.]+)", r"Om\s+([0-9.]+)",
    )
]

RESULT_FILE_HINTS = ("summary", "fit", "contour")

# =============================================================================
# SMALL UTILITIES
# =============================================================================


class _BBox:
    """Accumulates a 2D bounding box across multiple contributions."""

    __slots__ = ("min_x", "max_x", "min_y", "max_y")

    def __init__(self) -> None:
        self.min_x = self.min_y = float("inf")
        self.max_x = self.max_y = float("-inf")

    def add_array(self, x_arr: np.ndarray, y_arr: np.ndarray) -> None:
        self.min_x = min(self.min_x, np.nanmin(x_arr))
        self.max_x = max(self.max_x, np.nanmax(x_arr))
        self.min_y = min(self.min_y, np.nanmin(y_arr))
        self.max_y = max(self.max_y, np.nanmax(y_arr))

    def add_point(self, x: Optional[float], y: Optional[float]) -> None:
        if x is None or y is None:
            return
        self.min_x, self.max_x = min(self.min_x, x), max(self.max_x, x)
        self.min_y, self.max_y = min(self.min_y, y), max(self.max_y, y)

    @property
    def is_empty(self) -> bool:
        return self.min_x == float("inf")

    def apply_limits(self, ax, pad_frac: float = 0.08,
                      x_bounds=(0.0, 1.0), y_bounds=(40.0, 100.0)) -> None:
        """Set axis limits to comfortably contain all accumulated data.

        `x_bounds`/`y_bounds` are a *floor*, not a clamp: they keep the view
        from zooming in absurdly tight when every model happens to sit in a
        small region, but they never truncate real data. Previously this used
        `min(x_bounds[1], ...)` / `max(x_bounds[0], ...)`, which silently cut
        off contours that extended past Om=1.0 or below Om=0.0 -- exactly the
        clipped edge seen when a model's contour reached the plot border.
        """
        if self.is_empty:
            ax.set_xlim(0.1, 0.6)
            ax.set_ylim(55, 85)
            return
        x_pad = (self.max_x - self.min_x) * pad_frac
        y_pad = (self.max_y - self.min_y) * pad_frac
        ax.set_xlim(min(x_bounds[0], self.min_x - x_pad), max(x_bounds[1], self.max_x + x_pad))
        ax.set_ylim(min(y_bounds[0], self.min_y - y_pad), max(y_bounds[1], self.max_y + y_pad))


def get_base_dir(cli_arg: Optional[str] = None) -> Path:
    """Resolve BASE_DIR: CLI arg > Environment Variable > Hardcoded Default."""
    return Path(cli_arg or os.environ.get("COSMO_BASE_DIR") or DEFAULT_BASE_DIR)


def setup_matplotlib() -> None:
    """Configure matplotlib with LaTeX rendering or fallback gracefully to mathtext."""
    try:
        rc("text", usetex=True)
        rc("font", family="serif")
        log.info("✓ Matplotlib initialized with LaTeX rendering.")
    except Exception:
        rc("text", usetex=False)
        rc("font", family="DejaVu Sans")
        log.info("! LaTeX unavailable. Matplotlib initialized with mathtext.")
    finally:
        rc("font", size=12)
        rc("axes", labelsize=14, titlesize=16)
        rc("legend", fontsize=10)
        rc("figure", dpi=300)


def lighten_color(hex_color: str, factor: float = 0.75) -> str:
    """Blend a hex color towards white for table cell background highlights."""
    hex_color = hex_color.lstrip("#")
    rgb = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    light_rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*light_rgb)


def get_full_path(model_config: ModelConfig, base_dir: Path) -> Path:
    """Resolve the absolute path to a model's target results folder."""
    folder_path = base_dir / model_config.folder
    return folder_path / model_config.subfolder if model_config.subfolder else folder_path


def _find_file_by_hint(folder_path: Path, hint: str, ext: str = ".txt") -> Optional[Path]:
    """Search recursively inside folder_path for a file matching a keyword hint."""
    if not folder_path.exists():
        return None
    for candidate in folder_path.rglob(f"*{ext}"):
        if hint in candidate.name.lower():
            return candidate
    return None

# =============================================================================
# DATA EXTRACTION & ORIENTATION
# =============================================================================


def _first_match(patterns: list[re.Pattern], content: str) -> Optional[float]:
    """Return the float captured by the first regex (in priority order) that matches."""
    for pattern in patterns:
        m = pattern.search(content)
        if m:
            return float(m.group(1))
    return None


def extract_best_fit_from_summary(
    model_config: ModelConfig, base_dir: Path
) -> tuple[Optional[float], Optional[float]]:
    """Extract H0 and Om using comprehensive regex patterns across summary files."""
    folder_path = get_full_path(model_config, base_dir)
    if not folder_path.exists():
        return None, None

    filepath = folder_path / model_config.summary_file
    if not filepath.exists():
        filepath = _find_file_by_hint(folder_path, "summary", ".txt") or filepath
    if not filepath.exists():
        return None, None

    try:
        content = filepath.read_text()
    except OSError:
        return None, None

    return _first_match(H0_PATTERNS, content), _first_match(OM_PATTERNS, content)


def load_contour_data(model_config: ModelConfig, base_dir: Path) -> Optional[dict]:
    """Load Delta-chi2 grid file dictionary (.npy) supporting multiple dictionary key formats."""
    folder_path = get_full_path(model_config, base_dir)
    if not folder_path.exists():
        log.warning("    ! Folder does not exist: %s", folder_path)
        return None

    if model_config.contour_file:
        filepath = folder_path / model_config.contour_file
        if filepath.exists():
            try:
                return np.load(filepath, allow_pickle=True).item()
            except Exception as exc:
                # Previously swallowed silently -> a broken/mis-pickled file looked
                # identical to a missing one, with no way to tell which happened.
                log.warning("    ! Failed to load %s: %s", filepath.name, exc)

    # Fallback to finding any valid contour file in folder.
    for candidate in folder_path.rglob("contour_*.npy"):
        try:
            return np.load(candidate, allow_pickle=True).item()
        except Exception as exc:
            log.warning("    ! Failed to load fallback %s: %s", candidate.name, exc)
            continue
    return None


def _orient_grid(X, Y, Z, best_H0, best_Om):
    """
    Ensure Om is placed on X-axis (Om ~ 0-1) and H0 on Y-axis (H0 ~ 50-100).
    Uses data magnitude inspection to correct transposed inputs automatically.
    """
    x_mean = np.nanmean(X)
    y_mean = np.nanmean(Y)

    if x_mean > 10 and y_mean <= 10:  # X is H0, Y is Om -> swap
        plot_X, plot_Y = Y, X
        plot_Z = Z.T if Z.shape[0] != Z.shape[1] else Z
    else:
        plot_X, plot_Y = X, Y
        plot_Z = Z

    return plot_X, plot_Y, plot_Z, best_Om, best_H0


def _renormalize_delta_chi2(Z: np.ndarray) -> np.ndarray:
    """Shift a Delta-chi2 grid so its minimum sits exactly at 0.

    Without this, `contourf(levels=[0.0, 2.30])` can fail to fill anything
    near the best-fit point whenever the grid's discrete minimum is offset
    above 0 (e.g. the printed best-fit came from a continuous optimizer that
    found a lower chi2 than any single grid cell reached, or delta_chi2 was
    saved relative to a chi2_min computed separately from the grid itself).
    That offset is exactly what produces a white "cutout" instead of a filled
    1-sigma region: real data, not a plotting bug — but this neutralizes it.
    """
    z_min = np.nanmin(Z)
    if not np.isfinite(z_min) or z_min == 0.0:
        return Z
    return Z - z_min


def _smooth_grid(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                  upsample: int = 1, sigma: float = 0.0):
    """Optionally upsample + Gaussian-smooth a coarse grid for less jagged contours.

    Pure cosmetic fix for angular/kinked contour lines that come from an
    underlying Om/H0 scan grid that's too coarse for smooth ellipses. No-op
    when upsample<=1 and sigma<=0, and degrades gracefully if scipy isn't
    installed. The real fix is a finer scan grid at generation time; this is
    a stopgap for display purposes only.
    """
    if upsample <= 1 and sigma <= 0:
        return X, Y, Z
    try:
        from scipy.ndimage import zoom, gaussian_filter
    except ImportError:
        log.warning("    ! scipy not available; skipping grid smoothing/upsampling.")
        return X, Y, Z

    if upsample > 1:
        X = zoom(X, upsample, order=1)
        Y = zoom(Y, upsample, order=1)
        Z = zoom(np.nan_to_num(Z, nan=np.nanmax(Z)), upsample, order=3)
    if sigma > 0:
        Z = gaussian_filter(Z, sigma=sigma)
    return X, Y, Z


def load_model_data(model_config: ModelConfig, base_dir: Path) -> dict:
    """Load best-fit and contour grid data for a given model config."""
    log.info("  Processing: %s", model_config.label)
    H0, Om = extract_best_fit_from_summary(model_config, base_dir)
    best_fit = (H0, Om) if (H0 is not None and Om is not None) else None

    if best_fit:
        log.info("    ✓ Best fit found: H0 = %.2f, Ωm = %.4f", H0, Om)
    else:
        log.info("    ! Best-fit values missing")

    data = load_contour_data(model_config, base_dir)
    grid_X = grid_Y = grid_Z = None

    if data is not None:
        if all(k in data for k in ("X", "Y", "delta_chi2")):
            grid_X, grid_Y, grid_Z = data["X"], data["Y"], data["delta_chi2"]
            log.info("    ✓ Grid loaded (X, Y, delta_chi2)")
        elif all(k in data for k in ("xx", "yy", "delta_chisq")):
            grid_X, grid_Y, grid_Z = data["xx"], data["yy"], data["delta_chisq"]
            log.info("    ✓ Grid loaded (xx, yy, delta_chisq)")
        else:
            log.info("    ! Unexpected key structure: %s", list(data.keys()))
    else:
        log.info("    ! Contour .npy data missing")

    return {
        "name": model_config.name,
        "label": model_config.label,
        "color": model_config.color,
        "marker": model_config.marker,
        "best_fit": best_fit,
        "grid_X": grid_X,
        "grid_Y": grid_Y,
        "grid_Z": grid_Z,
    }


def load_all_model_data(models: tuple[ModelConfig, ...], base_dir: Path, jobs: int = 4) -> list[dict]:
    """Load every model's data in parallel (I/O-bound: reading .txt/.npy files)."""
    if jobs <= 1:
        return [load_model_data(m, base_dir) for m in models]

    # Preserve MODELS ordering regardless of completion order.
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda m: load_model_data(m, base_dir), models))

# =============================================================================
# DIAGNOSTIC CHECKER
# =============================================================================


def check_directory_structure(base_dir: Path) -> bool:
    """Print a complete diagnostic tree of directories, summaries, and contour files.

    Walks the tree exactly once (previously this scanned the full tree three
    separate times: once for the top-level folder listing, once for all
    summary files, once for all contour files).
    """
    log.info("\n%s", "=" * 70)
    log.info("DIRECTORY STRUCTURE CHECK")
    log.info("=" * 70)
    log.info("Base Directory: %s\n", base_dir)

    if not base_dir.exists():
        log.error("ERROR: Target base directory does not exist: %s", base_dir)
        return False

    per_subdir_files: dict[str, list[str]] = {}
    all_summary_files: list[str] = []
    all_contour_files: list[str] = []

    top_level_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())

    for top_dir in top_level_dirs:
        matches = []
        for f in top_dir.rglob("*"):
            if not f.is_file():
                continue
            name_lower = f.name.lower()
            if f.suffix in (".txt", ".npy") and any(k in name_lower for k in RESULT_FILE_HINTS):
                matches.append(str(f.relative_to(top_dir)))
        per_subdir_files[top_dir.name] = matches

    for f in base_dir.rglob("*.txt"):
        if "summary" in f.name.lower():
            all_summary_files.append(str(f.relative_to(base_dir)))

    for f in base_dir.rglob("*.npy"):
        if "contour" in f.name.lower():
            all_contour_files.append(str(f.relative_to(base_dir)))

    log.info("Existing sub-directories:")
    for name, matches in per_subdir_files.items():
        log.info("  📁 %s", name)
        if matches:
            for rf in matches[:4]:
                log.info("       - %s", rf)
            if len(matches) > 4:
                log.info("       ... and %d more", len(matches) - 4)
        else:
            log.info("       (no relevant files)")

    log.info("\nSummary files (.txt):")
    for rf in all_summary_files:
        log.info("  - %s", rf)

    log.info("\nContour grid files (.npy):")
    for rf in all_contour_files:
        log.info("  - %s", rf)

    log.info("%s\n", "=" * 70)
    return True

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================


def _draw_model_contours(
    ax, model_data: dict, bbox: _BBox, use_linestyles: bool = False,
    grid_upsample: int = 1, grid_smooth_sigma: float = 0.0,
) -> None:
    """Draw the tiered contour fill + outline + best-fit marker for one model onto `ax`.

    Shared by `create_comparison_plot` (the only remaining caller); kept as its
    own function since the table plot needs the best-fit values but not the
    contour grid itself.
    """
    Z = model_data.get("grid_Z")
    if Z is None:
        return

    X, Y = model_data["grid_X"], model_data["grid_Y"]
    best_H0, best_Om = model_data.get("best_fit", (None, None))

    plot_X, plot_Y, plot_Z, plot_best_X, plot_best_Y = _orient_grid(X, Y, Z, best_H0, best_Om)
    plot_Z = _renormalize_delta_chi2(plot_Z)
    plot_X, plot_Y, plot_Z = _smooth_grid(plot_X, plot_Y, plot_Z, grid_upsample, grid_smooth_sigma)
    plot_Z = np.ma.masked_invalid(plot_Z)

    color = model_data.get("color", "gray")
    label = model_data["label"]
    marker = model_data.get("marker", "*")

    # Tiered alpha fill for 1σ, 2σ, 3σ.
    for i in range(len(LEVELS) - 1):
        ax.contourf(
            plot_X, plot_Y, plot_Z,
            levels=[LEVELS[i], LEVELS[i + 1]],
            colors=[color], alpha=TIER_ALPHAS[i], zorder=10 - i,
        )

    contour_kwargs = dict(colors=[color], linewidths=1.5, zorder=15)
    if use_linestyles:
        contour_kwargs["linestyles"] = CONTOUR_LINESTYLES[: len(LEVELS) - 1]
    ax.contour(plot_X, plot_Y, plot_Z, levels=LEVELS[1:], **contour_kwargs)

    bbox.add_array(plot_X, plot_Y)

    if plot_best_X is not None and plot_best_Y is not None:
        ax.plot(
            plot_best_X, plot_best_Y, marker=marker, markersize=12,
            color=color, markeredgecolor="white", markeredgewidth=1.2, zorder=25,
        )
        bbox.add_point(plot_best_X, plot_best_Y)

    ax.plot([], [], color=color, linewidth=2.5, label=label)


def _add_literature_bands(ax, bbox: _BBox, annotate: bool = False) -> None:
    """Draw horizontal Planck/SH0ES reference bands and optionally label them."""
    for name, lit in LITERATURE.items():
        ax.axhline(lit.mean, color="gray", linestyle=":", alpha=0.5, linewidth=1.5, zorder=1)
        ax.axhspan(lit.mean - lit.err, lit.mean + lit.err, color="gray", alpha=0.08, zorder=0)
        bbox.min_y = min(bbox.min_y, lit.mean - lit.err)
        bbox.max_y = max(bbox.max_y, lit.mean + lit.err)

    if annotate and not bbox.is_empty:
        x_hi = bbox.max_x + (bbox.max_x - bbox.min_x) * 0.08
        for name, lit in LITERATURE.items():
            ax.text(x_hi, lit.mean, f" {name}", va="center", ha="right", fontsize=9, alpha=0.7)


def create_comparison_plot(
    model_data_list: list[dict], output_dir: Path, show_literature: bool = False,
    grid_upsample: int = 1, grid_smooth_sigma: float = 0.0,
) -> None:
    """Generate a clean, stand-alone overlapping contour plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    bbox = _BBox()

    for model_data in model_data_list:
        _draw_model_contours(
            ax, model_data, bbox, use_linestyles=True,
            grid_upsample=grid_upsample, grid_smooth_sigma=grid_smooth_sigma,
        )

    if show_literature:
        _add_literature_bands(ax, bbox, annotate=False)

    bbox.apply_limits(ax)

    ax.set_xlabel(r"$\Omega_{m,0}$", fontsize=14)
    ax.set_ylabel(r"$H_0$ [km/s/Mpc]", fontsize=14)
    ax.set_title("Comparison of all cosmological models", fontsize=16, pad=12)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95, edgecolor="gray")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    plt.tight_layout()
    save_path = output_dir / "Comparison_all_cosmological_models.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    log.info("✓ Comparison plot saved to: %s", save_path)
    plt.close(fig)


def _collect_table_rows(model_data_list: list[dict]) -> list[dict]:
    """Pull (label, Om, H0, color) rows for every model that has a best fit."""
    rows = []
    for model_data in model_data_list:
        best_fit = model_data.get("best_fit")
        if not best_fit:
            continue
        H0, Om = best_fit
        rows.append({
            "model": model_data["label"],
            "Om": Om,
            "H0": H0,
            "color": model_data["color"],
        })
    return rows


def create_table_plot(model_data_list: list[dict], output_dir: Path) -> None:
    """Generate a stand-alone, styled Parameter Summary Table figure (no contours)."""
    table_data = _collect_table_rows(model_data_list)

    if not table_data:
        log.warning("! No best-fit data available; skipping table plot.")
        return

    # Scale figure height gently with row count so the table doesn't look
    # cramped with few models or absurdly tall with many.
    fig_height = max(1.8, 0.5 + 0.5 * len(table_data))
    fig, ax_table = plt.subplots(figsize=(8, fig_height))
    ax_table.axis("off")
    ax_table.set_position([0.05, 0.05, 0.9, 0.8])

    table_headers = ["Model", r"$\Omega_{m,0}$", r"$H_0$ [km/s/Mpc]"]
    cell_data = [[r["model"], f"{r['Om']:.4f}", f"{r['H0']:.2f}"] for r in table_data]

    table = ax_table.table(
        cellText=cell_data,
        colLabels=table_headers,
        loc="center",
        cellLoc="center",
        colWidths=[0.38, 0.28, 0.34],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for j in range(len(table_headers)):
        cell = table[(0, j)]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2C3E50")

    for i, row in enumerate(table_data):
        light_bg = lighten_color(row["color"], factor=0.85)
        for j in range(len(table_headers)):
            cell = table[(i + 1, j)]
            cell.set_facecolor(light_bg)
            cell.set_edgecolor("#DDDDDD")

    ax_table.set_title("Best-fit Parameters", fontsize=15, fontweight="bold", pad=10)

    save_path = output_dir / "Parameter_Summary_Table.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    log.info("✓ Parameter summary table saved to: %s", save_path)
    plt.close(fig)


def print_summary(model_data_list: list[dict]) -> None:
    """Print clean summary table of best-fit results to stdout."""
    lines = [
        "\n" + "=" * 70,
        "BEST-FIT PARAMETER SUMMARY",
        "=" * 70,
        f"{'Model':<25} {'H0 [km/s/Mpc]':<16} {'Ωm,0':<14} {'Grid Status':<10}",
        "-" * 68,
    ]
    for md in model_data_list:
        H0, Om = md["best_fit"] if md["best_fit"] else ("N/A", "N/A")
        grid_status = "OK" if md.get("grid_Z") is not None else "Missing"
        if isinstance(H0, float):
            lines.append(f"{md['label']:<25} {H0:<16.2f} {Om:<14.4f} {grid_status:<10}")
        else:
            lines.append(f"{md['label']:<25} {H0:<16} {Om:<14} {grid_status:<10}")
    lines.append("=" * 70 + "\n")
    log.info("\n".join(lines))


def report_incomplete_models(model_data_list: list[dict]) -> None:
    """Loudly flag any model missing a contour grid or best-fit, at WARNING level.

    This is the fix for models silently vanishing from the final plot: a
    model with grid_Z=None is skipped by `_draw_model_contours` with no
    further comment, so without this explicit call-out it's easy to end up
    with (say) 5 of 6 legend entries and not notice why.
    """
    missing = [md for md in model_data_list if md.get("grid_Z") is None or not md.get("best_fit")]
    if not missing:
        return
    log.warning("\n%s", "!" * 70)
    log.warning("WARNING: %d model(s) will NOT appear (fully or partially) in the plots:", len(missing))
    for md in missing:
        reasons = []
        if md.get("grid_Z") is None:
            reasons.append("no contour grid loaded -> will be absent from the contour plot")
        if not md.get("best_fit"):
            reasons.append("no best-fit values parsed -> will be absent from the table")
        log.warning("  - %-25s %s", md["label"], "; ".join(reasons))
    log.warning("Re-run with -v to see per-file diagnostics (which file was searched, why it failed).")
    log.warning("%s\n", "!" * 70)

# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimized Cosmological Model Comparison & Contour Plotter."
    )
    parser.add_argument("--base-dir", default=None, help="Override base directory path for model files.")
    parser.add_argument("--check-only", action="store_true",
                         help="Run directory structure inspection and exit without plotting.")
    parser.add_argument("--no-literature", action="store_true",
                         help="Disable Planck/SH0ES literature reference bands in plots.")
    parser.add_argument("--jobs", type=int, default=4,
                         help="Thread-pool size for parallel model data loading (default: 4).")
    parser.add_argument("--grid-upsample", type=int, default=1,
                         help="Upsample factor for contour grids before plotting (e.g. 3). "
                              "Fixes jagged/angular contour edges caused by a coarse Om/H0 scan "
                              "grid. Requires scipy. Default 1 (no upsampling).")
    parser.add_argument("--grid-smooth-sigma", type=float, default=0.0,
                         help="Gaussian smoothing sigma (in grid cells) applied to contour grids. "
                              "Cosmetic only -- the real fix for jagged contours is a finer scan "
                              "grid at generation time. Requires scipy. Default 0.0 (off).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug-level logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only log warnings/errors.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)

    base_dir = get_base_dir(args.base_dir)
    output_dir = base_dir / "comparison" / "results_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("COSMOLOGICAL MODEL ANALYSIS & VISUALIZATION PIPELINE")
    log.info("=" * 70)
    log.info("Base Directory  : %s", base_dir)
    log.info("Output Directory: %s\n", output_dir)

    # Step 1: Run directory check.
    check_directory_structure(base_dir)
    if args.check_only:
        log.info("Flag --check-only provided. Exiting.")
        return

    # Step 2: Initialize plotting settings.
    setup_matplotlib()

    # Step 3: Load model data (parallelized I/O).
    log.info("\nLoading cosmological model datasets...")
    model_data_list = load_all_model_data(MODELS, base_dir, jobs=args.jobs)

    # Step 4: Display parameters summary in console, and loudly flag any gaps.
    print_summary(model_data_list)
    report_incomplete_models(model_data_list)

    # Step 5: Render both visualization figures.
    log.info("Generating figures...")
    create_comparison_plot(
        model_data_list, output_dir, show_literature=not args.no_literature,
        grid_upsample=args.grid_upsample, grid_smooth_sigma=args.grid_smooth_sigma,
    )
    create_table_plot(model_data_list, output_dir)

    log.info("\n%s", "=" * 70)
    log.info("Pipeline Execution Complete!")
    log.info("Results saved in: %s", output_dir)
    log.info("=" * 70)


if __name__ == "__main__":
    main()