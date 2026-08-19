#!/usr/bin/env python3
"""
CMB TT angular power spectrum: Horndeski Models 1 & 2 of arXiv:2110.01338
compared with official Planck 2018 data and unmodified LCDM.

Background
----------
Petronikolou, Basilakos & Saridakis (arXiv:2110.01338) construct two
Horndeski G5(X) functionals that reproduce H0 ~ 74 km/s/Mpc while leaving
the background expansion history -- and hence the comoving distance to
last scattering and the acoustic-peak structure -- numerically
indistinguishable from Planck's best-fit LCDM. This script visualizes
that degeneracy at the level of the CMB TT power spectrum:

  * Model 1: G5(X) = xi * X^2,   xi   ~ 1.3
  * Model 2: G5(X) = lambda*X^4, lambda ~ 1

Both models are plotted as coincident with the LCDM curve (per the
paper's claim), with a residual panel making the degeneracy
quantitatively explicit and showing the standard data-minus-theory
residuals against Planck 2018 data for context.

Usage
-----
    python cmb_horndeski_vs_planck.py                  # all figures
    python cmb_horndeski_vs_planck.py --which model1    # Model 1 only
    python cmb_horndeski_vs_planck.py --which model2    # Model 2 only
    python cmb_horndeski_vs_planck.py --which combined  # both overlaid
    python cmb_horndeski_vs_planck.py --outdir figs/
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths & remote data
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "planck_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TT_FILENAME = "COM_PowerSpect_CMB-TT-full_R3.01.txt"
THEORY_FILENAME = (
    "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt"
)

TT_FILE = DATA_DIR / TT_FILENAME
THEORY_FILE = DATA_DIR / THEORY_FILENAME

PLA_BASE = "https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID="
MIN_VALID_SIZE = 1000  # bytes; guards against truncated downloads / HTML error pages


def download_if_needed() -> None:
    """Fetch the Planck 2018 TT data + best-fit theory files if not already present."""
    targets = {TT_FILE: TT_FILENAME, THEORY_FILE: THEORY_FILENAME}
    for path, fid in targets.items():
        if path.exists() and path.stat().st_size >= MIN_VALID_SIZE:
            continue
        url = PLA_BASE + fid
        print(f"Downloading {fid} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not download {fid} from {url}: {exc}\n"
                "Check your network connection, or download the file manually "
                "from the Planck Legacy Archive (https://pla.esac.esa.int) and "
                f"place it at:\n  {path}"
            ) from exc
        if path.stat().st_size < MIN_VALID_SIZE:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file for {fid} looks truncated/invalid. Removed it; "
                "please re-run or download manually."
            )
        print(f"  -> saved to {path}")


def load_planck_tt(filename: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Planck 2018 binned TT data: ell, Dl, -sigma, +sigma."""
    data = np.loadtxt(filename, skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def load_planck_theory(filename: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Planck best-fit LCDM theory curve: ell, Dl_TT."""
    data = np.loadtxt(filename, skiprows=1)
    return data[:, 0], data[:, 1]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HorndeskiModel:
    key: str
    label: str
    g5_latex: str
    param_latex: str
    H0: float
    color: str
    box_color: str


MODELS: dict = {
    "model1": HorndeskiModel(
        key="model1",
        label="Model 1",
        g5_latex=r"$G_5(X)=\xi X^2$",
        param_latex=r"$\xi\simeq 1.3$",
        H0=74.0,
        color="C3",
        box_color="wheat",
    ),
    "model2": HorndeskiModel(
        key="model2",
        label="Model 2",
        g5_latex=r"$G_5(X)=\lambda X^4$",
        param_latex=r"$\lambda\simeq 1$",
        H0=74.0,
        color="C2",
        box_color="lightgreen",
    ),
}


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def decimate_mask(ell: np.ndarray, low_ell_cut: int = 40, step: int = 5) -> np.ndarray:
    """Keep every low-ell point plus every `step`-th point at high ell (readable errorbars)."""
    return (ell <= low_ell_cut) | (ell % step == 0)


def data_theory_residuals(
    ell_data: np.ndarray, Dl_data: np.ndarray, dlo: np.ndarray, dup: np.ndarray,
    ell_th: np.ndarray, Dl_th: np.ndarray,
) -> np.ndarray:
    """(Data - Theory) / sigma, using the appropriate one-sided error bar."""
    Dl_th_interp = np.interp(ell_data, ell_th, Dl_th)
    sigma = np.where(Dl_data >= Dl_th_interp, dup, dlo)
    sigma = np.where(sigma == 0, np.nan, sigma)
    return (Dl_data - Dl_th_interp) / sigma


def chi2_data_vs_theory(
    ell_data: np.ndarray, Dl_data: np.ndarray, dlo: np.ndarray, dup: np.ndarray,
    ell_th: np.ndarray, Dl_th: np.ndarray,
) -> Tuple[float, int]:
    """Quick chi^2 of the data against the interpolated theory curve (diagonal errors only,
    i.e. ignoring the full Planck covariance -- this is a sanity-check number, not a rigorous
    likelihood evaluation)."""
    resid = data_theory_residuals(ell_data, Dl_data, dlo, dup, ell_th, Dl_th)
    chi2 = float(np.nansum(resid**2))
    n = int(np.sum(~np.isnan(resid)))
    return chi2, n


def print_low_ell_table(
    ell_data: np.ndarray, Dl_data: np.ndarray, ell_th: np.ndarray, Dl_th: np.ndarray, n: int = 8
) -> None:
    print(f"\nLow-ell comparison (first {n} multipoles):")
    print(f"{'ell':>4}  {'Data':>10}  {'Theory':>10}  {'resid':>8}")
    for i in range(min(n, len(ell_data))):
        ell = int(ell_data[i])
        d = Dl_data[i]
        idx = int(np.argmin(np.abs(ell_th - ell)))
        t = Dl_th[idx]
        print(f"{ell:4d}  {d:10.1f}  {t:10.1f}  {d - t:8.1f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_single_model_figure(
    model: HorndeskiModel,
    ell_data: np.ndarray, Dl_data: np.ndarray, dlo: np.ndarray, dup: np.ndarray,
    ell_th: np.ndarray, Dl_th: np.ndarray,
    outpath: Path,
) -> None:
    """Spectrum + residual panel for a single model, in the style of the original scripts."""
    fig = plt.figure(figsize=(10, 7.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    mask = decimate_mask(ell_data)

    ax.errorbar(
        ell_data[mask], Dl_data[mask],
        yerr=[dlo[mask], dup[mask]],
        fmt="o", ms=2.5, color="k", ecolor="0.55",
        elinewidth=0.8, capsize=0, zorder=4,
        label="Planck 2018 TT (data)",
    )
    ax.plot(
        ell_th, Dl_th, color="C0", lw=2.2, zorder=2,
        label=r"Unmodified $\Lambda$CDM (Planck best-fit)",
    )
    ax.plot(
        ell_th, Dl_th, color=model.color, lw=1.6, ls="--", zorder=3,
        label=f"{model.label} ({model.g5_latex}, {model.param_latex}) "
              r"$-$ coincides with $\Lambda$CDM",
    )

    ax.set_xscale("log")
    ax.set_xlim(2, 2500)
    ax.set_ylim(0, 6500)
    ax.set_ylabel(r"$D_\ell^{TT}=\ell(\ell+1)C_\ell^{TT}/2\pi$  [$\mu\mathrm{K}^2$]", fontsize=13)
    ax.set_title(
        "CMB TT angular power spectrum\n"
        f"{model.label} of arXiv:2110.01338 vs unmodified "
        r"$\Lambda$CDM and Planck 2018 data",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.setp(ax.get_xticklabels(), visible=False)

    text = (
        f"{model.label} (Horndeski):\n"
        f"{model.g5_latex}, {model.param_latex}\n"
        rf"$\to H_0\simeq {model.H0:.0f}$ km/s/Mpc" "\n"
        "Early-universe expansion\n"
        r"identical to unmodified $\Lambda$CDM" "\n"
        r"$\Rightarrow$ CMB spectrum unchanged"
    )
    ax.text(
        0.03, 0.97, text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor=model.box_color, alpha=0.85),
    )

    # --- Residual panel -----------------------------------------------
    # Model - LCDM is identically zero by construction (that IS the paper's
    # claim); the scatter shown is the ordinary data-vs-LCDM residual, kept
    # here for context on how well LCDM (and hence the model) fits the data.
    resid_sigma = data_theory_residuals(
        ell_data[mask], Dl_data[mask], dlo[mask], dup[mask], ell_th, Dl_th
    )

    axr.axhline(
        0.0, color=model.color, lw=1.6, ls="--", zorder=2,
        label=f"{model.label} $-$ $\\Lambda$CDM (identically 0)",
    )
    axr.scatter(
        ell_data[mask], resid_sigma, s=6, color="k", zorder=3,
        label=r"(Data $-$ $\Lambda$CDM) / $\sigma$",
    )
    axr.axhspan(-1, 1, color="0.85", zorder=0)
    axr.set_xscale("log")
    axr.set_xlim(2, 2500)
    axr.set_ylim(-6, 6)
    axr.set_xlabel(r"Multipole $\ell$", fontsize=13)
    axr.set_ylabel("Residual\n" r"[$\sigma$]", fontsize=9)
    axr.grid(True, which="both", ls=":", alpha=0.4)
    axr.legend(loc="upper right", fontsize=7.5, ncol=2)

    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {outpath}")


def make_combined_figure(
    ell_data: np.ndarray, Dl_data: np.ndarray, dlo: np.ndarray, dup: np.ndarray,
    ell_th: np.ndarray, Dl_th: np.ndarray,
    outpath: Path,
) -> None:
    """Single figure overlaying both models against LCDM and Planck data."""
    fig, ax = plt.subplots(figsize=(10, 6.5))

    mask = decimate_mask(ell_data)
    ax.errorbar(
        ell_data[mask], Dl_data[mask],
        yerr=[dlo[mask], dup[mask]],
        fmt="o", ms=2.5, color="k", ecolor="0.55",
        elinewidth=0.8, capsize=0, zorder=4,
        label="Planck 2018 TT (data)",
    )
    ax.plot(
        ell_th, Dl_th, color="C0", lw=2.4, zorder=2,
        label=r"Unmodified $\Lambda$CDM (Planck best-fit)",
    )

    # The two model curves are physically identical to the LCDM curve (that is
    # the paper's point). A sub-percent cosmetic offset is applied purely so
    # both dashed lines remain visible instead of perfectly overlapping.
    cosmetic_offsets = [1.000, 0.994]
    for model, offset in zip(MODELS.values(), cosmetic_offsets):
        ax.plot(
            ell_th, Dl_th * offset,
            color=model.color, lw=1.6, ls="--", zorder=3,
            label=f"{model.label} ({model.g5_latex}) $-$ coincides with $\\Lambda$CDM",
        )

    ax.set_xscale("log")
    ax.set_xlim(2, 2500)
    ax.set_ylim(0, 6500)
    ax.set_xlabel(r"Multipole $\ell$", fontsize=13)
    ax.set_ylabel(r"$D_\ell^{TT}=\ell(\ell+1)C_\ell^{TT}/2\pi$  [$\mu\mathrm{K}^2$]", fontsize=13)
    ax.set_title(
        "CMB TT angular power spectrum\n"
        r"Horndeski Models 1 & 2 of arXiv:2110.01338 vs unmodified $\Lambda$CDM "
        "and Planck 2018 data",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)

    note = (
        "Both Horndeski models reproduce\n"
        r"$H_0\simeq 74$ km/s/Mpc while leaving the" "\n"
        "background expansion (and hence the\n"
        r"CMB acoustic-peak structure) unchanged" "\n"
        r"relative to Planck's $\Lambda$CDM best fit." "\n"
        "(Curves offset by <1% here purely for visibility.)"
    )
    ax.text(
        0.03, 0.97, note,
        transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {outpath}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMB TT spectrum: Horndeski Models 1 & 2 (arXiv:2110.01338) vs Planck 2018.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--which", choices=["model1", "model2", "combined", "all"], default="all",
        help="Which figure(s) to produce (default: all).",
    )
    p.add_argument(
        "--outdir", type=Path, default=SCRIPT_DIR,
        help="Directory to save figures in (default: script directory).",
    )
    p.add_argument(
        "--table-n", type=int, default=8,
        help="Number of low-ell rows to print in the data-vs-theory table (default: 8).",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    try:
        download_if_needed()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ell_data, Dl_data, dlo, dup = load_planck_tt(TT_FILE)
    ell_th, Dl_th = load_planck_theory(THEORY_FILE)

    chi2, n = chi2_data_vs_theory(ell_data, Dl_data, dlo, dup, ell_th, Dl_th)
    print(
        f"\nData vs Planck best-fit LCDM theory: chi^2 = {chi2:.1f} for {n} points "
        f"(chi^2/dof = {chi2 / n:.2f})"
    )
    print_low_ell_table(ell_data, Dl_data, ell_th, Dl_th, n=args.table_n)

    if args.which in ("model1", "all"):
        make_single_model_figure(
            MODELS["model1"], ell_data, Dl_data, dlo, dup, ell_th, Dl_th,
            args.outdir / "CMB_TT_Model1_vs_Planck2018.png",
        )
    if args.which in ("model2", "all"):
        make_single_model_figure(
            MODELS["model2"], ell_data, Dl_data, dlo, dup, ell_th, Dl_th,
            args.outdir / "CMB_TT_Model2_vs_Planck2018.png",
        )
    if args.which in ("combined", "all"):
        make_combined_figure(
            ell_data, Dl_data, dlo, dup, ell_th, Dl_th,
            args.outdir / "CMB_TT_Model1and2_vs_Planck2018.png",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())