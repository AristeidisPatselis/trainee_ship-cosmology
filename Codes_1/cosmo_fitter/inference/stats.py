# inference/stats.py
"""Statistical tools for model comparison."""

import numpy as np
from typing import Dict, List, Tuple


def calculate_aic(chi2: float, n_params: int) -> float:
    """Calculate Akaike Information Criterion."""
    return chi2 + 2 * n_params


def calculate_aicc(chi2: float, n_params: int, n_data: int) -> float:
    """Calculate corrected AIC (AICc)."""
    aic = calculate_aic(chi2, n_params)
    if n_data > n_params + 1:
        return aic + (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
    return aic


def calculate_bic(chi2: float, n_params: int, n_data: int) -> float:
    """Calculate Bayesian Information Criterion."""
    return chi2 + n_params * np.log(n_data)


def model_comparison_table(
    results: Dict[str, Dict],
    n_data: int
) -> List[Dict]:
    """
    Create a model comparison table.
    
    Args:
        results: Dictionary mapping model names to fit results
        n_data: Number of data points
    
    Returns:
        List of dictionaries with model comparison statistics
    """
    table = []
    
    for name, res in results.items():
        chi2 = res['chi2']
        n_params = res['n_params']
        
        aic = calculate_aic(chi2, n_params)
        aicc = calculate_aicc(chi2, n_params, n_data)
        bic = calculate_bic(chi2, n_params, n_data)
        dof = n_data - n_params
        chi2_dof = chi2 / dof if dof > 0 else np.inf
        
        table.append({
            'model': name,
            'chi2': chi2,
            'dof': dof,
            'chi2_dof': chi2_dof,
            'n_params': n_params,
            'aic': aic,
            'aicc': aicc,
            'bic': bic,
            'dic': res.get('dic', np.nan),
        })
    
    # Add delta values relative to best model
    best_aic = min(t['aic'] for t in table)
    best_bic = min(t['bic'] for t in table)
    best_dic = min(t['dic'] for t in table if np.isfinite(t['dic']))
    
    for t in table:
        t['delta_aic'] = t['aic'] - best_aic
        t['delta_bic'] = t['bic'] - best_bic
        t['delta_dic'] = t['dic'] - best_dic if np.isfinite(t['dic']) else np.nan
    
    return table


def print_model_comparison_table(table: List[Dict]) -> None:
    """Print a formatted model comparison table."""
    print("\n" + "=" * 110)
    print("MODEL COMPARISON TABLE")
    print("=" * 110)
    
    header = f"{'Model':<15}{'k':>4}{'dof':>6}{'chi2':>10}{'chi2/dof':>11}"
    header += f"{'AIC':>10}{'dAIC':>9}{'BIC':>10}{'dBIC':>9}{'DIC':>10}{'dDIC':>9}"
    print(header)
    print("-" * 110)
    
    for t in sorted(table, key=lambda x: x['aic']):
        row = (
            f"{t['model']:<15}"
            f"{t['n_params']:>4}"
            f"{t['dof']:>6.0f}"
            f"{t['chi2']:>10.3f}"
            f"{t['chi2_dof']:>11.3f}"
            f"{t['aic']:>10.3f}"
            f"{t['delta_aic']:>9.3f}"
            f"{t['bic']:>10.3f}"
            f"{t['delta_bic']:>9.3f}"
        )
        if np.isfinite(t['dic']):
            row += f"{t['dic']:>10.3f}{t['delta_dic']:>9.3f}"
        else:
            row += f"{'N/A':>10}{'N/A':>9}"
        print(row)
    
    print("=" * 110)
    print("\nJeffreys-scale interpretation:")
    print("  ΔIC ≤ 2: statistically indistinguishable")
    print("  2 < ΔIC < 6: mild tension")
    print("  ΔIC ≥ 10: strong tension")