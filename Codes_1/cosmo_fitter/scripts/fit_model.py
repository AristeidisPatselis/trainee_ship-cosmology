#!/usr/bin/env python3
# scripts/fit_model.py
"""
Fit a single cosmological model to data.

Usage:
    python fit_model.py --model delta --dataset cosmic_chronometers
    python fit_model.py --model lcdm --data-dir /path/to/data
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import get_config, reset_config
from data.loader import load_dataset
from models.registry import get_model, list_models
from inference.optimizer import find_best_fit, curve_fit_uncertainties
from inference.mcmc import run_mcmc, get_mcmc_summary, compute_dic
from inference.stats import calculate_aic, calculate_bic
from plotting.contours import plot_contours, plot_delta_chi2_contours
from plotting.hubble import plot_hubble_diagram


def main():
    parser = argparse.ArgumentParser(
        description="Fit a cosmological model to data"
    )
    parser.add_argument(
        '--model',
        required=True,
        choices=list_models(),
        help='Model to fit'
    )
    parser.add_argument(
        '--dataset',
        default='cosmic_chronometers',
        help='Dataset name'
    )
    parser.add_argument(
        '--data-dir',
        help='Data directory (overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Output directory'
    )
    parser.add_argument(
        '--no-mcmc',
        action='store_true',
        help='Skip MCMC (faster but no uncertainties)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FITTING MODEL: {args.model}")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}\n")
    
    # Load data
    config = get_config()
    if args.data_dir:
        # Override data directory
        from config.manager import Config, DataConfig
        config.data.base_dir = Path(args.data_dir)
    
    dataset = load_dataset(args.dataset, config=config)
    z, H_obs, sigma = dataset.z, dataset.H, dataset.sigma
    
    # Get model
    model = get_model(args.model, config=config)
    print(f"\nModel: {model.name}")
    print(f"Parameters: {model.param_names}")
    print(f"Bounds: {model.param_bounds}")
    
    # Find best fit
    print("\n--- Finding best fit ---")
    best_params, chi2_best, opt_metadata = find_best_fit(
        model, z, H_obs, sigma,
        verbose=args.verbose
    )
    
    # Print best-fit parameters
    print("\nBest-fit parameters:")
    for name, value in zip(model.param_names, best_params):
        print(f"  {name} = {value:.6f}")
    print(f"  chi^2 = {chi2_best:.4f}")
    
    # Curve fit uncertainties
    print("\n--- Estimating uncertainties ---")
    try:
        popt, perr, pcov = curve_fit_uncertainties(
            model, z, H_obs, sigma, best_params
        )
        print("Curve-fit uncertainties:")
        for name, val, err in zip(model.param_names, popt, perr):
            print(f"  {name} = {val:.6f} +/- {err:.6f}")
        
        # Update best fit if curve_fit found better
        chi2_cf = model.chi2(popt, z, H_obs, sigma)
        if chi2_cf < chi2_best:
            best_params, chi2_best = popt, chi2_cf
            print("  Updated best fit from curve_fit")
    except Exception as e:
        print(f"  Curve-fit failed: {e}")
        perr = np.full(model.n_params, np.nan)
        pcov = None
    
    # MCMC
    if not args.no_mcmc:
        print("\n--- Running MCMC ---")
        sampler, flat_samples = run_mcmc(
            model, z, H_obs, sigma, best_params,
            progress=args.verbose
        )
        
        # MCMC summary
        mcmc_summary = get_mcmc_summary(flat_samples, model.param_names)
        print("\nMCMC results:")
        for name, stats in mcmc_summary.items():
            print(f"  {name} = {stats['median']:.6f} "
                  f"(+{stats['upper_error']:.6f}/-{stats['lower_error']:.6f})")
        
        # DIC
        dic, pD = compute_dic(flat_samples, model, z, H_obs, sigma)
        print(f"\nDIC = {dic:.3f}, pD = {pD:.3f}")
        
        # Corner plot
        print("\n--- Creating corner plot ---")
        param_indices = list(range(model.n_params))
        plot_contours(
            model, flat_samples, param_indices,
            output_dir, truth_params=best_params
        )
        
        # Delta-chi2 contours
        print("\n--- Creating delta-chi2 contours ---")
        if model.n_params >= 2:
            param_pairs = [(i, j) for i in range(model.n_params) 
                          for j in range(i+1, model.n_params)]
            if len(param_pairs) > 3:
                param_pairs = param_pairs[:3]  # Limit to 3 plots
            plot_delta_chi2_contours(
                model, best_params, chi2_best,
                z, H_obs, sigma,
                param_pairs, output_dir
            )
    else:
        flat_samples = None
        mcmc_summary = None
        dic = np.nan
    
    # Hubble diagram
    print("\n--- Creating Hubble diagram ---")
    plot_hubble_diagram(
        model, best_params, z, H_obs, sigma,
        output_dir
    )
    
    # Save results
    print("\n--- Saving results ---")
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'best_params': best_params.tolist(),
        'chi2': float(chi2_best),
        'n_params': model.n_params,
        'n_data': len(z),
        'dof': len(z) - model.n_params,
        'chi2_dof': float(chi2_best / (len(z) - model.n_params)),
        'aic': float(chi2_best + 2 * model.n_params),
        'bic': float(chi2_best + model.n_params * np.log(len(z))),
    }
    if not args.no_mcmc and flat_samples is not None:
        results['dic'] = float(dic)
        results['mcmc'] = {
            name: {'median': float(stats['median']),
                   'lower_error': float(stats['lower_error']),
                   'upper_error': float(stats['upper_error'])}
            for name, stats in mcmc_summary.items()
        }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_dir}")
    print("Done!")
    
    return results


if __name__ == "__main__":
    main()