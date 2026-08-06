#!/usr/bin/env python3
# scripts/compare_models.py
"""
Compare multiple cosmological models.

Usage:
    python compare_models.py --models lcdm delta --dataset cosmic_chronometers
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.manager import get_config
from data.loader import load_dataset
from models.registry import get_model, list_models
from inference.optimizer import find_best_fit
from inference.mcmc import run_mcmc, compute_dic
from inference.stats import model_comparison_table, print_model_comparison_table
from plotting.hubble import plot_hubble_diagram_comparison
from plotting.contours import plot_contours


def main():
    parser = argparse.ArgumentParser(
        description="Compare cosmological models"
    )
    parser.add_argument(
        '--models',
        nargs='+',
        required=True,
        choices=list_models(),
        help='Models to compare'
    )
    parser.add_argument(
        '--dataset',
        default='cosmic_chronometers',
        help='Dataset name'
    )
    parser.add_argument(
        '--output-dir',
        default='./results_comparison',
        help='Output directory'
    )
    parser.add_argument(
        '--no-mcmc',
        action='store_true',
        help='Skip MCMC'
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"Models: {args.models}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}\n")
    
    # Load data
    config = get_config()
    dataset = load_dataset(args.dataset, config=config)
    z, H_obs, sigma = dataset.z, dataset.H, dataset.sigma
    
    # Fit all models
    results = {}
    for model_name in args.models:
        print(f"\n{'='*50}")
        print(f"FITTING: {model_name}")
        print(f"{'='*50}")
        
        model = get_model(model_name, config=config)
        best_params, chi2_best, _ = find_best_fit(
            model, z, H_obs, sigma,
            verbose=True
        )
        
        print(f"  chi^2 = {chi2_best:.4f}")
        for name, value in zip(model.param_names, best_params):
            print(f"  {name} = {value:.6f}")
        
        result_dict = {
            'model': model,
            'best_params': best_params,
            'chi2': chi2_best,
            'n_params': model.n_params,
        }
        
        # Run MCMC for DIC
        if not args.no_mcmc:
            print("  Running MCMC...")
            _, flat_samples = run_mcmc(
                model, z, H_obs, sigma, best_params,
                progress=False
            )
            dic, pD = compute_dic(flat_samples, model, z, H_obs, sigma)
            result_dict['dic'] = dic
            result_dict['flat_samples'] = flat_samples
            print(f"  DIC = {dic:.3f}, pD = {pD:.3f}")
        
        results[model_name] = result_dict
    
    # Create comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    
    table = model_comparison_table(results, len(z))
    print_model_comparison_table(table)
    
    # Save table
    with open(output_dir / 'comparison_table.json', 'w') as f:
        json.dump(table, f, indent=2)
    
    # Create comparison plots
    print("\n--- Creating comparison plots ---")
    
    # Hubble diagram comparison
    plot_hubble_diagram_comparison(
        results, z, H_obs, sigma, output_dir
    )
    
    # Overlay contours if we have MCMC samples
    if not args.no_mcmc:
        print("  Creating contour comparison...")
        # This would overlay contours from multiple models
        # Implement a function for this if needed
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()