#!/usr/bin/env python3
# scripts/batch_run.py
"""
Run all models on all datasets.

Usage:
    python batch_run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.registry import list_models
from data.loader import list_datasets
from scripts.fit_model import main as fit_main


def main():
    models = list_models()
    datasets = list_datasets()
    
    print(f"\n{'='*60}")
    print("BATCH RUN")
    print(f"{'='*60}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}\n")
    
    for model in models:
        for dataset in datasets:
            print(f"\n{'#'*50}")
            print(f"# {model} on {dataset}")
            print(f"{'#'*50}")
            
            # Run fit_model with arguments
            sys.argv = [
                'fit_model.py',
                '--model', model,
                '--dataset', dataset,
                '--output-dir', './results_batch'
            ]
            try:
                fit_main()
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    print("\nBatch run complete!")


if __name__ == "__main__":
    main()