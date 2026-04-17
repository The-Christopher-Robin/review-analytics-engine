#!/usr/bin/env python3
"""Standalone evaluation script. Loads saved results and prints comparison."""

import json
import argparse
from pathlib import Path
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser(description="Compare model evaluation results")
    parser.add_argument(
        "--results-file",
        default="data/processed/model_results.json",
        help="Path to model_results.json",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run `python scripts/train.py` first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    rows = []
    for model, metrics in results.items():
        rows.append([
            model,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision_macro']:.4f}",
            f"{metrics['recall_macro']:.4f}",
            f"{metrics['f1_macro']:.4f}",
            f"{metrics['f1_weighted']:.4f}",
        ])

    rows.sort(key=lambda x: float(x[1]), reverse=True)

    headers = ["Model", "Accuracy", "Prec (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)"]
    print("\n=== Model Comparison ===\n")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    best = rows[0]
    print(f"\nBest model: {best[0]} (accuracy: {best[1]})")


if __name__ == "__main__":
    main()
