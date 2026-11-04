#!/usr/bin/env python3
"""
Ground Truth Generation Script for ALPHANSO Integration Tests

Run this script to regenerate ground truth results after validating
the current implementation. Only run when you have confirmed the
implementation is correct.

Usage:
    python generate_ground_truth.py [--force]

    --force: Overwrite existing ground truth without prompting
"""

import json
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alphanso.transport import Transport
from tests.integration.configs import (
    BEAM_CONFIGS,
    HOMOGENEOUS_CONFIGS,
    INTERFACE_CONFIGS,
    SANDWICH_CONFIGS
)


def serialize_results(results: dict) -> dict:
    """Convert numpy arrays to lists for JSON serialization."""
    serialized = {}
    for key, value in results.items():
        if hasattr(value, 'tolist'):
            serialized[key] = value.tolist()
        elif isinstance(value, list):
            # Handle list of arrays or nested structures
            serialized[key] = [
                v.tolist() if hasattr(v, 'tolist') else v for v in value
            ]
        elif isinstance(value, dict):
            serialized[key] = serialize_results(value)
        else:
            serialized[key] = value
    return serialized


def generate_ground_truth_for_configs(configs: list, calc_type: str) -> dict:
    """Generate ground truth results for a list of configs."""
    results = {}
    for config in configs:
        name = config['name']
        print(f"  Running {name}...")
        try:
            # The binned gamma spectrum has its own tests and would add megabytes
            # to these reference files.
            result = Transport.calculate({**config, "gamma_doppler": False})
            results[name] = serialize_results(result)
            print(f"    OK (an_yield={result.get('an_yield', 'N/A'):.3e})")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"error": str(e)}
    return results


def save_ground_truth(results: dict, output_path: Path, calc_type: str):
    """Save ground truth with metadata."""
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "calc_type": calc_type,
        "num_configs": len(results),
        "notes": "Ground truth generated from validated implementation"
    }
    data = {
        "metadata": metadata,
        "results": results
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ground truth for ALPHANSO tests")
    parser.add_argument('--force', action='store_true', help="Overwrite without prompting")
    args = parser.parse_args()

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    if not args.force:
        existing = list(output_dir.glob("*.json"))
        if existing:
            response = input(f"Found {len(existing)} existing ground truth files. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return

    # Generate for each category
    print("\n" + "=" * 60)
    print("Generating BEAM ground truth (5 configs)...")
    print("=" * 60)
    beam_results = generate_ground_truth_for_configs(BEAM_CONFIGS, "beam")
    save_ground_truth(beam_results, output_dir / "beam.json", "beam")

    print("\n" + "=" * 60)
    print("Generating HOMOGENEOUS ground truth (10 configs)...")
    print("=" * 60)
    homog_results = generate_ground_truth_for_configs(HOMOGENEOUS_CONFIGS, "homogeneous")
    save_ground_truth(homog_results, output_dir / "homogeneous.json", "homogeneous")

    print("\n" + "=" * 60)
    print("Generating INTERFACE ground truth (5 configs)...")
    print("=" * 60)
    interface_results = generate_ground_truth_for_configs(INTERFACE_CONFIGS, "interface")
    save_ground_truth(interface_results, output_dir / "interface.json", "interface")

    print("\n" + "=" * 60)
    print("Generating SANDWICH ground truth (10 configs)...")
    print("=" * 60)
    sandwich_results = generate_ground_truth_for_configs(SANDWICH_CONFIGS, "sandwich")
    save_ground_truth(sandwich_results, output_dir / "sandwich.json", "sandwich")

    # Summary
    total = len(beam_results) + len(homog_results) + len(interface_results) + len(sandwich_results)
    errors = sum(1 for r in beam_results.values() if 'error' in r)
    errors += sum(1 for r in homog_results.values() if 'error' in r)
    errors += sum(1 for r in interface_results.values() if 'error' in r)
    errors += sum(1 for r in sandwich_results.values() if 'error' in r)

    print("\n" + "=" * 60)
    print(f"Ground truth generation complete!")
    print(f"Total configs: {total}")
    print(f"Successful: {total - errors}")
    print(f"Errors: {errors}")
    print("=" * 60)


if __name__ == "__main__":
    main()
