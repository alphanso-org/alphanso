from pathlib import Path

import numpy as np
import yaml


def normalize_results_payload(value):
    """Recursively convert results payloads to built-in YAML-safe Python types."""
    if isinstance(value, np.ndarray):
        return normalize_results_payload(value.tolist())

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: normalize_results_payload(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize_results_payload(item) for item in value]

    return value


def write_results_yaml(results: dict, output_dir, filename: str = "results.yaml") -> Path:
    """Write a normalized results dictionary to a YAML file and return the file path."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_results = normalize_results_payload(results)
    results_file = output_path / filename
    with open(results_file, "w") as f:
        yaml.safe_dump(
            normalized_results,
            f,
            default_flow_style=True,
            indent=2,
            sort_keys=False,
        )

    return results_file
