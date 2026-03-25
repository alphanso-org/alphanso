#!/usr/bin/env python3
"""
Main entry point for the alphanso package.

Usage:
    alphanso <config_path> [output_dir]         # backwards-compatible (implicit run)
    alphanso run <config_path> [output_dir]     # explicit run
    alphanso download-data [--dir PATH]         # pre-download data
    alphanso data-info                          # show data paths and status
"""

import argparse
import logging
import os
import sys
import yaml
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Union, Any
from collections import defaultdict

from .output_files import write_results_yaml
from .transport import Transport
from .data_manager import ensure_data, get_data_info

logger = logging.getLogger(__name__)

_SUBCOMMANDS = {"run", "download-data", "data-info"}


def read_in(config_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read configuration from a YAML file or directory of YAML files.

    Args:
        config_path: Path to YAML file or directory containing YAML files

    Returns:
        List of configuration dictionaries, each with a 'source' key indicating the source file

    Raises:
        FileNotFoundError: If config_path doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If no valid YAML files found
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration path does not exist: {config_path}")

    configs = []

    if config_path.is_file():
        try:
            with open(config_path, 'r') as f:
                yaml_content = yaml.safe_load(f)

            if yaml_content is None:
                raise ValueError(f"Empty YAML file: {config_path}")

            if isinstance(yaml_content, list):
                for i, config in enumerate(yaml_content):
                    if isinstance(config, dict):
                        config['source'] = f"{config_path.stem}_{i+1}"
                        configs.append(config)
            elif isinstance(yaml_content, dict):
                yaml_content['source'] = config_path.stem
                configs.append(yaml_content)
            else:
                raise ValueError(
                    f"Invalid YAML content in {config_path}: expected dict or list of dicts")

        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Failed to parse YAML file {config_path}: {e}")

    elif config_path.is_dir():
        yaml_files = list(config_path.glob("*.yaml")) + \
            list(config_path.glob("*.yml"))

        if not yaml_files:
            raise ValueError(
                f"No YAML files found in directory: {config_path}")

        for yaml_file in sorted(yaml_files):
            try:
                with open(yaml_file, 'r') as f:
                    yaml_content = yaml.safe_load(f)

                if yaml_content is None:
                    print(f"Warning: Empty YAML file: {yaml_file}")
                    continue

                if isinstance(yaml_content, list):
                    for i, config in enumerate(yaml_content):
                        if isinstance(config, dict):
                            config['source'] = f"{yaml_file.stem}_{i+1}"
                            configs.append(config)
                elif isinstance(yaml_content, dict):
                    yaml_content['source'] = yaml_file.stem
                    configs.append(yaml_content)
                else:
                    print(
                        f"Warning: Invalid YAML content in {yaml_file}: expected dict or list of dicts")
                    continue

            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse YAML file {yaml_file}: {e}")
                continue

    else:
        raise ValueError(
            f"Invalid config path: {config_path} is neither a file nor directory")

    if not configs:
        raise ValueError(f"No valid configurations found in {config_path}")

    print(f"Loaded {len(configs)} configuration(s) from {config_path}")
    return configs


def read_out(configs: List[Dict[str, Any]],
             output_dir: Union[str, Path] = None) -> Path:
    """
    Set up output directory structure for the configurations.

    Args:
        configs: List of configuration dictionaries
        output_dir: Optional output directory path (default: creates alphanso_output/)

    Returns:
        Path to the created output directory

    Raises:
        OSError: If output directory creation fails
    """
    if output_dir is None:
        output_dir = Path("alphanso_output")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_groups = defaultdict(list)
    for config in configs:
        source = config.get('source', 'unknown')
        source_groups[source].append(config)

    for source, config_list in source_groups.items():
        source_dir = output_dir / source
        source_dir.mkdir(exist_ok=True)

        if len(config_list) == 1:
            if '_result' in config_list[0]:
                write_results_yaml(config_list[0]['_result'], source_dir)
        else:
            for i, config in enumerate(config_list):
                if '_result' in config:
                    write_results_yaml(
                        config['_result'],
                        source_dir,
                        filename=f"results_{i+1}.yaml")

    print(f"Created output directory structure: {output_dir}")
    return output_dir


def _validate_and_normalize_config(config, config_id):
    required_fields = ['calc_type']
    missing_fields = [f for f in required_fields if f not in config]
    if missing_fields:
        raise ValueError(f"Configuration {config_id} missing required fields: {missing_fields}")

    calc_type = config.get('calc_type')
    if calc_type not in ['beam', 'homogeneous', 'interface', 'sandwich']:
        raise ValueError(
            f"Configuration {config_id} has invalid calc_type: {calc_type}. Must be 'beam', 'homogeneous', 'interface', or 'sandwich'")

    if calc_type == 'beam':
        if 'beam_energy' not in config and 'beam_intensities' not in config:
            raise ValueError(
                f"Beam calculation {config_id} missing required field: beam_energy or beam_intensities")
    elif calc_type == 'interface':
        if 'source_matdef' not in config or 'target_matdef' not in config:
            raise ValueError(
                f"Interface calculation {config_id} missing required field: source_matdef or target_matdef")
    elif calc_type == 'sandwich':
        required = ['source_matdef', 'source_density', 'target_matdef']
        missing = [f for f in required if f not in config]
        if missing:
            raise ValueError(f"Sandwich calculation {config_id} missing required fields: {missing}")

        has_layers_list = 'intermediate_layers' in config
        has_single_layer = all(
            k in config for k in ['intermediate_matdef', 'intermediate_density', 'intermediate_thickness'])

        if not (has_layers_list or has_single_layer):
            raise ValueError(
                f"Sandwich calculation {config_id} requires either 'intermediate_layers' (list) "
                "or 'intermediate_matdef', 'intermediate_density', 'intermediate_thickness' (single layer)")

        if has_single_layer and not has_layers_list:
            config['intermediate_layers'] = [{
                'matdef': config.pop('intermediate_matdef'),
                'density': config.pop('intermediate_density'),
                'thickness': config.pop('intermediate_thickness'),
                'include_targets': config.get('include_intermediate_targets', True)
            }]

        for j, layer in enumerate(config['intermediate_layers']):
            if 'matdef' not in layer or 'density' not in layer or 'thickness' not in layer:
                raise ValueError(
                    f"Sandwich calculation {config_id} layer {j} missing required fields: matdef, density, thickness")
            if layer['thickness'] <= 0:
                raise ValueError(
                    f"Sandwich calculation {config_id} layer {j} thickness must be positive, got {layer['thickness']}")
            if layer['density'] <= 0:
                raise ValueError(
                    f"Sandwich calculation {config_id} layer {j} density must be positive, got {layer['density']}")


def _run_one_config(config):
    try:
        return Transport.calculate(config), None
    except Exception as e:
        return None, str(e)


def _cmd_run(config_path, output_dir):
    """Execute the run subcommand (calculation)."""
    ensure_data()

    configs = read_in(config_path)

    valid_indices = []
    for i, config in enumerate(configs):
        config_id = config.get('id', config.get('name', f'config_{i+1}'))
        try:
            _validate_and_normalize_config(config, config_id)
            valid_indices.append(i)
        except Exception as e:
            print(f"Error in config {config_id}: {e}")

    if not valid_indices:
        print("No valid configurations to run.")
        read_out(configs, output_dir)
        return

    valid_configs = [configs[i] for i in valid_indices]
    print(f"\nRunning {len(valid_configs)} configuration(s)...")

    if len(valid_configs) == 1:
        pairs = [_run_one_config(valid_configs[0])]
    else:
        with ProcessPoolExecutor() as pool:
            pairs = list(pool.map(_run_one_config, valid_configs))

    for i, (result, error) in zip(valid_indices, pairs):
        config_id = configs[i].get('id', configs[i].get('name', f'config_{i+1}'))
        if error:
            print(f"Error processing {config_id}: {error}")
        else:
            configs[i]['_result'] = result
            logger.info(f"Completed {config_id}")

    output_dir_path = read_out(configs, output_dir)
    print(f"Output saved to: {output_dir_path}")


def _cmd_download_data(target_dir):
    """Execute the download-data subcommand."""
    if target_dir:
        os.environ["ALPHANSO_DATA_DIR"] = target_dir
    data_path = ensure_data()
    print(f"Nuclear data ready at: {data_path}")


def _cmd_data_info():
    """Execute the data-info subcommand."""
    info = get_data_info()
    print("ALPHANSO Data Info")
    print("=" * 40)
    for key, value in info.items():
        print(f"  {key}: {value}")


def main():
    """Main entry point for the alphanso package."""
    # Backwards compatibility: if the first positional arg is not a known
    # subcommand, treat the entire invocation as an implicit "run".
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] not in _SUBCOMMANDS and not raw_args[0].startswith("-"):
        # Implicit run: alphanso config.yaml [output_dir]
        raw_args = ["run"] + raw_args

    parser = argparse.ArgumentParser(
        description="ALPHANSO: Alpha-Neutron Source calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    alphanso config.yaml                        # Run with config (implicit run)
    alphanso run config.yaml                    # Explicit run
    alphanso run config.yaml results/           # Run with custom output dir
    alphanso download-data                      # Pre-download nuclear data
    alphanso download-data --dir /custom/path   # Download to custom location
    alphanso data-info                          # Show data paths and status
        """
    )

    subparsers = parser.add_subparsers(dest="command")

    # run subcommand
    run_parser = subparsers.add_parser(
        "run", help="Run ALPHANSO calculation(s)")
    run_parser.add_argument(
        "config_path",
        help="Path to YAML config file or directory containing YAML files"
    )
    run_parser.add_argument(
        "output_dir",
        nargs="?",
        help="Output directory (default: alphanso_output/)"
    )

    # download-data subcommand
    dl_parser = subparsers.add_parser(
        "download-data", help="Download nuclear data files")
    dl_parser.add_argument(
        "--dir",
        dest="target_dir",
        default=None,
        help="Target directory for data download"
    )

    # data-info subcommand
    subparsers.add_parser(
        "data-info", help="Show nuclear data paths and status")

    args = parser.parse_args(raw_args)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "run":
            _cmd_run(args.config_path, args.output_dir)
        elif args.command == "download-data":
            _cmd_download_data(args.target_dir)
        elif args.command == "data-info":
            _cmd_data_info()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
