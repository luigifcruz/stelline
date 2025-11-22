"""
Run command implementation for Stelline CLI.
"""

import importlib.util
from pathlib import Path

from stelline.app import App


def run_parser(subparsers):
    """
    Create and configure the run subcommand parser.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object to add the run command to

    Returns
    -------
    argparse.ArgumentParser
        The configured run parser
    """
    parser = subparsers.add_parser("run", help="Run a Stelline pipeline")
    parser.add_argument("config", help="Path to configuration YAML file")
    parser.add_argument(
        "--bits-files",
        action="append",
        help="Python files containing custom bit definitions to load (can be used multiple times)",
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Enable periodic metrics printing",
    )
    parser.add_argument(
        "--metrics-interval",
        type=float,
        default=1.0,
        help="Seconds between metrics refreshes (default: 1.0)",
    )
    return parser


def run_command(args):
    """
    Execute the run command with the given arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments containing config path and bits_files
    """
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")

    print(f"Using configuration file: {args.config}")

    # Load custom bits files if provided
    if args.bits_files:
        for bits_file in args.bits_files:
            bits_path = Path(bits_file)
            if not bits_path.exists():
                print(f"Warning: Custom bits file not found: {bits_file}")
                continue

            print(f"Loading custom bits from: {bits_file}")

            # Import the module to trigger bit registration
            spec = importlib.util.spec_from_file_location(
                f"custom_bits_{bits_path.stem}", bits_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

    # Create and run the application
    app = App(
        args.config,
        metrics=args.metrics,
        metrics_interval=args.metrics_interval,
    )
    app.run()
