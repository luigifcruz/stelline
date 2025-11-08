import argparse

from stelline import __version__
from stelline.scripts import (
    rap_command,
    rap_parser,
    report_command,
    report_parser,
    run_command,
    run_parser,
)


def main():
    parser = argparse.ArgumentParser(
        description="Stelline - Software Defined Observatory"
    )
    parser.add_argument(
        "--version", "-v", action="version", version=f"Stelline v{__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser(subparsers)
    rap_parser(subparsers)
    report_parser(subparsers)

    args = parser.parse_args()

    if args.command == "run":
        run_command(args)
    elif args.command == "rap":
        rap_command(args)
    elif args.command == "report":
        report_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
