"""
mlscratch CLI
=============
Usage
-----
    python -m mlscratch              # same as 'info'
    python -m mlscratch version      # print version
    python -m mlscratch info         # version + sub-package summary
    python -m mlscratch list         # list all available algorithm classes
    python -m mlscratch list supervised
    python -m mlscratch list unsupervised
    python -m mlscratch list bayesian
    python -m mlscratch list reinforcement
"""

from __future__ import annotations

import argparse
import sys


# ── helpers ───────────────────────────────────────────────────────────────────

def _print_version() -> None:
    import mlscratch
    print(f"mlscratch {mlscratch.__version__}")


def _print_info() -> None:
    import mlscratch
    import numpy as np

    print(f"\nmlscratch {mlscratch.__version__}")
    print(f"  numpy   : {np.__version__}")
    print(f"  python  : {sys.version.split()[0]}")
    print()

    modules = {
        "supervised":     "mlscratch.supervised",
        "unsupervised":   "mlscratch.unsupervised",
        "bayesian":       "mlscratch.bayesian",
        "reinforcement":  "mlscratch.reinforcement",
    }

    for name, mod_path in modules.items():
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            n = len(getattr(mod, "__all__", []))
            print(f"  {name:<18} {n} public symbol(s)")
        except ImportError:
            print(f"  {name:<18} not yet installed")

    print()


def _list_algorithms(subpackage: str | None = None) -> None:
    import importlib

    targets = (
        {"supervised", "unsupervised", "bayesian", "reinforcement"}
        if subpackage is None
        else {subpackage}
    )

    for name in sorted(targets):
        mod_path = f"mlscratch.{name}"
        try:
            mod = importlib.import_module(mod_path)
            symbols = getattr(mod, "__all__", [])
            print(f"\n[{name}]")
            for s in sorted(symbols):
                print(f"  {s}")
        except ImportError:
            print(f"\n[{name}]  — not available (sub-package not installed)")


# ── entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlscratch",
        description="mlscratch — ML algorithms from scratch",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print version and exit")
    subparsers.add_parser("info",    help="Print version, numpy, and sub-package summary")

    list_parser = subparsers.add_parser("list", help="List available algorithms")
    list_parser.add_argument(
        "subpackage",
        nargs="?",
        choices=["supervised", "unsupervised", "bayesian", "reinforcement"],
        default=None,
        help="Restrict listing to one sub-package",
    )

    args = parser.parse_args(argv)

    if args.command in (None, "info"):
        _print_info()
    elif args.command == "version":
        _print_version()
    elif args.command == "list":
        _list_algorithms(args.subpackage)

    return 0


if __name__ == "__main__":
    sys.exit(main())
