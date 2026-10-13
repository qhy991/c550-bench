"""Prepare or inspect a frozen whole-task C550 Bench campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .prepare import prepare
from .report import report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("prepare", help="create fresh host workspaces; no provider or GPU")
    freeze.add_argument("--plan", required=True, type=Path)
    freeze.add_argument("--routing", required=True, type=Path)
    freeze.add_argument("--host", required=True)
    freeze.add_argument("--output", required=True, type=Path)
    inspect = commands.add_parser("report", help="project the complete frozen denominator")
    inspect.add_argument("--allocation", required=True, type=Path)
    inspect.add_argument("--state-directory", type=Path)
    args = parser.parse_args(argv)
    try:
        result = (prepare(args.plan, args.routing, args.host, args.output)
                  if args.command == "prepare"
                  else report(args.allocation, args.state_directory))
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"C550 campaign refused: {type(error).__name__}: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    sys.exit(main())
