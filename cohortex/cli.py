"""Command-line interface:  python -m cohortex run <crew> "<task>"."""
from __future__ import annotations

import argparse
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cohortex", description="Run a Cohortex crew.")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="run a crew against a task")
    run.add_argument("crew")
    run.add_argument("task", nargs="+")
    run.add_argument("--quiet", action="store_true", help="print only the final answer")

    sub.add_parser("backends", help="list available LLM backends")
    sub.add_parser("crews", help="list configured crews")

    args = p.parse_args(argv)

    if args.cmd == "backends":
        from cohortex.providers import available_backends
        print("\n".join(available_backends()))
        return 0

    if args.cmd == "crews":
        d = _config().CONFIG_DIR / "crews"
        crews = sorted(f.stem for f in d.glob("*.yaml")) if d.exists() else []
        print("\n".join(crews) if crews else "(no crews configured)")
        return 0

    if args.cmd == "run":
        from cohortex.runtime import run_crew
        result = run_crew(args.crew, " ".join(args.task))
        if not args.quiet:
            for s in result.steps:
                print(f"[{s.agent}] {s.output}\n")
            print("=" * 60)
        print(result.output)
        return 0

    p.print_help()
    return 1


def _config():
    from cohortex import config
    return config


if __name__ == "__main__":
    raise SystemExit(main())
