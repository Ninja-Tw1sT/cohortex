"""Run every example and report pass/fail. Needs a local Ollama with phi3:mini."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))


def main() -> int:
    results = []
    for script in EXAMPLES:
        print(f"\n{'=' * 70}\n▶ {script.name}\n{'=' * 70}")
        start = time.monotonic()
        rc = subprocess.run([sys.executable, str(script)], cwd=ROOT, timeout=600).returncode
        results.append((script.name, rc == 0, time.monotonic() - start))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, secs in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {secs:6.1f}s  {name}")
    print(f"\n{passed}/{len(results)} examples passed.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
