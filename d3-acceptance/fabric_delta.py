#!/usr/bin/env python3
"""Usage: fabric_delta.py --before FILE --after FILE --output FILE [--dry-run].

Aggregate perfquery counters, calculate deltas, and fail on new fabric errors.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CRITICAL = {
    "LinkDownedCounter",
    "LinkErrorRecoveryCounter",
    "PortRcvErrors",
    "PortRcvRemotePhysicalErrors",
    "PortXmitDiscards",
    "PortXmitConstraintErrors",
    "PortRcvConstraintErrors",
    "SymbolErrorCounter",
    "VL15Dropped",
}
COUNTER = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]+):[.\s]*(0x[0-9a-fA-F]+|[0-9]+)\s*$")


def parse(path: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = COUNTER.match(line)
        if match:
            totals[match.group(1)] = totals.get(match.group(1), 0) + int(match.group(2), 0)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"would compare {args.before} with {args.after} and write {args.output}")
        return 0

    before = parse(args.before)
    after = parse(args.after)
    names = sorted(before.keys() | after.keys())
    failures = 0
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("counter\tbefore\tafter\tdelta\tstatus\n")
        for name in names:
            old = before.get(name, 0)
            new = after.get(name, 0)
            delta = new - old
            status = "FAIL" if name in CRITICAL and delta > 0 else "PASS"
            failures += status == "FAIL"
            handle.write(f"{name}\t{old}\t{new}\t{delta}\t{status}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

