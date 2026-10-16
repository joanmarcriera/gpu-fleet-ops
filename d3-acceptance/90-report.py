#!/usr/bin/env python3
"""Usage: 90-report.py [--results-dir DIR] [--output FILE] [--dry-run].

Parse D3 artifacts recursively and write a compact Markdown acceptance table.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any


def walk_statuses(value: Any) -> list[str]:
    statuses: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() == "status" and isinstance(child, str):
                statuses.append(child)
            statuses.extend(walk_statuses(child))
    elif isinstance(value, list):
        for child in value:
            statuses.extend(walk_statuses(child))
    return statuses


def dcgm_result(path: Path) -> tuple[str, str]:
    try:
        statuses = [item.lower() for item in walk_statuses(json.loads(path.read_text()))]
    except (OSError, json.JSONDecodeError) as error:
        return "FAIL", f"invalid JSON: {error}"
    if not statuses:
        return "WARN", "no status fields found"
    if any(item in {"fail", "failed", "error"} for item in statuses):
        return "FAIL", ", ".join(sorted(set(statuses)))
    if any(item in {"skip", "skipped", "not run", "not_run"} for item in statuses):
        return "WARN", ", ".join(sorted(set(statuses)))
    return "PASS", ", ".join(sorted(set(statuses)))


def nccl_result(path: Path) -> tuple[str, str]:
    bandwidths: list[float] = []
    wrong = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 13 and parts[0].isdigit() and parts[1].isdigit():
            try:
                bandwidths.extend((float(parts[7]), float(parts[11])))
                wrong += int(parts[8]) + int(parts[12])
            except ValueError:
                continue
        elif len(parts) == 5 and parts[0].isdigit():
            try:
                bandwidths.append(float(parts[4]))
            except ValueError:
                continue
    if not bandwidths:
        return "FAIL", "no busbw rows parsed"
    status = "PASS" if wrong == 0 else "FAIL"
    return status, f"median busbw {statistics.median(bandwidths):.2f} GB/s; wrong {wrong}"


def hpl_result(path: Path) -> tuple[str, str]:
    matches = re.findall(r"\b([0-9]+(?:\.[0-9]+)?(?:[Ee][+-]?[0-9]+))\s*$", path.read_text(errors="replace"), re.MULTILINE)
    if not matches:
        return "FAIL", "no HPL GFLOPS result parsed"
    gflops = float(matches[-1])
    return "PASS", f"{gflops / 1000:.2f} TFLOPS (threshold TBD)"


def fabric_result(path: Path) -> tuple[str, str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
    failed = [line.split("\t", 1)[0] for line in lines if line.endswith("\tFAIL")]
    return ("FAIL", "increased: " + ", ".join(failed)) if failed else ("PASS", "no critical counter increases")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output or args.results_dir / "acceptance-report.md"
    if args.dry_run:
        print(f"would scan {args.results_dir} recursively and write {output}")
        return 0

    rows: list[tuple[str, str, str, str]] = []
    patterns = [
        ("DCGM level 3", "**/dcgm-diag-r3.json", dcgm_result),
        ("DCGM level 4", "**/dcgm-diag-r4.json", dcgm_result),
        ("NCCL single-node", "**/nccl-single-node.log", nccl_result),
        ("NCCL multi-node", "**/nccl-multi-node-*.log", nccl_result),
        ("HPL", "**/hpl.log", hpl_result),
        ("Fabric counters", "**/fabric-counter-delta.tsv", fabric_result),
    ]
    for stage, pattern, parser_fn in patterns:
        paths = sorted(args.results_dir.glob(pattern))
        if not paths:
            rows.append((stage, "NOT RUN", "—", "no artifact found"))
        for path in paths:
            status, evidence = parser_fn(path)
            rows.append((stage, status, str(path.relative_to(args.results_dir)), evidence))

    lines = [
        "# Node acceptance report",
        "",
        "> Generated from captured artifacts. Performance thresholds marked TBD in the runbook remain a manual gate.",
        "",
        "| Test | Status | Artifact | Evidence |",
        "|---|---|---|---|",
    ]
    lines.extend(f"| {stage} | {status} | `{artifact}` | {evidence} |" for stage, status, artifact, evidence in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 1 if any(row[1] == "FAIL" for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())

