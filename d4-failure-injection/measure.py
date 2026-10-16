#!/usr/bin/env python3
"""Usage: measure.py [--results-dir DIR] [--events FILE] [--dry-run].

Compare checkpoint intervals using detection, restart, wall time, and effective
training time ratio derived from train.py and inject.sh CSV files.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Experiment:
    experiment_id: str
    interval: int
    records: list[dict[str, str]] = field(default_factory=list)
    events: list[dict[str, str]] = field(default_factory=list)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def calculate(exp: Experiment) -> list[str]:
    timeline = sorted(exp.records, key=lambda row: number(row, "epoch_s"))
    steps = [row for row in timeline if row.get("event") == "step" and row.get("step_time_s")]
    if not timeline or not steps:
        return [exp.experiment_id, str(exp.interval), "0", "—", "—", "—", "—"]

    unique_steps = len({int(row["step"]) for row in steps})
    median_step = statistics.median(number(row, "step_time_s") for row in steps)
    wall = number(timeline[-1], "epoch_s") - number(timeline[0], "epoch_s")
    ratio = unique_steps * median_step / wall if wall > 0 else 0.0

    events = sorted(exp.events, key=lambda row: number(row, "epoch_s"))
    injected = next((number(row, "epoch_s") for row in events if row.get("event") == "injected"), None)
    detected = next(
        (number(row, "epoch_s") for row in events if row.get("event") == "detected" and (injected is None or number(row, "epoch_s") >= injected)),
        None,
    )
    detection = detected - injected if injected is not None and detected is not None else None
    first_recovered_step = next(
        (number(row, "epoch_s") for row in steps if detected is not None and number(row, "epoch_s") > detected),
        None,
    )
    restart = first_recovered_step - detected if first_recovered_step is not None and detected is not None else None
    return [
        exp.experiment_id,
        str(exp.interval),
        str(unique_steps),
        f"{wall:.2f}",
        fmt(detection),
        fmt(restart),
        f"{ratio:.4f}",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--events", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(f"would scan training CSV files under {args.results_dir} and compare checkpoint intervals")
        return 0

    groups: dict[tuple[str, int], Experiment] = {}
    event_path = args.events or args.results_dir / "events.csv"
    events = read_csv(event_path) if event_path.exists() else []
    for path in sorted(args.results_dir.rglob("*.csv")):
        if path == event_path or path.name == "events.csv":
            continue
        for row in read_csv(path):
            if not row.get("experiment_id") or not row.get("checkpoint_interval") or not row.get("event"):
                continue
            key = (row["experiment_id"], int(row["checkpoint_interval"]))
            groups.setdefault(key, Experiment(*key)).records.append(row)
    for row in events:
        if not row.get("experiment_id") or not row.get("checkpoint_interval"):
            continue
        key = (row["experiment_id"], int(row["checkpoint_interval"]))
        groups.setdefault(key, Experiment(*key)).events.append(row)

    if not groups:
        print("No experiment records found", file=sys.stderr)
        return 1
    headers = ["experiment", "checkpoint steps", "useful steps", "wall s", "detect s", "restart s", "effective ratio"]
    rows = [calculate(groups[key]) for key in sorted(groups, key=lambda item: (item[1], item[0]))]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())

