#!/usr/bin/env python3
"""Usage: torchrun [torchrun options] torch_all_reduce.py [options].

Torch/NCCL fallback for multi-node all-reduce when MPI is unavailable. This is
not nccl-tests; it prints a compatible bandwidth table for explicit comparison.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys


def parse_size(value: str) -> int:
    suffixes = {"K": 1024, "M": 1024**2, "G": 1024**3}
    value = value.upper()
    multiplier = suffixes.get(value[-1], 1)
    number = value[:-1] if multiplier != 1 else value
    return int(float(number) * multiplier)


def sizes_between(start: int, end: int, factor: float) -> list[int]:
    sizes: list[int] = []
    current = start
    while current <= end:
        sizes.append(current)
        next_value = int(current * factor)
        if next_value <= current:
            raise ValueError("factor must increase the message size")
        current = next_value
    return sizes


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-bytes", default="8")
    parser.add_argument("--max-bytes", default="1G")
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.dry_run:
        print("would initialise torch.distributed with NCCL and benchmark all_reduce")
        return 0

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print("size_bytes\titerations\tavg_time_s\talgbw_GBps\tbusbw_GBps")

    try:
        for size in sizes_between(parse_size(args.min_bytes), parse_size(args.max_bytes), args.factor):
            elements = max(1, size // 4)
            tensor = torch.ones(elements, dtype=torch.float32, device=local_rank)
            for _ in range(args.warmup):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            timings: list[float] = []
            for _ in range(args.iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                dist.all_reduce(tensor)
                end.record()
                end.synchronize()
                timings.append(start.elapsed_time(end) / 1000.0)
            average = statistics.mean(timings)
            algbw = size / average / 1e9
            busbw = algbw * 2 * (world_size - 1) / world_size
            if rank == 0:
                print(f"{size}\t{args.iterations}\t{average:.9f}\t{algbw:.3f}\t{busbw:.3f}")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

