#!/usr/bin/env python3
"""Usage: torchrun --nproc-per-node=N train.py [options].

Run deterministic DDP training on synthetic tokens, checkpoint periodically,
resume after failure, and append rank-zero step timings to CSV. CUDA/NCCL is
the experiment default; CPU/Gloo is available for local recovery validation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import random
import sys
import time
from pathlib import Path


CSV_FIELDS = [
    "epoch_s",
    "utc",
    "experiment_id",
    "attempt",
    "rank",
    "event",
    "step",
    "step_time_s",
    "checkpoint_interval",
    "detail",
]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/checkpoints"))
    parser.add_argument("--resume", help="checkpoint path or 'latest'")
    parser.add_argument("--log-csv", type=Path, default=Path("results/training.csv"))
    parser.add_argument("--control-dir", type=Path, default=Path("results/control"))
    parser.add_argument("--experiment-id", default="checkpoint-50")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Training device; CPU uses Gloo and is intended for local validation.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="Expected torchrun world size; fail if WORLD_SIZE does not match.",
    )
    parser.add_argument(
        "--ddp-timeout-seconds",
        type=int,
        default=int(os.environ.get("DDP_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    invalid_world_size = args.world_size is not None and args.world_size < 1
    if args.steps < 1 or args.checkpoint_interval < 1 or invalid_world_size:
        parser.error("steps, checkpoint interval, and world size must be positive")
    if args.d_model % args.heads:
        parser.error("d-model must be divisible by heads")
    return args


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


class CsvLogger:
    def __init__(self, path: Path, fixed: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        self.handle = path.open("a", encoding="utf-8", newline="", buffering=1)
        self.writer = csv.DictWriter(self.handle, fieldnames=CSV_FIELDS)
        self.fixed = fixed
        if new_file:
            self.writer.writeheader()

    def write(self, event: str, step: int | str = "", step_time: float | str = "", detail: str = "") -> None:
        row = dict(self.fixed)
        row.update(
            epoch_s=f"{time.time():.6f}",
            utc=utc_now(),
            event=event,
            step=step,
            step_time_s=f"{step_time:.9f}" if isinstance(step_time, float) else step_time,
            detail=detail,
        )
        self.writer.writerow(row)

    def close(self) -> None:
        self.handle.close()


def latest_checkpoint(directory: Path) -> Path:
    candidates = sorted(directory.glob("checkpoint-step-*.pt"))
    if not candidates:
        raise SystemExit(f"no checkpoint found in {directory}")
    return candidates[-1]


def main() -> int:
    args = arguments()
    if args.dry_run:
        print(
            f"would run deterministic {args.device} DDP: "
            f"steps={args.steps}, checkpoint_interval={args.checkpoint_interval}, "
            f"world_size={args.world_size or 'from torchrun'}, "
            f"resume={args.resume or 'none'}, log={args.log_csv}"
        )
        return 0

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(args.vocab_size, args.d_model)
            self.position = nn.Parameter(torch.zeros(args.sequence_length, args.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.heads,
                dim_feedforward=args.d_model * 4,
                dropout=0.0,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=args.layers)
            self.norm = nn.LayerNorm(args.d_model)
            self.output = nn.Linear(args.d_model, args.vocab_size)

        def forward(self, tokens: object) -> object:
            hidden = self.embedding(tokens) + self.position
            return self.output(self.norm(self.encoder(hidden)))

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.world_size is not None and args.world_size != world_size:
        raise SystemExit(
            f"--world-size {args.world_size} does not match torchrun WORLD_SIZE={world_size}"
        )
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is unavailable; use --device cpu only for local validation")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
        ddp_options = {"device_ids": [local_rank]}
    else:
        device = torch.device("cpu")
        backend = "gloo"
        ddp_options = {}
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = False

    timeout = dt.timedelta(seconds=args.ddp_timeout_seconds)
    dist.init_process_group(backend=backend, timeout=timeout)
    model = DistributedDataParallel(TinyTransformer().to(device), **ddp_options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_step = 0

    resume_path: Path | None = None
    if args.resume:
        resume_path = latest_checkpoint(args.checkpoint_dir) if args.resume == "latest" else Path(args.resume)
        payload = torch.load(resume_path, map_location=device, weights_only=False)
        model.module.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["next_step"])

    args.control_dir.mkdir(parents=True, exist_ok=True)
    pid_file = args.control_dir / f"rank-{rank}.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="ascii")
    logger: CsvLogger | None = None
    if rank == 0:
        logger = CsvLogger(
            args.log_csv,
            {
                "experiment_id": args.experiment_id,
                "attempt": args.attempt,
                "rank": rank,
                "checkpoint_interval": args.checkpoint_interval,
            },
        )
        logger.write(
            "run_start",
            step=start_step,
            detail=(
                f"world_size={world_size};device={args.device};backend={backend};"
                f"resume={resume_path or 'none'}"
            ),
        )

    try:
        for step in range(start_step, args.steps):
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + step * world_size + rank)
            tokens = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.sequence_length),
                generator=generator,
                device=device,
            )
            targets = torch.roll(tokens, shifts=-1, dims=1)
            if args.device == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits.reshape(-1, args.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            if args.device == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if logger:
                logger.write("step", step=step, step_time=elapsed, detail=f"loss={loss.item():.6f}")

            if (step + 1) % args.checkpoint_interval == 0 or step + 1 == args.steps:
                if rank == 0:
                    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    destination = args.checkpoint_dir / f"checkpoint-step-{step + 1:08d}.pt"
                    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "next_step": step + 1,
                            "seed": args.seed,
                            "world_size": world_size,
                        },
                        temporary,
                    )
                    os.replace(temporary, destination)
                    if logger:
                        logger.write("checkpoint", step=step + 1, detail=str(destination))
                dist.barrier()
        if logger:
            logger.write("run_end", step=args.steps)
    finally:
        if logger:
            logger.close()
        try:
            if pid_file.exists() and pid_file.read_text(encoding="ascii").strip() == str(os.getpid()):
                pid_file.unlink()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
