# D4: DDP failure injection

`train.py` is a small deterministic transformer over synthetic token batches.
It defaults to torchrun/NCCL DDP, writes atomic periodic checkpoints, resumes
from a named or latest checkpoint, publishes one PID file per rank, and appends
rank-0 events plus per-step timing to CSV. CPU/Gloo is a local validation mode,
not a substitute for the GPU experiment. `inject.sh` applies one controlled
fault; `measure.py` compares checkpoint intervals.

Dry-run everything first:

```bash
./train.py --dry-run --checkpoint-interval 25
./inject.sh --dry-run --action kill --rank 1
./inject.sh --dry-run --action pause --rank 1 --duration 30
./inject.sh --dry-run --action network-drop --interface ib0 --peer 192.0.2.2
./inject.sh --dry-run --action nccl-timeout --rank 1
./measure.py --dry-run
```

Run the non-performance CPU recovery path through torchrun:

```bash
torchrun --standalone --nproc-per-node=1 train.py \
  --device cpu --world-size 1 --steps 30 --checkpoint-interval 10 \
  --experiment-id cpu-smoke --checkpoint-dir results/cpu-smoke/checkpoints \
  --control-dir results/cpu-smoke/control --log-csv results/cpu-smoke/training.csv
./measure.py --results-dir results/cpu-smoke
```

`--world-size` is an assertion: it must match torchrun's `WORLD_SIZE`, so a
mistyped topology fails before process-group initialisation. Leave `--device`
unset for the real CUDA/NCCL experiment.

Illustrative two-node launch (run once per node with node rank 0/1 and a shared
checkpoint path):

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --rdzv-id=fi-25-a --rdzv-backend=c10d --rdzv-endpoint=node-a:29500 \
  --max-restarts=0 train.py --experiment-id fi-25-a --attempt 1 \
  --checkpoint-interval 25 --checkpoint-dir /shared/fi-25-a/checkpoints \
  --control-dir /shared/fi-25-a/control --log-csv /shared/fi-25-a/training.csv
```

After a killed run, repeat the command with `--attempt 2 --resume latest`. Use a
distinct shared directory and experiment ID per repetition. Read `METHOD.md`
before using `--really`; network injection can sever access if the wrong
interface or peer is chosen.

On 2026-09-09 this code ran under torch 2.14.0 on the GPU-less Mac with
`torchrun`, CPU/Gloo, one rank, 30 steps, and checkpoints every 10 steps. A live
rank was SIGKILLed after checkpoint 10, attempt 2 resumed from `latest`, reached
checkpoint 30, and `measure.py` produced its Markdown table from the appended
CSV. A separate uninterrupted/resumed comparison produced identical final
model and optimizer state. Generated CPU results remain ignored by Git.

CUDA, NCCL, Linux `inject.sh` PID validation/signalling, multiple ranks/nodes,
rendezvous across hosts, shared storage, network faults, NCCL watchdog behaviour,
and GPU performance remain untested. The Mac result does not establish provider
permission, hardware recovery behaviour, or automated job relaunch. Synthetic
fixtures are explicitly not experiment results.
