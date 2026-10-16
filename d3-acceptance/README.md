# D3: node acceptance

The numbered stages inventory a node, run long DCGM diagnostics, measure NCCL
collectives, run NVIDIA HPL, and detect new InfiniBand error counters. Read
`RUNBOOK.md` before allocating the rental; all results for one attempt should
share `RUN_ID`.

```bash
export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
./00-inventory.sh
./40-fabric.sh
./10-dcgm-diag.sh
./20-nccl-tests.sh --launcher mpirun --hosts node-a:8,node-b:8
./30-hpl.sh --dat HPL.dat
./90-report.py --results-dir "results/$RUN_ID"
```

Run the dry-run path before renting anything:

```bash
./00-inventory.sh --dry-run
./10-dcgm-diag.sh --dry-run
./20-nccl-tests.sh --dry-run --launcher mpirun --hosts node-a:8,node-b:8
./30-hpl.sh --dry-run
./40-fabric.sh --dry-run --wait-seconds 0
./90-report.py --dry-run
./torch_all_reduce.py --dry-run
./fabric_delta.py --dry-run --before before.txt --after after.txt --output delta.tsv
```

`nccl-tests` is pinned to `v2.16.9`; NVIDIA HPC Benchmarks is pinned to `26.02`.
Confirm both are compatible with the rental image before the paid clock starts.
The MPI route is authoritative for nccl-tests. The torchrun route measures a
PyTorch/NCCL all-reduce and labels its output accordingly; it is not a way to
launch the nccl-tests executable.

The command flags were checked against upstream documentation on 2026-09-09:

- DCGM documents `dcgmi diag -r 3|4 -j` for suite selection and JSON output.
- nccl-tests v2.16.9 documents `all_reduce_perf -b 8 -e 8G -f 2 -g N`;
  under MPI, `-g 1` is GPUs per process and process count supplies total ranks.
- ibdiagnet documents `-o|--output_path DIRECTORY`; the script creates that
  directory first rather than relying on tool-specific creation behaviour.
- rdma-core documents `perfquery -x -a`; `-x` selects optional extended port
  counters and `-a` aggregates all ports of the selected local destination.

These checks validate syntax, not command availability, fabric permissions, or
whether the target HCA exposes the optional extended counter attribute.

As of 2026-09-09 this directory is untested on Linux, NVIDIA hardware, H100,
multiple nodes, Docker with NVIDIA runtime, MPI/torchrun multi-node rendezvous,
or InfiniBand. Dry-run and synthetic parser checks on macOS cannot validate
command availability, privileges, ABI compatibility, benchmark performance,
provider networking, or correct HPL sizing.
