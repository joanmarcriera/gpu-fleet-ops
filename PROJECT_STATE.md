# Project state

## Current objective

Hardware validation has started: the fabric and NCCL layer ran on real H100s on
2026-10-12 (see "Verified on H100" below). The remaining D1/D3/D4 hardware work
is still open.

## Verified on H100 (2026-10-12)

Two rented VMs on one GPU cluster, each 8 x H100 80GB HBM3 and 8 InfiniBand HCAs
at 400 Gb/s; driver 570.211.01, CUDA 12.8, NCCL 2.26.2 (nccl-tests; the PyTorch DDP job used its bundled NCCL 2.28.9), nccl-tests `afd59ab`.
Curated, scrubbed raw files and plots: `results/2026-10-12-2x8xH100-ib/`
(`SUMMARY.md` has every number with its file).

- `ib_write_bw`: 395.0 to 395.1 Gb/s on each of 8 rails (98.8% of 400).
- all_reduce busbw, 8 GiB: 1 node 480.5 GB/s; 2 nodes 466.8 to 468.0 GB/s (3 runs each).
- Forced algorithm, 1 node: NVLS 481.0, Ring 368.4, Tree 298.1 GB/s. Forced Ring over 2 nodes: 332.1 GB/s.
- Transport study: one rail only 91.5 GB/s at 1 GiB; InfiniBand disabled (TCP) 1.93 GB/s at 256 MiB.
- all_gather 318.3 GB/s and all-to-all 72.3 GB/s (16 GPUs, 4 GiB).
- ResNet-50 DDP, synthetic data: 3,704 / 28,196 / 56,255 img/s for 1 / 8 / 16 GPUs.
- Zero IB error counters since boot on node0 (sysfs snapshot).

Caveats: node1 was recreated twice (first on a different image, then not attached to the GPU cluster) and the
first perftest pass re-run; default multi-node runs exit 139 at teardown while
`NCCL_NET_PLUGIN=none` runs exit 0 at the same speed (cause not investigated);
n is at most 3 and often 1; the first-pass perftest sweep failed just above
1 MiB; the default transport was the Spectrum-X NCCL plugin shipped in HPC-X.

Not run: HPL, OSU, nvbandwidth, DCGM diagnostics, D1 exporter on an H100, D3 runbook
as written, GPU-memory perftest. `ibdiagnet` and the other management tools were
blocked in the VM (0 nodes discovered; not evidence). **D4 failure injection
(kill/resume on GPUs) was not run**; its only evidence is still the 2026-09-09 CPU/Gloo
exercise. The one-rail and TCP runs are NCCL-setting transport tests, not D4.

## Completed work

- Every repository file was read and the final changes were reviewed.
- D1's three exact Docker Hub tags were queried through the HTTP tag API. All
  returned HTTP 200 with active manifests; no tag version required replacement.
  The dcgm-exporter reference was changed from NGC to the verified Docker Hub
  repository.
- All 15 DCGM/exporter names used by `metrics.csv`, the dashboard, and alerts
  were found in dcgm-exporter 4.4.1-4.6.0's shipped `default-counters.csv` or
  NVIDIA's DCGM field/exporter documentation. No invalid field was removed.
- RTX 4060 ECC and memory-temperature non-applicability, conditional fields,
  and the difference between a valid identifier and an emitted series are now
  explicit.
- D3 flags were checked against the official `dcgmi diag`, nccl-tests v2.16.9,
  ibdiagnet, and rdma-core perfquery documentation. The flags were valid; the
  ibdiagnet output directory is now created explicitly before use.
- D4 gained explicit CPU/Gloo validation while preserving CUDA/NCCL as default.
  `--world-size` now verifies torchrun's real `WORLD_SIZE`.

## Verified on this Mac

- Docker Hub HTTP API: all three requested tags returned active manifests on
  2026-09-09.
- `docker-compose 5.5.0 config --quiet`: pass. Images were not pulled or run.
- Dashboard JSON parse and configured-versus-used metric-set comparison: pass;
  zero dashboard/alert metric names are missing from the collector.
- All shell scripts under D1, D3, and D4: shellcheck pass.
- All five Python scripts: `py_compile` pass under local Python 3.14.7.
- All documented D1/D3/D4 dry-run paths: pass, including both D3 launchers and
  all four D4 injection actions.
- D3 report and fabric parsers: pass against explicitly synthetic fixtures.
- D4 measurement tool: pass against synthetic fixtures and the real CPU CSV.
- A git-ignored `.venv` using Python 3.13.12, torch 2.14.0, and numpy 2.5.3 ran
  a one-rank CPU/Gloo job under torchrun with 30 steps and checkpoints every 10.
- The live rank was verified from its PID file and SIGKILLed after checkpoint
  10 (after it had completed step 17). Attempt 2 resumed from `latest`, wrote
  checkpoints 20 and 30, recorded one clean `run_end`, and produced 30 unique
  steps plus eight replayed post-checkpoint steps in the appended CSV.
- `measure.py` produced the Markdown table for that CSV. A separate deterministic
  control confirmed identical final model and optimizer state for uninterrupted
  versus checkpoint-resumed execution.

## Untested on GPU or target infrastructure (as of 2026-09-09; the H100 section above supersedes the NCCL, perftest and DDP items)

- No image has been pulled or started against NVIDIA Container Toolkit.
- No exporter metric has been observed on a local RTX 4060 or H100; field
  presence, absent-series behaviour, alert firing, and driver/DCGM compatibility
  remain hardware tests.
- No `dcgmi`, nccl-tests, CUDA/NCCL DDP, NVIDIA HPL, `ibdiagnet`, or `perfquery`
  command has executed against GPU/InfiniBand hardware.
- Multi-rank/multi-node rendezvous, MPI, provider networking, NVLink/NVSwitch,
  shared checkpoint storage, Linux `inject.sh` signalling, iptables failure,
  NCCL watchdog timeout, and scheduler restart automation remain untested.
- Acceptance performance thresholds remain TBD until provider/vendor baselines
  for the exact H100 cluster are supplied.

## Files changed in the review

- D1: compose image references, field applicability comments, and README.
- D3: fabric output-directory handling and documented flag verification.
- D4: CPU/Gloo execution path, world-size assertion, README, and method status.
- Root: `.gitignore`, `README.md`, `DECISIONS.md`, this file, and `TASKS.md`.

## Unresolved defects or risks

- Valid upstream syntax and tag existence do not establish installed-version or
  runtime compatibility on the future Linux hosts.
- NVIDIA documents GeForce field watches generally, not guaranteed per-field
  emission. RTX 4060 conditional telemetry must be classified from live output.
- The CPU recovery path cannot validate NCCL failure detection or GPU state.

## Known blockers

D1, D3 (DCGM, HPL) and D4 on GPUs still need another rental or a local RTX 4060.
Performance gates additionally require provider baselines.

## Next recommended action

Run D1 on the drained RTX 4060 first, capture `/metrics`, and update the field
applicability table from observed series. Then execute the D3 runbook and D4
pilot on a single drained H100 node before allocating the full two-node run.
