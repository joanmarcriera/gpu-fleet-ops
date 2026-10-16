# H100 node-acceptance runbook

This runbook is for two drained 8×H100 SXM nodes joined by InfiniBand. Run it
before admitting either node to a scheduler. It records evidence; it does not
replace the provider's field diagnostics, burn-in procedure, or RMA decision.

## Prerequisites

- Exact node SKU, GPU count, HBM size, CPU/NUMA layout, NIC count/speed, rail
  mapping, image ID, driver/CUDA/DCGM/NCCL/OFED/firmware versions, and expected
  topology are agreed with the provider.
- Both nodes are drained and exclusive. DCGM 3/4, HPL, and collective tests are
  disruptive and may take hours.
- Passwordless SSH (for Open MPI), identical shared paths, host keys, MPI/NCCL
  ABI compatibility, NVIDIA Container Toolkit, DCGM diagnostics, HPC-X/OFED,
  `ibdiagnet`, and `perfquery` are available.
- The provider has supplied its acceptance baseline and permitted diagnostic
  operations. Never reset ports, change firmware, or clear counters merely to
  make a test pass.
- Export one run ID for all stages: `export RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)`.

## Order of tests

1. On both nodes, run `./00-inventory.sh`; compare the two inventories before
   applying load. Stop on missing GPUs/NICs, wrong topology, or version skew.
2. Run `./40-fabric.sh --wait-seconds 60` at idle to establish clean fabric
   counters and detect link recovery/down events.
3. Run `./10-dcgm-diag.sh` on each node. Level 4 can take more than two hours on
   eight GPUs; preserve JSON and `/var/log/nvidia-dcgm/nvvs.log`.
4. Run `./20-nccl-tests.sh --launcher mpirun --hosts node-a:8,node-b:8` from the
   controlling node. The single-node phase executes first. If MPI is not
   available, launch the script on both nodes with `--launcher torchrun`, the
   same rendezvous address, and node ranks 0/1; that fallback is a PyTorch NCCL
   benchmark, not `nccl-tests`.
5. Copy and tune `HPL.dat.example` to the exact memory/topology, then run
   `./30-hpl.sh --dat HPL.dat` once on each node. Do not blindly use the sample
   matrix size on a different H100 memory configuration.
6. Re-run `40-fabric.sh` after load. Generate the summary with
   `./90-report.py --results-dir results/$RUN_ID` and attach raw artifacts.

## Pass/fail gates

| Area | Pass | Fail or hold | Evidence source |
|---|---|---|---|
| Inventory | 8 expected H100s and all HCAs per node; topology and versions match the provider manifest | Missing device, unexpected PCIe/NVLink path, link not Active/expected rate, unexplained version skew | Provider cluster manifest **[CITATION REQUIRED]** |
| DCGM | Every applicable level 3 and 4 test is `Pass` | Any `Fail`; any unexplained `Skip`, unavailable plugin, XID, DBE, thermal event, or PCIe replay breach is a hold | NVIDIA DCGM diagnostic release docs plus provider policy **[CITATION REQUIRED]** |
| NCCL correctness | `#wrong` is zero at every size | Any wrong result or rank failure | NVIDIA nccl-tests documentation **[CITATION REQUIRED]** |
| NCCL performance | Single- and two-node large-message algbw/busbw meet the provider baseline with run-to-run variance ≤ **TBD %** | Below **TBD GB/s**, variance above **TBD %**, or unexplained sawtooth/rail imbalance | Exact cluster/provider baseline **[CITATION REQUIRED]** |
| HPL | Residual passes and result ≥ **TBD TFLOPS per 8×H100 node** after exact HPL configuration is recorded | Residual failure, GPU error, or result below the agreed threshold | NVIDIA/provider result for the exact H100 SXM SKU **[CITATION REQUIRED]** |
| Fabric | No increment in link-down/recovery, symbol, receive, constraint, discard, or VL15-drop counters; `ibdiagnet` has no new errors | Any link flap or new critical counter; any unresolved `ibdiagnet` error | NVIDIA Networking counter guidance and provider fabric policy **[CITATION REQUIRED]** |

Do not substitute a generic published peak for the performance placeholders.
NCCL `busbw` is a corrected algorithm metric and becomes harder to interpret
with hierarchical or NVLink SHARP algorithms; compare like-for-like software,
message sizes, topology, and algorithm settings.

## Drain criteria

Keep the node drained immediately for any missing/inaccessible GPU or HCA,
critical XID, uncorrectable ECC, DCGM failure, HPL residual failure, collective
wrong result/hang, InfiniBand link flap/error-counter increase, thermal shutdown,
or unexplained material performance regression. Also hold—not pass—a node when
a required test is skipped, its parser cannot read the artifact, or a threshold
is still TBD. Preserve logs and timestamps, open a provider ticket, and only
return the node after the fault is explained and the failed stage plus its
downstream load stages pass again.

## Validation status

As of 2026-09-09 none of these commands has run on Linux, NVIDIA hardware, an
H100, a second node, MPI, Slurm/Soperator, or InfiniBand. The thresholds are
intentionally incomplete until the exact rental provider supplies a baseline.
Local dry-run, parser-fixture, shell, and Python syntax checks passed on the
GPU-less Mac; upstream documentation checks validate flag syntax only.
