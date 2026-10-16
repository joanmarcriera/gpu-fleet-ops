# 12 Oct 2026: 2 x 8 H100, InfiniBand, NCCL and perftest

Two rented VMs on one GPU cluster (Nebius, spot): each 8 x H100 80GB HBM3, 8 ConnectX HCAs at
400 Gb/s (NDR 4X). Ubuntu 24.04, driver 570.211.01, CUDA 12.8, NCCL 2.26.2 for the nccl-tests runs (the
PyTorch DDP training job used PyTorch's bundled NCCL 2.28.9, see caveats), HPC-X 2.26,
nccl-tests `afd59ab`. Raw files are in this folder; nothing below is rounded in its favour.
Figures (all drawn by `plots/plot_lab2.py` from the raw files; each cites its sources in its footer):
[`degradation_ladder.png`](plots/degradation_ladder.png) (8 rails vs 1 rail vs TCP, 256 MiB),
[`rail_topology.png`](plots/rail_topology.png) (tenant-view GPU/NIC affinity; switches not observed),
[`busbw_vs_size.png`](plots/busbw_vs_size.png), [`algorithms.png`](plots/algorithms.png),
[`perftest_rails.png`](plots/perftest_rails.png), [`collectives.png`](plots/collectives.png),
[`resnet_scaling.png`](plots/resnet_scaling.png). What was scrubbed: [`SCRUB.md`](SCRUB.md).

## Headline numbers

busbw is `algbw x 2(n-1)/n`, a yardstick and not wire speed. Out-of-place column, largest size
in the file. "Runs" is how many identical runs exist; it is never more than three.

| Measurement | Result | Runs | Raw file |
|---|---|---|---|
| `ib_write_bw`, each of 8 rails, 1 MiB, 4 QPs | 395.02 to 395.09 Gb/s (98.8% of 400) | 1 per rail | [`perftest/51-*`](perftest/) |
| `ib_write_bw` size sweep plateau | 394.97 Gb/s from 1 MiB | 1 | [`52-...sweep`](perftest/52-perftest-write-bw-sweep.txt) |
| `ib_write_lat` 2 B | 2.68 us typical at 2 B, min 2.61; 3.72 us at 1 KiB | 1 | [`53-...lat`](perftest/53-perftest-write-lat.txt) |
| all_reduce, 1 node, 8 GiB | 480.50 / 480.24 / 480.39 GB/s | 3 | [`41-*`](nccl/) |
| all_reduce, 1 node, forced algorithm, 8 GiB | NVLS 480.99, Ring 368.35, Tree 298.07 GB/s (block order in the file: Ring, NVLS, Tree; each block is labelled `# NCCL_ALGO=...`) | 1 each | [`46-*`](nccl/46-nccl-single-algo.txt) |
| all_reduce, 2 nodes, 8 GiB, default | 466.76 / 467.61 / 468.00 GB/s | 3 | [`42-*`](nccl/) |
| all_reduce, 2 nodes, forced Ring, 8 GiB | 332.10 GB/s | 1 | [`45-*`](nccl/45-nccl-multi-ring.txt) |
| all_reduce, 2 nodes, `NCCL_NET_PLUGIN=none`, 8 GiB | 469.0 / 468.0 / 468.9 GB/s | 3 | [`47-*`, `48-*`](nccl/) |
| all_reduce, 2 nodes, one rail (`NCCL_IB_HCA=mlx5_0`), 1 GiB | 91.46 GB/s | 1 | [`43-*`](nccl/43-nccl-inject-one-rail.txt) |
| all_reduce, 2 nodes, InfiniBand disabled (TCP), 256 MiB | 1.93 GB/s | 1 | [`44-*`](nccl/44-nccl-inject-ib-disabled.txt) |
| all_gather, 16 GPUs, 4 GiB | 318.3 GB/s | 1 | [`49-...allgather`](nccl/49-nccl-multi-allgather.txt) |
| all-to-all, 16 GPUs, 4 GiB | 72.3 GB/s | 1 | [`49-...alltoall`](nccl/49-nccl-multi-alltoall.txt) |
| ResNet-50 DDP, synthetic data, 1 / 8 / 16 GPUs | 3,704 / 28,196 / 56,255 img/s (median; 100 / 95.2 / 94.9% scaling) | 2 / 2 / 3 | [`training/`](training/) |
| IB error counters since boot (sysfs) | all 0 on node0 mlx5_0..7 (node1 stated in the run log; no file kept) | 1 | [`system/19-*`](system/19-counters-sysfs-after.txt) |

The one-rail and TCP rows were measured at different message sizes from the 8 GiB rows (see
the plot). The one-rail figure is therefore compared with the 2-node curve at the same size
(444.2 / 445.9 / 445.9 GB/s at 1 GiB), not with the 8 GiB peak.

## Caveats, as found

* **node1 was recreated twice.** The first node1 came up on a different image (driver 580,
  CUDA 13) and the second was not attached to the GPU cluster, so each was deleted and
  recreated. The numbers here are from the third node1, which matches node0. The first perftest pass ran against the deleted node1 and
  was re-run; only the re-run is published.
* **NCCL versions differ by job.** The nccl-tests runs link the system NCCL 2.26.2
  (`system/03-versions.txt`, "NCCL version 2.26.2+cuda12.8" in `nccl/`). The ResNet-50 DDP
  jobs ran on PyTorch's bundled NCCL 2.28.9: every `training/*` RESULT line records
  `"torch": "2.11.0+cu128", "nccl": "2.28.9"`.
* **Exit code 139 at multi-node teardown.** Every default (plugin) multi-node run printed its
  full table and then segfaulted at exit (also the one-rail run). The three
  `NCCL_NET_PLUGIN=none` runs exited 0 at the same speed. The crash was not investigated.
* **The default multi-node transport was `NET/SPCX_v10`**, the Spectrum-X NCCL plugin that ships
  in the HPC-X install, picked up through `LD_LIBRARY_PATH`; `NCCL_NET_PLUGIN=none` selects the
  built-in `NET/IB`. Not a choice made on purpose.
* **n = 3 at most** per configuration, 1 for forced-algorithm, one-rail, TCP, all_gather and
  all-to-all. No confidence intervals are claimed.
* **perftest above 1 MiB.** The first-pass sweep failed with a remote access error just past
  1 MiB. The re-run sweep in this folder completes to 8 MiB with exit 0; cause of the first
  failure not chased. `ib_write_bw --use_cuda` was not available in this perftest build.
* **Torn rows.** A few result rows in the 16-rank captures were split by interleaved debug
  output and are missing (listed in `SCRUB.md`).
* **Management plane blocked.** `sminfo`, `ibnetdiscover`, `iblinkinfo`, `ibqueryerrors` and
  `ibdiagnet` could not open a UMAD port in the VM; `system/24-ibdiagnet.txt` reports 0 nodes
  and proves nothing about the fabric. The diagnosis here is endpoint-side only.
  The likely reason: the nodes are KVM VMs and each HCA is a ConnectX SR-IOV virtual function
  (`system/08-lspci.txt` lists all 8 as "ConnectX Family mlx5Gen Virtual Function";
  `system/10-ibstat.txt` reports CA type MT4126 on every port). A VF has no subnet-management
  (MAD) access, so these tools cannot work from inside the guest; this is not a fabric fault.
  The VMs, not bare metal, are what `00_env.sh` now says (see `SCRUB.md`, annotations).
* **`ibdev2netdev` shows "(Down)".** In `system/12-ibdev2netdev.txt` that refers to the IPoIB
  netdevs (`ibp...`), not to the IB ports. The IB ports themselves are Active/LinkUp at
  400 Gb/s in `system/10-ibstat.txt`, and the traffic ran over them.
* **Training** is ResNet-50 on random tensors: a gentle test of the fabric, not a model run.
  `63-train-2node-long-run3.txt` printed its RESULT, then hung at teardown and was killed by hand.

## Not run

HPL, OSU micro-benchmarks, nvbandwidth, GPU-memory perftest (`--use_cuda`), `ibdiagnet` (blocked),
subnet-manager and switch topology views, and the D4 checkpoint/failure-injection study.
