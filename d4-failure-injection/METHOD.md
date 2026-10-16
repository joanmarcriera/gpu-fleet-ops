# Failure-injection method

## Question and metric

Measure how checkpoint interval changes lost work and recovery overhead for the
same deterministic DDP workload on two 8×H100 nodes. Llama 3 defines effective
training time as time spent on useful training divided by elapsed time
([arXiv:2407.21783, §3.3.4](https://arxiv.org/abs/2407.21783)). This small study
uses the requested operational estimator:

`effective ratio = unique useful steps × median successful step time / wall time`

Unique step numbers exclude replayed steps after rollback. Median step time is
reported because the experiment has fixed model/batch shape and it resists the
injected tail. This estimator is comparable across this experiment only; it is
not a reproduction of Meta's 16K-GPU measurement.

## Controlled variables

Keep provider/SKU, node image, driver, CUDA, torch, NCCL, firmware, topology,
GPU clocks/power, network environment, model arguments, seed, global batch,
world size, total target steps, and warm-up treatment constant. Record all of
them. Use exclusive drained nodes and a shared durable checkpoint directory.
Run at least three repetitions per checkpoint interval and rotate experiment
order to reduce thermal/time-order bias.

## Procedure

1. Choose two intervals, initially 25 and 100 steps. Give every repetition a
   distinct experiment ID and one CSV file that all restart attempts append to.
2. Launch with `torchrun --nnodes=2 --nproc-per-node=8 --max-restarts=0 ...`.
   Record the full command and both node logs.
3. After the same post-warm-up step in each run, call `inject.sh`. Test SIGKILL,
   SIGSTOP/SIGCONT, node-local network drop, and an NCCL watchdog timeout as
   separate experiments. Never combine faults in the primary comparison.
4. Let torchrun fail, preserve its detection log, then relaunch manually with
   `--attempt 2 --resume latest` and the same `--log-csv`. A scheduler study may
   replace manual relaunch, but must identify its queue/prologue contribution.
5. Continue to the same target step. Run `measure.py`, inspect duplicate/missing
   records, and retain raw CSV/checkpoints/launcher logs with the table.

For the NCCL timeout case, first run `inject.sh --action nccl-timeout` and source
the generated environment **before** torchrun. Then rerun the action with
`--really`; it pauses a rank longer than the configured 30-second process-group
watchdog. Environment cannot be injected into an already-running process.

The network action inserts a narrow OUTPUT drop to one peer and always attempts
to delete it using a trap. It only executes with `--really`. Use a provider
console as a recovery path, select the data-plane interface and peer explicitly,
and record the iptables rules before and after. Do not use it on a management
interface or where provider policy forbids packet filtering.

## Report

For each repetition report checkpoint interval, action, injection step/time,
time to detection, detection-to-first-recovered-step time, total wall time,
unique useful steps, replayed steps, median/p50/p95 step time, effective ratio,
checkpoint size/time, exit reason, and whether recovery was automatic or manual.
Show median and range across repetitions, plus raw tables. Compare cautiously
with Llama 3's reported >90% at up to 16K H100s: scale, workload, automation,
fault mix, and observation window are fundamentally different.

## Validation status

As of 2026-09-09 the job has imported torch 2.14.0, formed a one-rank CPU/Gloo
process group under torchrun, saved and loaded real checkpoints, survived a
manual SIGKILL/relaunch cycle, and produced a measurement table from the real
CPU CSV on the GPU-less Mac. A deterministic control showed identical final
model and optimizer state after uninterrupted and checkpoint-resumed execution.

It has not run on CUDA/NCCL, multiple ranks or hosts, Linux `inject.sh`, shared
storage, or a network. The first hardware run must still validate signal
targeting, watchdog semantics, cross-node checkpoint visibility, measurement
clocks, and restart orchestration before collecting publishable results.
