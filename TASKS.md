# Tasks

- [x] D1 observability stack is complete, pinned, documented, and statically validated.
- [x] D3 acceptance runbook and scripts dry-run successfully on macOS.
- [x] D4 deterministic DDP experiment and analysis tools are complete and testable without a GPU.
- [x] Every repository file was read during the 2026-09-09 operator review.
- [x] All three pinned Docker Hub tags return active HTTP API records.
- [x] Every configured/dashboard/alert DCGM name is present in the pinned
  exporter defaults or NVIDIA DCGM field documentation.
- [x] RTX 4060 unavailable and conditional field expectations are documented.
- [x] D3 CLI flags are checked against upstream documentation and the ibdiagnet
  output directory is created explicitly.
- [x] D4 runs through CPU/Gloo under torchrun, survives a real kill/relaunch,
  resumes from a checkpoint, and produces the measurement table.
- [x] All README files explicitly distinguish Mac validation from hardware validation.
- [x] Shellcheck and Python byte-compilation pass across the repository.
- [x] Final diffs are reviewed and logical Conventional Commits exist locally.
- [ ] Run D1 on a local RTX 4060 and replace expected field availability
  with captured `/metrics` evidence.
- [ ] Run D3 and D4 on a drained single H100 node before the two-node test.
- [ ] Obtain exact provider thresholds and run the full two-node H100/InfiniBand
  acceptance and failure-injection programme.
