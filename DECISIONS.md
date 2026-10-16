# Decisions

## 2026-09-09: Add CPU/Gloo as a recovery-validation path only

- Decision: keep CUDA/NCCL as `train.py`'s default, add explicit `--device cpu`
  for local execution, and make `--world-size` an optional assertion against
  torchrun's `WORLD_SIZE`.
- Alternatives considered: leave checkpoint/resume entirely dry-run-only on
  non-GPU hosts; silently fall back from CUDA to CPU; make CPU the default.
- Rationale: a real CPU process group can exercise model, optimizer, atomic
  checkpoint, interruption, resume, CSV, and measurement behaviour on this Mac.
  Explicit selection avoids mistaking that smoke test for GPU/NCCL evidence.
- Consequences: local recovery mechanics are now testable and a topology typo
  fails early, while CUDA/NCCL remains the unchanged experiment path.
- Revisit when: hardware tests reveal backend-specific checkpoint, timeout, or
  process-lifecycle differences.

## 2026-09-09: Pin the observability stack to verified Docker Hub references

- Decision: use explicit `docker.io` references for
  `nvidia/dcgm-exporter:4.4.1-4.6.0-ubuntu22.04`,
  `prom/prometheus:v3.5.0`, and `grafana/grafana:12.1.1`.
- Alternatives considered: retain the NVIDIA NGC registry reference; move to
  newer unreviewed image versions; pin immutable digests in this review.
- Rationale: Docker Hub's tag API returned HTTP 200 for all three exact tags on
  2026-09-09. The previous dcgm-exporter reference used NGC even though this
  review's required source was Docker Hub. No tag itself required correction.
- Consequences: all three image sources now match the repositories verified by
  the review. Existence does not prove runtime compatibility with the target
  NVIDIA driver, Container Toolkit, or host architecture.
- Revisit when: upgrading any component or when the target host validation can
  replace tag pins with tested per-platform digest pins.

## 2026-09-09: Keep valid legacy DCGM exporter field names

- Decision: retain the field names accepted by dcgm-exporter 4.4.1-4.6.0 and
  label their RTX 4060 applicability rather than renaming them to newer DCGM
  canonical aliases.
- Alternatives considered: rename fields to the latest canonical identifiers;
  remove H100-only ECC and memory-temperature series from the shared dashboard.
- Rationale: every selected name exists in the pinned exporter's shipped
  default-counter list or NVIDIA's field-identifier documentation. NVIDIA notes
  that catalogue presence does not guarantee device emission; retaining the
  H100 fields keeps one useful fleet view while explicit labels prevent absent
  GeForce series from being mistaken for zero.
- Consequences: the RTX 4060 will have absent/non-applicable series for ECC and
  memory temperature, and violation/event telemetry remains a live hardware
  check. Prometheus expressions tolerate absent series.
- Revisit when: the stack is run on the RTX 4060 and the observed `/metrics`
  output can replace the documented expectations.

## 2026-09-08: Treat hardware results as a separate validation stage

- Decision: build commands, parsers, dashboards, and runbooks now, but do not
  populate performance thresholds or claim runtime success without GPU evidence.
- Alternatives considered: use published H100 figures as pass thresholds; invent
  representative fixture results.
- Rationale: provider topology, firmware, image versions, and fabric design change
  observed results. The current machine has no NVIDIA GPU.
- Consequences: performance gates remain explicit placeholders until a rented run;
  structural and dry-run checks can still be completed locally.
- Revisit when: the exact RunPod or Nebius 2-node cluster is allocated and provider
  acceptance figures are available.
