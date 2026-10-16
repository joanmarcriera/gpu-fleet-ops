# D1: GPU observability

This directory provisions one DCGM Exporter, Prometheus, and Grafana instance
for a single Linux GPU host. Images are pinned; upgrade them deliberately after
checking NVIDIA driver/DCGM compatibility and reading the release notes.

## Run

Prerequisites are Linux, Docker Engine with Compose v2, a supported NVIDIA
driver, and NVIDIA Container Toolkit. Set a non-default Grafana password before
exposing any port:

```bash
export GRAFANA_ADMIN_PASSWORD='set-this-outside-the-repository'
docker compose config
docker compose up -d
curl --fail http://127.0.0.1:9400/metrics
curl --fail http://127.0.0.1:9090/-/ready
```

Grafana is at `http://127.0.0.1:3000`; its datasource and **GPU fleet
overview** dashboard are provisioned automatically. The loopback port bindings
are intentional. Use authenticated TLS termination rather than changing them
to public interfaces casually.

The `GPUCountBelowExpected` rule is set to one GPU for a single-GPU test host. Change its
literal threshold to eight on each intended H100 node, or generate a
node-specific rule in a fleet deployment. Prometheus evaluates alerts, but this
compose file does not include Alertmanager or notification routing.

Capture a baseline only while the GPU is drained. Level 3 is invasive and can
take tens of minutes:

```bash
./baseline.sh --dry-run
./baseline.sh
```

Results are ignored by Git except for the directory marker. Review them for
private host identifiers before publishing.

## Field verification and RTX 4060 limitations

Every configured metric name was checked on 2026-09-09 against the pinned
exporter's shipped `default-counters.csv` or NVIDIA's DCGM field-identifier
documentation. The exporter-only `DCGM_EXP_XID_ERRORS_COUNT` is also listed in
that shipped file. The legacy names used by exporter 4.4.1-4.6.0 remain valid;
newer DCGM documentation links some of them to clearer canonical aliases.

NVIDIA's feature matrix supports field-value watches on GeForce, but its field
documentation explicitly warns that a field's presence does not guarantee a
sample on every GPU/driver. Apply these RTX 4060 labels:

| Fields | RTX 4060 status |
|---|---|
| `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_FB_USED`, `DCGM_FI_DEV_FB_FREE`, `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_XID_ERRORS`, `DCGM_EXP_XID_ERRORS_COUNT`, `DCGM_FI_DRIVER_VERSION` | Expected basic telemetry; still unverified on the target card. |
| `DCGM_FI_DEV_MEMORY_TEMP` | Treat as unavailable on this card; retain for H100. |
| `DCGM_FI_DEV_ECC_DBE_VOL_TOTAL`, `DCGM_FI_DEV_ECC_DBE_AGG_TOTAL` | Unavailable: RTX 4060 has no exposed ECC telemetry. Missing series are not zero errors. |
| `DCGM_FI_DEV_POWER_VIOLATION`, `DCGM_FI_DEV_THERMAL_VIOLATION`, `DCGM_FI_DEV_CLOCKS_EVENT_REASONS` | Valid fields, but device/driver-dependent; hold as unverified until live samples are observed. |

A local RTX 4060 is a consumer GeForce card with 8 GB and is shared with
other containers. It is useful for validating basic field watches: utilisation,
SM clock, framebuffer use, board power, core temperature, XID, and (subject to
driver support) throttle/violation fields. It is not equivalent to an H100
acceptance target:

- RTX 4060 has no user-visible ECC memory, so ECC counters will be absent or
  blank; no-data must not be interpreted as zero errors.
- It has no NVLink/NVSwitch, MIG, or HBM memory-temperature telemetry, so those
  device-specific fields are unavailable.
- NVIDIA documents policy notification only for data-centre/Tesla class GPUs.
- NVIDIA documents GeForce diagnostics at level 1 only. `baseline.sh` includes
  the schedule-requested level 3 command, but it is expected to be unsupported
  or partially skipped on this card. Record that result; do not call it a pass.
- Profiling fields (`DCGM_FI_PROF_*`) are intentionally not selected here.
  Their availability and compatible counter groups must be probed with
  `dcgmi profile -l -i 0`, and they can conflict with Nsight tools.

References: [DCGM feature support by GPU class](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html),
[DCGM exporter metrics](https://docs.nvidia.com/datacenter/dcgm/latest/reference/dcgm-exporter-metrics.html),
and [container installation](https://docs.nvidia.com/datacenter/dcgm/latest/installation/install-dcgm-exporter.html).

## Validation status

As of 2026-09-09 this compose model, provisioning YAML, alert syntax, dashboard
JSON, and shell dry-run have only been checked on a GPU-less macOS machine. The
three pinned Docker Hub tags were confirmed by HTTP API, but the images have not
been pulled or started here. The exporter has not been connected to an NVIDIA
driver, metrics have not been observed, alerts have not been fired, and the
stack has not been run on a local RTX 4060 or an H100 node.
