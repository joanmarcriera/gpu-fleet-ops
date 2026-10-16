#!/usr/bin/env bash
# 10_fabric_checks.sh - inventory of one node: GPUs, PCIe/NVLink topology, HCAs, fabric.
#
# Usage:
#   ./10_fabric_checks.sh            full inventory of THIS node (run on node0 and on node1)
#   ./10_fabric_checks.sh fabric     also the fabric-wide checks (sminfo, ibnetdiscover,
#                                    iblinkinfo, ibqueryerrors, ibdiagnet). Run on node0 only.
#   ./10_fabric_checks.sh counters <tag>
#                                    only the local port counters, saved as 19-counters-<tag>.txt
#                                    (take "before" ahead of the benchmarks and "after" at the end)
#
# Output: one file per command in results/<hostname>/, named NN-tool.txt.
# Safe to run anywhere: missing tools are recorded as SKIPPED. Rehearse it in the ibsim
# container (fabric commands work there) and on the RTX 4060 box (GPU commands work there).
# Nothing here changes the fabric: no counter resets, no port changes.

set -u
source "$(dirname "$0")/00_env.sh"
mode=${1:-node}

# CA names as the IB tools see them (works on real HCAs and on the ibsim fake "ibsim0").
cas() { ibstat -l 2>/dev/null; }

counters() {
    local tag=$1
    local f="$RESULTS/19-counters-$tag.txt"
    : > "$f"
    for ca in $(cas); do
        echo "### $ca port 1" >> "$f"
        # perfquery -x: 64-bit extended counters (data and packets); plain: error counters.
        perfquery -C "$ca" -P 1 >> "$f" 2>&1
        perfquery -x -C "$ca" -P 1 >> "$f" 2>&1
    done
    echo "saved $f"
}

if [ "$mode" = counters ]; then counters "${2:-snapshot}"; exit 0; fi

# --- Node identity and software versions ---
cap 01-os            bash -c 'hostname; uname -a; cat /etc/os-release; uptime'
cap 02-cpu-numa      bash -c 'lscpu; numactl -H 2>/dev/null || true'
cap 03-versions      bash -c '
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "cuda (nvcc): $(nvcc --version 2>/dev/null | tail -1)"
  echo "nccl pkg: $(dpkg -l 2>/dev/null | awk "/libnccl2/{print \$3}" | head -1)"
  echo "ofed: $(ofed_info -s 2>/dev/null || echo none)"
  echo "rdma-core: $(dpkg -l 2>/dev/null | awk "/ rdma-core /{print \$3}")"
  echo "mpi: $(mpirun --version 2>/dev/null | head -1)"
  echo "perftest: $(dpkg -l 2>/dev/null | awk "/ perftest /{print \$3}")"
  echo "hca fw: $(ibv_devinfo 2>/dev/null | awk "/fw_ver/{print \$2}" | sort -u | paste -sd" " -)"'

# --- GPUs ---
cap 04-nvidia-smi    nvidia-smi
cap 05-nvidia-smi-q  nvidia-smi -q
cap 06-topo          nvidia-smi topo -m          # GPU-GPU (NV18) and GPU-NIC (PIX/PXB/SYS) paths
cap 07-nvlink        nvidia-smi nvlink -s        # per-link NVLink speed
cap 08-lspci         bash -c 'lspci | grep -i -E "mellanox|nvidia"'

# --- RDMA devices on this node ---
cap 10-ibstat        ibstat
cap 11-ibv_devinfo   ibv_devinfo -v
cap 12-ibdev2netdev  ibdev2netdev               # mlx5_N -> netdev name (from mlnx-tools)
cap 13-rdma-link     rdma link show
cap 14-gdr           bash -c '
  echo "## GPUDirect RDMA kernel module (nvidia_peermem) or the older nv_peer_mem:"
  lsmod | grep -E "nvidia_peermem|nv_peer_mem" || echo "not loaded"
  echo "## PCIe ACS on bridges (ACS on can block GPU-NIC peer traffic):"
  lspci -vvv 2>/dev/null | grep -c "ACSCtl:.*SrcValid+" || true'
counters before

# --- Fabric-wide checks: node0 only, read-only ---
if [ "$mode" = fabric ]; then
    cap 20-sminfo        sminfo                             # which SM is master, its LID and priority
    cap 21-ibnetdiscover timeout 300 ibnetdiscover          # every switch and HCA the SM sees
    cap 22-iblinkinfo    timeout 300 iblinkinfo             # every link: width, speed, state
    cap 23-ibqueryerrors timeout 300 ibqueryerrors          # ports whose error counters are non-zero
    # ibdiagnet: the full fabric health check. No --pc: that flag RESETS counters fabric-wide,
    # which is not ours to do on a rented fabric.
    cap 24-ibdiagnet     timeout 900 ibdiagnet -o "$RESULTS/ibdiagnet"
fi

echo "done: $(ls "$RESULTS" | wc -l | tr -d ' ') files in $RESULTS"
