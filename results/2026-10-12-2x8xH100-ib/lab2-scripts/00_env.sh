# shellcheck shell=bash
# 00_env.sh - shared settings and helpers for the lab 2 scripts.
#
# Usage:   source ./00_env.sh          (source it; do not run it)
#
# Set these before sourcing, or edit the defaults below on the day:
#   NODES="<ip> <ip>"   private IPs of node0 and node1. node0 first: you run
#                                 mpirun and perftest from node0.
#   PROVIDER, REGION, INSTANCE_TYPE, PRICE_USD_PER_HOUR, METER_START
#                                 go into the manifest written by 90_collect.sh.
#   FABRIC=ib | roce              which RDMA link layer to use (default ib).
#   GID_INDEX=3                   RoCE only: the RoCE v2 GID index (see show_gids).
#   PRINT_ONLY=1                  print the commands instead of running them (dry run).
#
# What it gives you:
#   $RESULTS        results/<hostname>/ : every script writes its raw output here
#   $NCCL_IB_HCA    the active RDMA devices, exact-match form (=mlx5_0:1,mlx5_3:1,...)
#   $OOB_IF         the Ethernet interface for SSH/MPI bootstrap (the default route)
#   cap NN-name cmd args...   run a command, save the output with a header, never abort

LAB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- Facts for the manifest. Fill these on Monday, before the first benchmark. ---
export PROVIDER=${PROVIDER:-"Nebius"}
export REGION=${REGION:-"eu-north1"}
# [annotation, added when curating] Not bare metal: the nodes were KVM VMs and the
# ConnectX HCAs are SR-IOV virtual functions (see system/08-lspci.txt).
export INSTANCE_TYPE=${INSTANCE_TYPE:-"2 x 8x H100 SXM, VMs with SR-IOV VFs, InfiniBand)"}
export PRICE_USD_PER_HOUR=${PRICE_USD_PER_HOUR:-"<removed>"}
export METER_START=${METER_START:-"2026-10-12T12:02+01:00)"}

# --- Cluster layout ---
export NODES=${NODES:-"node0 node1"}
export NODE0=${NODES%% *}
export NODE1=${NODES##* }
export GPUS_PER_NODE=${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')}
export FABRIC=${FABRIC:-ib}
export GID_INDEX=${GID_INDEX:-3}

# --- Where things live ---
export RESULTS=${RESULTS:-$LAB_DIR/results/$(hostname -s)}
# nccl-tests commit to build (fill in on Friday after the RTX 4060 dry run: git -C nccl-tests rev-parse HEAD).
export NCCL_TESTS_REF=${NCCL_TESTS_REF:-afd59abab774a6a4b492eb344bd0c845a0b7bb4d}
export NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-$HOME/nccl-tests}
export MAXBYTES=${MAXBYTES:-8G}      # largest NCCL message; use 256M on the RTX 4060 dry run
mkdir -p "$RESULTS"

# rdma_devs: list RDMA devices whose port 1 is ACTIVE and whose link layer matches FABRIC.
# Why: a node has both fabric HCAs and front-end Ethernet NICs; NCCL must use only the fabric ones.
rdma_devs() {
    local want=InfiniBand d port netdev
    [ "$FABRIC" = roce ] && want=Ethernet
    for d in /sys/class/infiniband/*; do
        [ -e "$d" ] || continue
        port=$d/ports/1
        [ "$(cat "$port/link_layer" 2>/dev/null)" = "$want" ] || continue
        grep -q ACTIVE "$port/state" 2>/dev/null || continue
        # RoCE: skip the NIC that carries the default route (that is the front-end network).
        if [ "$FABRIC" = roce ]; then
            netdev=$(ls "$d/device/net" 2>/dev/null | head -1)
            [ -n "$netdev" ] && [ "$netdev" = "$OOB_IF" ] && continue
        fi
        basename "$d"
    done
}

# Ethernet interface used by SSH and by MPI/NCCL bootstrap: the one the kernel uses to reach
# node1's private IP (falls back to the default route). Why not just the default route: the
# private network may sit on a different NIC from the public one.
route_dev() { awk '{for(i=1;i<NF;i++) if($i=="dev"){print $(i+1); exit}}'; }
OOB_IF=${OOB_IF:-$(ip -o route get "$NODE1" 2>/dev/null | route_dev)}
[ -n "$OOB_IF" ] && [ "$OOB_IF" != lo ] || OOB_IF=$(ip route show default 2>/dev/null | route_dev)
export OOB_IF

# NCCL settings. Why each one:
#   NCCL_DEBUG=INFO          the log names the transport (NET/IB vs NET/Socket) and GDRDMA.
#   NCCL_DEBUG_SUBSYS        keep the log to init and network lines, so it stays readable.
#   NCCL_IB_HCA              "=" means exact names; without it mlx5_1 would also match mlx5_10.
#   NCCL_SOCKET_IFNAME       bootstrap over the front-end Ethernet, never over docker0 or lo.
#   NCCL_NET_GDR_LEVEL=PXB   allow GPUDirect RDMA when GPU and NIC share a PCIe switch path.
#   NCCL_TOPO_DUMP_FILE      NCCL writes the topology it detected; good evidence for the note.
#                            Written into $RESULTS (per host), never a shared /tmp path.
DEVS=$(rdma_devs | sed 's/$/:1/' | paste -sd, -)
if [ -n "$DEVS" ]; then export NCCL_IB_HCA="=$DEVS"; fi
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
[ -n "$OOB_IF" ] && export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$OOB_IF}
export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-PXB}
mkdir -p "$RESULTS"
export NCCL_TOPO_DUMP_FILE=${NCCL_TOPO_DUMP_FILE:-$RESULTS/nccl-topology.xml}
[ "$FABRIC" = roce ] && export NCCL_IB_GID_INDEX=$GID_INDEX

# Open MPI refuses to run as root unless told; only matters inside the dry-run container.
if [ "$(id -u)" = 0 ]; then
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi

# ssh_check: node0 must reach node1 with no prompt. mpirun and the perftest server cannot
# answer a password or host-key question; they would just hang while the meter runs.
# accept-new stores node1's host key the first time, without asking.
ssh_check() {
    [ "${PRINT_ONLY:-0}" = 1 ] && return 0
    ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$NODE1" true \
        || { echo "ssh to $NODE1 failed or prompted: fix step B3 first" >&2; exit 1; }
}

# cap NN-name cmd [args...]
# Runs the command, writes host, date, command, output and exit code to $RESULTS/NN-name.txt.
# A missing tool is recorded as SKIPPED instead of stopping the script: the same scripts run
# on the ibsim container, the RTX 4060 box and the rented nodes.
cap() {
    local name=$1; shift
    local f="$RESULTS/$name.txt"
    if [ "${PRINT_ONLY:-0}" = 1 ]; then echo "[print-only] $* > $f"; return 0; fi
    {
        echo "# host: $(hostname -s)   date: $(date '+%Y-%m-%dT%H:%M:%S%z')"
        echo "# cmd:  $*"
        echo
    } > "$f"
    if command -v "$1" >/dev/null 2>&1; then
        "$@" >> "$f" 2>&1
        echo "# exit: $?" >> "$f"
    else
        echo "SKIPPED: $1 is not installed on this host" >> "$f"
    fi
    echo "saved $f"
}

echo "lab2 env: host=$(hostname -s) nodes=[$NODES] gpus/node=${GPUS_PER_NODE:-0} fabric=$FABRIC"
echo "          NCCL_IB_HCA=${NCCL_IB_HCA:-<none found>} OOB_IF=${OOB_IF:-<none>} results=$RESULTS"
