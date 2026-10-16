#!/usr/bin/env bash
# Usage: ./20-nccl-tests.sh [options]; run with --help for launcher details.
# Build pinned nccl-tests, run single-node, then run a multi-node collective.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=d3-acceptance/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

DRY_RUN=0
RESULTS_OVERRIDE=""
LAUNCHER=mpirun
HOSTS=""
GPUS_PER_NODE=8
NODES=2
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NCCL_TESTS_REF=v2.16.9
BUILD_DIR="$SCRIPT_DIR/.build/nccl-tests"

usage() {
  sed -n '2,3p' "$0" | sed 's/^# //'
  cat <<'EOF'
Options:
  --launcher mpirun|torchrun   Multi-node launcher (default: mpirun).
  --hosts HOST1,HOST2          Required for mpirun; shared build path assumed.
  --nodes N                    Node count (default: 2).
  --gpus-per-node N            GPUs per node (default: 8).
  --node-rank N                This node's torchrun rank.
  --master-addr HOST           Torchrun rendezvous host.
  --master-port PORT           Torchrun rendezvous port.
  --results-dir DIR            Put output in this full run directory.
  --build-dir DIR              Clone/build directory.
  --dry-run                    Print commands without checking dependencies.
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --results-dir) RESULTS_OVERRIDE=$2; shift 2 ;;
    --launcher) LAUNCHER=$2; shift 2 ;;
    --hosts) HOSTS=$2; shift 2 ;;
    --nodes) NODES=$2; shift 2 ;;
    --gpus-per-node) GPUS_PER_NODE=$2; shift 2 ;;
    --node-rank) NODE_RANK=$2; shift 2 ;;
    --master-addr) MASTER_ADDR=$2; shift 2 ;;
    --master-port) MASTER_PORT=$2; shift 2 ;;
    --build-dir) BUILD_DIR=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ $LAUNCHER != mpirun && $LAUNCHER != torchrun ]]; then
  printf 'Launcher must be mpirun or torchrun\n' >&2
  exit 2
fi
if [[ $LAUNCHER == mpirun && -z $HOSTS ]]; then
  printf '%s\n' '--hosts is required with --launcher mpirun' >&2
  exit 2
fi

prepare_results_dir "$SCRIPT_DIR" "$RESULTS_OVERRIDE"
for command_name in git make; do require_command "$command_name"; done
if [[ $LAUNCHER == mpirun ]]; then require_command mpirun; else require_command torchrun; fi

if ((DRY_RUN)); then
  printf 'git clone --branch %q --depth 1 %q %q\n' "$NCCL_TESTS_REF" https://github.com/NVIDIA/nccl-tests.git "$BUILD_DIR"
  printf 'make -C %q -j MPI=1 MPI_HOME=%q\n' "$BUILD_DIR" "${MPI_HOME:-/usr}"
else
  if [[ ! -d $BUILD_DIR/.git ]]; then
    mkdir -p "$(dirname "$BUILD_DIR")"
    git clone --branch "$NCCL_TESTS_REF" --depth 1 https://github.com/NVIDIA/nccl-tests.git "$BUILD_DIR"
  fi
  make -C "$BUILD_DIR" -j MPI=1 MPI_HOME="${MPI_HOME:-/usr}"
fi

binary="$BUILD_DIR/build/all_reduce_perf"
single=("$binary" -b 8 -e 8G -f 2 -g "$GPUS_PER_NODE")
capture_command "$RESULTS_DIR/nccl-single-node.log" env NCCL_DEBUG=INFO "${single[@]}"

if [[ $LAUNCHER == mpirun ]]; then
  ranks=$((NODES * GPUS_PER_NODE))
  multi=(mpirun -np "$ranks" --map-by "ppr:${GPUS_PER_NODE}:node" --host "$HOSTS" -x NCCL_DEBUG=INFO)
  [[ -n ${NCCL_SOCKET_IFNAME:-} ]] && multi+=(-x NCCL_SOCKET_IFNAME)
  [[ -n ${NCCL_IB_HCA:-} ]] && multi+=(-x NCCL_IB_HCA)
  multi+=("$binary" -b 8 -e 8G -f 2 -g 1)
else
  multi=(torchrun --nnodes "$NODES" --nproc-per-node "$GPUS_PER_NODE" --node-rank "$NODE_RANK" --rdzv-id "${RUN_ID:-acceptance}" --rdzv-backend c10d --rdzv-endpoint "${MASTER_ADDR}:${MASTER_PORT}" "$SCRIPT_DIR/torch_all_reduce.py" --min-bytes 8 --max-bytes 1G)
fi
capture_command "$RESULTS_DIR/nccl-multi-node-${LAUNCHER}.log" env NCCL_DEBUG=INFO "${multi[@]}"
