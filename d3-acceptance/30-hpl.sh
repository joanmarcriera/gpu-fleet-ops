#!/usr/bin/env bash
# Usage: ./30-hpl.sh [--dry-run] [--results-dir DIR] [--dat FILE]
# Run NVIDIA HPL once per drained 8xH100 node using a pinned benchmark image.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=d3-acceptance/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

DRY_RUN=0
RESULTS_OVERRIDE=""
DAT_FILE="$SCRIPT_DIR/HPL.dat.example"
IMAGE=nvcr.io/nvidia/hpc-benchmarks:26.02
usage() { sed -n '2,3p' "$0" | sed 's/^# //'; }

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --results-dir) RESULTS_OVERRIDE=$2; shift 2 ;;
    --dat) DAT_FILE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

prepare_results_dir "$SCRIPT_DIR" "$RESULTS_OVERRIDE"
require_command docker
if ((DRY_RUN == 0)) && [[ ! -f $DAT_FILE ]]; then
  printf 'HPL input not found: %s\n' "$DAT_FILE" >&2
  exit 1
fi

command=(docker run --rm --gpus all --ipc host --ulimit memlock=-1 --ulimit stack=67108864 -v "$DAT_FILE:/workspace/HPL.dat:ro" "$IMAGE" /workspace/hpl.sh --dat /workspace/HPL.dat)
capture_command "$RESULTS_DIR/hpl.log" "${command[@]}"
