#!/usr/bin/env bash
# Usage: ./00-inventory.sh [--dry-run] [--results-dir DIR]
# Capture GPU, NVLink, InfiniBand, OS, driver, and firmware inventory.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=d3-acceptance/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

DRY_RUN=0
RESULTS_OVERRIDE=""

usage() { sed -n '2,3p' "$0" | sed 's/^# //'; }

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --results-dir) RESULTS_OVERRIDE=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

prepare_results_dir "$SCRIPT_DIR" "$RESULTS_OVERRIDE"
for command_name in nvidia-smi ibstat ibv_devinfo; do
  require_command "$command_name"
done

capture_command "$RESULTS_DIR/uname.txt" uname -a
capture_command "$RESULTS_DIR/nvidia-smi-q.txt" nvidia-smi -q
capture_command "$RESULTS_DIR/nvidia-smi-topo.txt" nvidia-smi topo -m
capture_command "$RESULTS_DIR/nvidia-smi-nvlink-status.txt" nvidia-smi nvlink --status
capture_command "$RESULTS_DIR/gpu-firmware.csv" nvidia-smi --query-gpu=index,uuid,name,driver_version,vbios_version,pci.bus_id,pci.link.gen.current,pci.link.width.current --format=csv
capture_command "$RESULTS_DIR/ibstat.txt" ibstat
capture_command "$RESULTS_DIR/ibv-devinfo.txt" ibv_devinfo -v

if ((DRY_RUN)); then
  print_redirected_command "$RESULTS_DIR/ofed-version.txt" ofed_info -s
elif command -v ofed_info >/dev/null 2>&1; then
  capture_command "$RESULTS_DIR/ofed-version.txt" ofed_info -s
else
  printf 'ofed_info not installed\n' >"$RESULTS_DIR/ofed-version.txt"
fi
