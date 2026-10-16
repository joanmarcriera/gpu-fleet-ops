#!/usr/bin/env bash
# Usage: ./40-fabric.sh [--dry-run] [--results-dir DIR] [--wait-seconds N]
# Run fabric diagnostics and compare InfiniBand error/link-flap counters.

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=d3-acceptance/lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

DRY_RUN=0
RESULTS_OVERRIDE=""
WAIT_SECONDS=60
usage() { sed -n '2,3p' "$0" | sed 's/^# //'; }

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --results-dir) RESULTS_OVERRIDE=$2; shift 2 ;;
    --wait-seconds) WAIT_SECONDS=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

prepare_results_dir "$SCRIPT_DIR" "$RESULTS_OVERRIDE"
for command_name in perfquery ibdiagnet python3; do require_command "$command_name"; done

capture_command "$RESULTS_DIR/perfquery-before.txt" perfquery -x -a
ibdiagnet_output="$RESULTS_DIR/ibdiagnet"
if ((DRY_RUN)); then
  printf 'mkdir -p %q\n' "$ibdiagnet_output"
else
  mkdir -p "$ibdiagnet_output"
fi
capture_command "$RESULTS_DIR/ibdiagnet.log" ibdiagnet --output_path "$ibdiagnet_output"
if ((DRY_RUN)); then
  printf 'sleep %q\n' "$WAIT_SECONDS"
else
  sleep "$WAIT_SECONDS"
fi
capture_command "$RESULTS_DIR/perfquery-after.txt" perfquery -x -a

delta_command=(python3 "$SCRIPT_DIR/fabric_delta.py" --before "$RESULTS_DIR/perfquery-before.txt" --after "$RESULTS_DIR/perfquery-after.txt" --output "$RESULTS_DIR/fabric-counter-delta.tsv")
if ((DRY_RUN)); then
  printf '%q ' "${delta_command[@]}"
  printf '\n'
else
  "${delta_command[@]}"
fi
