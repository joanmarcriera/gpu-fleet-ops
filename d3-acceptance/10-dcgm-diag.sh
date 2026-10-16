#!/usr/bin/env bash
# Usage: ./10-dcgm-diag.sh [--dry-run] [--results-dir DIR]
# Run administrator-led DCGM diagnostic suites 3 and 4 with JSON output.

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
require_command dcgmi
capture_command "$RESULTS_DIR/dcgm-diag-r3.json" dcgmi diag -r 3 -j
capture_command "$RESULTS_DIR/dcgm-diag-r4.json" dcgmi diag -r 4 -j
