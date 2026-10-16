#!/usr/bin/env bash
# Usage: source lib/common.sh from a D3 stage script.
# Shared output and dry-run helpers; this file is not intended to be executed.

set -euo pipefail

stage_timestamp() {
  date -u +%Y%m%dT%H%M%SZ
}

prepare_results_dir() {
  local script_dir=$1
  local override=${2:-}
  local run_id=${RUN_ID:-$(stage_timestamp)}
  if [[ -n $override ]]; then
    RESULTS_DIR=$override
  else
    RESULTS_DIR="$script_dir/results/$run_id"
  fi
  if ((DRY_RUN)); then
    printf 'mkdir -p %q\n' "$RESULTS_DIR"
  else
    mkdir -p "$RESULTS_DIR"
  fi
  export RESULTS_DIR
}

print_redirected_command() {
  local output_file=$1
  shift
  printf '%q ' "$@"
  printf '> %q 2>&1\n' "$output_file"
}

capture_command() {
  local output_file=$1
  shift
  if ((DRY_RUN)); then
    print_redirected_command "$output_file" "$@"
  else
    "$@" >"$output_file" 2>&1
  fi
}

require_command() {
  local name=$1
  if ((DRY_RUN == 0)) && ! command -v "$name" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$name" >&2
    exit 1
  fi
}

