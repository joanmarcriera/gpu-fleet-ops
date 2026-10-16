#!/usr/bin/env bash
# Usage: ./baseline.sh [--dry-run] [--output-dir DIR]
# Capture a dated, read-only DCGM and nvidia-smi baseline on one GPU node.

set -euo pipefail

DRY_RUN=0
OUTPUT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"

usage() {
  sed -n '2,3p' "$0" | sed 's/^# //'
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --output-dir) OUTPUT_ROOT=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
output_dir="$OUTPUT_ROOT/$timestamp"

print_command() {
  local output_file=$1
  shift
  printf '%q ' "$@"
  printf '> %q 2>&1\n' "$output_file"
}

capture() {
  local output_file=$1
  shift
  if ((DRY_RUN)); then
    print_command "$output_file" "$@"
    return
  fi
  printf 'Running %s -> %s\n' "$1" "$output_file"
  "$@" >"$output_file" 2>&1
}

if ((DRY_RUN)); then
  printf 'mkdir -p %q\n' "$output_dir"
else
  command -v nvidia-smi >/dev/null || { printf 'nvidia-smi not found\n' >&2; exit 1; }
  command -v dcgmi >/dev/null || { printf 'dcgmi not found\n' >&2; exit 1; }
  mkdir -p "$output_dir"
fi

capture "$output_dir/nvidia-smi-q.txt" nvidia-smi -q
capture "$output_dir/dcgm-discovery.txt" dcgmi discovery -l
capture "$output_dir/dcgm-diag-r1.txt" dcgmi diag -r 1
capture "$output_dir/dcgm-diag-r3.txt" dcgmi diag -r 3

if ((DRY_RUN == 0)); then
  printf 'Baseline written to %s\n' "$output_dir"
fi

