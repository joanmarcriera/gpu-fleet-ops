#!/usr/bin/env bash
# Usage: ./inject.sh --action kill|pause|network-drop|nccl-timeout [options]
# Inject a controlled DDP failure; --dry-run prints actions without signalling.

set -euo pipefail

ACTION=""
RANK=1
CONTROL_DIR=results/control
EVENTS_CSV=results/events.csv
EXPERIMENT_ID=checkpoint-50
CHECKPOINT_INTERVAL=50
DURATION=30
DETECT_TIMEOUT=120
INTERFACE=""
PEER=""
ENV_FILE=results/nccl-timeout.env
REALLY=0
DRY_RUN=0

usage() {
  sed -n '2,3p' "$0" | sed 's/^# //'
  cat <<'EOF'
Options:
  --rank N                    Rank whose PID file is used (default: 1).
  --control-dir DIR           Directory written by train.py.
  --events-csv FILE           Append injection/detection events here.
  --experiment-id ID          Join key used by measure.py.
  --checkpoint-interval N     Join key used by measure.py.
  --duration N                Pause/drop duration in seconds.
  --detect-timeout N          How long kill waits for a sibling rank to exit.
  --interface IFACE           Interface for network-drop.
  --peer IP                   Peer IP for network-drop OUTPUT rule.
  --env-file FILE             NCCL timeout environment file to create.
  --really                    Required before changing iptables or forcing timeout.
  --dry-run                   Print commands; do not signal, sleep, or write.
EOF
}

while (($#)); do
  case "$1" in
    --action) ACTION=$2; shift 2 ;;
    --rank) RANK=$2; shift 2 ;;
    --control-dir) CONTROL_DIR=$2; shift 2 ;;
    --events-csv) EVENTS_CSV=$2; shift 2 ;;
    --experiment-id) EXPERIMENT_ID=$2; shift 2 ;;
    --checkpoint-interval) CHECKPOINT_INTERVAL=$2; shift 2 ;;
    --duration) DURATION=$2; shift 2 ;;
    --detect-timeout) DETECT_TIMEOUT=$2; shift 2 ;;
    --interface) INTERFACE=$2; shift 2 ;;
    --peer) PEER=$2; shift 2 ;;
    --env-file) ENV_FILE=$2; shift 2 ;;
    --really) REALLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$ACTION" in kill|pause|network-drop|nccl-timeout) ;; *) usage >&2; exit 2 ;; esac

record_event() {
  local event=$1
  local detail=$2
  if ((DRY_RUN)); then
    printf 'record event=%q detail=%q in %q\n' "$event" "$detail" "$EVENTS_CSV"
    return
  fi
  mkdir -p "$(dirname "$EVENTS_CSV")"
  if [[ ! -s $EVENTS_CSV ]]; then
    printf 'epoch_s,utc,experiment_id,checkpoint_interval,event,rank,detail\n' >"$EVENTS_CSV"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$(date +%s.%N)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$EXPERIMENT_ID" "$CHECKPOINT_INTERVAL" "$event" "$RANK" "$detail" >>"$EVENTS_CSV"
}

rank_pid() {
  local file="$CONTROL_DIR/rank-$RANK.pid"
  [[ -f $file ]] || { printf 'Rank PID file not found: %s\n' "$file" >&2; exit 1; }
  local value
  value=$(<"$file")
  [[ $value =~ ^[0-9]+$ ]] || { printf 'Invalid PID file: %s\n' "$file" >&2; exit 1; }
  printf '%s' "$value"
}

validate_training_pid() {
  local pid=$1
  [[ -r /proc/$pid/cmdline ]] || { printf 'PID %s is not a live Linux process\n' "$pid" >&2; exit 1; }
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q 'train.py' || {
    printf 'Refusing to signal PID %s: command is not train.py\n' "$pid" >&2
    exit 1
  }
}

wait_for_sibling_exit() {
  local target_pid=$1
  local deadline=$((SECONDS + DETECT_TIMEOUT))
  local file sibling
  while ((SECONDS < deadline)); do
    for file in "$CONTROL_DIR"/rank-*.pid; do
      [[ -e $file ]] || continue
      sibling=$(<"$file")
      [[ $sibling == "$target_pid" ]] && continue
      if ! kill -0 "$sibling" 2>/dev/null; then
        record_event detected "sibling_rank_exited_pid_$sibling"
        return 0
      fi
    done
    sleep 1
  done
  record_event detection_timeout "no_sibling_exit_within_${DETECT_TIMEOUT}s"
  return 1
}

if [[ $ACTION == network-drop ]]; then
  [[ -n $INTERFACE && -n $PEER ]] || { printf 'network-drop requires --interface and --peer\n' >&2; exit 2; }
  add=(sudo iptables -I OUTPUT 1 -o "$INTERFACE" -d "$PEER" -j DROP)
  remove=(sudo iptables -D OUTPUT -o "$INTERFACE" -d "$PEER" -j DROP)
  if ((REALLY == 0 || DRY_RUN)); then
    printf '%q ' "${add[@]}"; printf '\n'
    printf 'sleep %q\n' "$DURATION"
    printf '%q ' "${remove[@]}"; printf '\n'
    ((DRY_RUN)) && exit 0
    printf 'Refusing to change iptables without --really\n' >&2
    exit 2
  fi
  record_event injected "network_drop_${INTERFACE}_to_${PEER}"
  "${add[@]}"
  # shellcheck disable=SC2329  # Invoked indirectly by trap.
  cleanup() { "${remove[@]}" || true; }
  trap cleanup EXIT INT TERM
  sleep "$DURATION"
  "${remove[@]}"
  trap - EXIT INT TERM
  record_event restored "network_restored"
  exit 0
fi

if [[ $ACTION == nccl-timeout ]]; then
  if ((DRY_RUN)); then
    printf 'write NCCL timeout exports to %q\n' "$ENV_FILE"
  else
    mkdir -p "$(dirname "$ENV_FILE")"
    {
      printf 'export DDP_TIMEOUT_SECONDS=30\n'
      printf 'export TORCH_NCCL_ASYNC_ERROR_HANDLING=1\n'
      printf 'export TORCH_NCCL_ENABLE_MONITORING=1\n'
      printf 'export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=30\n'
      printf 'export NCCL_IB_TIMEOUT=14\n'
      printf 'export NCCL_DEBUG=INFO\n'
    } >"$ENV_FILE"
    record_event configured "source_${ENV_FILE}_before_torchrun"
  fi
  if ((REALLY == 0)); then
    printf 'Timeout environment prepared only; source it before torchrun, then rerun with --really to pause a rank.\n'
    exit 0
  fi
  # A timeout setting cannot be injected into an existing process. --really is
  # valid only when this job was launched with the generated environment.
  ACTION=pause
  ((DURATION > 30)) || DURATION=45
fi

if ((DRY_RUN)); then
  pid="<rank-$RANK-pid>"
else
  pid=$(rank_pid)
  validate_training_pid "$pid"
fi

if [[ $ACTION == kill ]]; then
  record_event injected "sigkill_pid_$pid"
  if ((DRY_RUN)); then printf 'kill -KILL %q\n' "$pid"; exit 0; fi
  kill -KILL "$pid"
  wait_for_sibling_exit "$pid"
else
  record_event injected "sigstop_pid_$pid"
  if ((DRY_RUN)); then
    printf 'kill -STOP %q\n' "$pid"
    printf 'sleep %q\n' "$DURATION"
    printf 'kill -CONT %q\n' "$pid"
    exit 0
  fi
  kill -STOP "$pid"
  sleep "$DURATION"
  kill -CONT "$pid"
  record_event restored "sigcont_pid_$pid"
fi
