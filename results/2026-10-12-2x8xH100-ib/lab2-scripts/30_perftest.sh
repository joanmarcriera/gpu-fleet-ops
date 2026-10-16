#!/usr/bin/env bash
# 30_perftest.sh - raw RDMA bandwidth and latency between node0 and node1, no GPUs, no NCCL.
#
# Usage (on node0; needs passwordless ssh from node0 to node1 and perftest on both):
#   ./30_perftest.sh rails      ib_write_bw on every rail pair (mlx5_i on node0 -> same name on
#                               node1), 1 MiB messages, 4 QPs, 5 s each. Finds a weak cable.
#   ./30_perftest.sh sweep      ib_write_bw over all message sizes, then ib_write_lat, on rail 1
#   ./30_perftest.sh gdr        ib_write_bw from GPU memory (--use_cuda), only if perftest
#                               was built with CUDA; otherwise recorded as skipped.
#                               GDR_DEV=mlx5_N picks the HCA that 06-topo.txt shows at PIX/PXB
#                               to GPU0 (default: the first HCA, which may not be the closest)
#   ./30_perftest.sh all        rails + sweep + gdr
#   PRINT_ONLY=1 ./30_perftest.sh all     show the commands without running them
#
# How it works: for each test it starts the server on node1 over ssh, waits 2 s, then runs the
# client here pointed at node1. perftest servers exit after one test, so each test starts a new one.
# Output: results/<hostname>/5N-perftest-*.txt. Bandwidth is printed in Gb/s (--report_gbits)
# so it compares directly with the link rate (NDR = 400 Gb/s per port).

set -u
source "$(dirname "$0")/00_env.sh"
DEVS=$(rdma_devs)
FIRST=$(echo "$DEVS" | head -1)
# -F: do not fail on the CPU-frequency warning. RoCE needs the GID index (-x).
COMMON="-F"
[ "$FABRIC" = roce ] && COMMON="$COMMON -x $GID_INDEX"

pair() {  # pair <outfile-name> <tool> <device> <args...>
    local name=$1 tool=$2 dev=$3; shift 3
    local f="$RESULTS/$name.txt"
    if [ "${PRINT_ONLY:-0}" = 1 ]; then
        echo "[print-only] node1: $tool -d $dev $COMMON $*"
        echo "[print-only] node0: $tool -d $dev $COMMON $* $NODE1 > $f"
        return 0
    fi
    # Server on node1, in the background, killed by timeout if the client never connects.
    # ssh -n: do not read this terminal; BatchMode: fail instead of prompting.
    # shellcheck disable=SC2029  # we want $tool etc. expanded here, on node0
    ssh -n -o BatchMode=yes "$NODE1" "timeout 60 $tool -d $dev $COMMON $*" \
        > "$RESULTS/$name.server.txt" 2>&1 &
    # The client retries: on a cold ssh the server may need more than 2 s to start listening.
    # A refused connection does not use up the server's single accept, so retrying is safe.
    local try rc=1
    for try in 1 2 3 4 5; do
        sleep 2
        $tool -d "$dev" $COMMON "$@" "$NODE1" > "$f.try" 2>&1; rc=$?
        grep -q -i "couldn't connect" "$f.try" || break
    done
    {
        echo "# host: $(hostname -s) -> $NODE1   date: $(date '+%Y-%m-%dT%H:%M:%S%z')   attempts: $try"
        echo "# cmd:  $tool -d $dev $COMMON $* $NODE1"
        echo
        cat "$f.try"
        echo "# exit: $rc"
    } > "$f"
    rm -f "$f.try"
    wait
    echo "saved $f"
}

rails() {
    local i=0
    for dev in $DEVS; do
        i=$((i + 1))
        # 1 MiB messages, 4 queue pairs, 5 seconds: one QP alone may not fill a 400G port.
        pair "51-perftest-rail$i-$dev-write-bw" ib_write_bw "$dev" -s 1048576 -q 4 -D 5 --report_gbits
    done
}

sweep() {
    pair "52-perftest-write-bw-sweep" ib_write_bw "$FIRST" -a -q 4 --report_gbits
    # Latency: small messages, single QP; report the typical (median) column, not the minimum.
    pair "53-perftest-write-lat" ib_write_lat "$FIRST" -a
}

gdr() {
    if ib_write_bw --help 2>&1 | grep -q use_cuda; then
        pair "54-perftest-write-bw-gpu0" ib_write_bw "${GDR_DEV:-$FIRST}" -s 1048576 -q 4 -D 5 --report_gbits --use_cuda=0
    else
        echo "SKIPPED: this perftest build has no --use_cuda (GPU memory) support" \
            > "$RESULTS/54-perftest-write-bw-gpu0.txt"
        echo "saved $RESULTS/54-perftest-write-bw-gpu0.txt (skipped)"
    fi
}

[ -n "$FIRST" ] || [ "${PRINT_ONLY:-0}" = 1 ] || { echo "no ACTIVE $FABRIC RDMA device found; run ibstat" >&2; exit 1; }
[ -n "$FIRST" ] || FIRST=mlx5_0 DEVS=mlx5_0   # print-only on a box with no HCA

ssh_check
case "${1:-}" in
    rails) rails ;;
    sweep) sweep ;;
    gdr)   gdr ;;
    all)   rails; sweep; gdr ;;
    *)     sed -n '2,19p' "$0"; exit 2 ;;
esac
