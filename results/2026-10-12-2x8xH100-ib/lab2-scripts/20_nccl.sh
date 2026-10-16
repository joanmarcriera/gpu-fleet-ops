#!/usr/bin/env bash
# 20_nccl.sh - build nccl-tests and run all_reduce_perf on one node, then across both nodes.
#
# Usage (on node0, inside tmux):
#   ./20_nccl.sh build      clone and build nccl-tests with MPI (do this on BOTH nodes,
#                           same path, because mpirun starts the binary on node1 too)
#   ./20_nccl.sh single     3 runs, one process driving all GPUs of this node (NVLink only)
#   ./20_nccl.sh multi      3 runs, 2 nodes x GPUS_PER_NODE ranks over the fabric (mpirun)
#   ./20_nccl.sh ring       1 two-node run with NCCL_ALGO=Ring: the only busbw you may quote
#                           as "% of line rate" (tree/NVLS can legitimately exceed the link)
#   ./20_nccl.sh inject     2 short runs that break things on purpose:
#                           a) one rail only (NCCL_IB_HCA = first HCA)  b) IB off (TCP sockets)
#   ./20_nccl.sh all        single + multi + ring + inject
#   PRINT_ONLY=1 ./20_nccl.sh multi    show the exact mpirun line without running it
#
# Needs: CUDA toolkit (nvcc), NCCL (libnccl-dev), Open MPI (openmpi-bin libopenmpi-dev), git, make.
# Output: results/<hostname>/4N-nccl-*.txt. Read "Avg bus bandwidth" and the 8 GB row.
# Dry run on the RTX 4060: MAXBYTES=256M ./20_nccl.sh build && MAXBYTES=256M ./20_nccl.sh single
# (busbw is 0 with one GPU: the 2(n-1)/n factor is 0 when n=1. That is correct.)

set -u
source "$(dirname "$0")/00_env.sh"
BIN=$NCCL_TESTS_DIR/build/all_reduce_perf
# -b/-e/-f: sizes from 8 bytes to MAXBYTES, doubling. -w/-n: warm-up and timed iterations.
# -c 1: check results; the "#wrong" column must be 0.
SWEEP="-b 8 -e $MAXBYTES -f 2 -w 5 -n 20 -c 1"

# Open MPI default install paths on Ubuntu; override MPI_HOME if HPC-X or another MPI is used.
MPI_HOME=${MPI_HOME:-/usr/lib/x86_64-linux-gnu/openmpi}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

go() {  # go <outfile-name> <command...>: run or print, saving output through cap
    # PRINT_ENV: the env prefix of this run (e.g. NCCL_ALGO=Ring), shown in dry runs.
    local name=$1; shift
    if [ "${PRINT_ONLY:-0}" = 1 ]; then echo "[print-only] ${PRINT_ENV:-}${PRINT_ENV:+ }$*"; return 0; fi
    cap "$name" "$@"
}

build() {
    # NCCL_TESTS_REF pins the exact nccl-tests commit (set it in 00_env.sh to the hash you dry-ran
    # on Friday), so Monday builds the code you tested and not whatever main is that morning.
    : "${NCCL_TESTS_REF:?set NCCL_TESTS_REF to a nccl-tests commit hash or tag before building}"
    if [ ! -d "$NCCL_TESTS_DIR" ]; then
        git clone https://github.com/NVIDIA/nccl-tests.git "$NCCL_TESTS_DIR" || exit 1
    fi
    git -C "$NCCL_TESTS_DIR" checkout -q "$NCCL_TESTS_REF" || exit 1
    # MPI=1 builds the multi-process version; NCCL_HOME=/usr finds libnccl from the apt package.
    make -C "$NCCL_TESTS_DIR" -j"$(nproc)" MPI=1 MPI_HOME="$MPI_HOME" \
         CUDA_HOME="$CUDA_HOME" NCCL_HOME="${NCCL_HOME:-/usr}" || exit 1
    git -C "$NCCL_TESTS_DIR" rev-parse HEAD > "$RESULTS/40-nccl-tests-commit.txt"
    ls -l "$BIN"
}

single() {
    # One process, -g N GPUs: traffic stays inside the node (NVLink/NVSwitch). This is the
    # ceiling the two-node run is compared with.
    for run in 1 2 3; do
        go "41-nccl-single-run$run" "$BIN" $SWEEP -g "$GPUS_PER_NODE"
    done
}

mpi_line() {  # prints the mpirun prefix for 2 nodes x GPUS_PER_NODE ranks, one GPU per rank
    # timeout 900: a hung run (bad HCA, stuck rank) must not keep the meter running.
    # -H node0:8,node1:8 : 8 ranks per node. --map-by ppr:8:node: 8 ranks on each node.
    # --bind-to none: Open MPI does not know where the GPUs are; NCCL pins its own threads
    #   next to each GPU and NIC, so MPI must not pin ranks to the wrong socket first.
    # btl/oob_tcp_if_include: MPI talks over the private Ethernet only; NCCL moves the data.
    # -x: export each variable to the ranks on node1 too (mpirun does not copy env by default).
    #   Only variables that are set: an unset one may make mpirun stop.
    local v args="timeout 900 mpirun"
    [ "$(id -u)" = 0 ] && args="$args --allow-run-as-root"
    args="$args -np $((2 * GPUS_PER_NODE)) -H $NODE0:$GPUS_PER_NODE,$NODE1:$GPUS_PER_NODE"
    args="$args --map-by ppr:$GPUS_PER_NODE:node --bind-to none"
    [ -n "$OOB_IF" ] && args="$args --mca btl_tcp_if_include $OOB_IF --mca oob_tcp_if_include $OOB_IF"
    for v in PATH LD_LIBRARY_PATH NCCL_DEBUG NCCL_DEBUG_SUBSYS NCCL_IB_HCA NCCL_SOCKET_IFNAME \
             NCCL_NET_GDR_LEVEL NCCL_TOPO_DUMP_FILE NCCL_IB_GID_INDEX NCCL_ALGO NCCL_IB_DISABLE; do
        [ -n "${!v:-}" ] && args="$args -x $v"
    done
    echo "$args"
}

multi() {
    ssh_check
    for run in 1 2 3; do
        # shellcheck disable=SC2046  # word splitting of mpi_line is intended
        go "42-nccl-multi-run$run" $(mpi_line) "$BIN" $SWEEP -g 1
    done
}

ring() {
    # NCCL picks ring, tree or NVLS per message size. busbw is defined with the ring formula,
    # so only a forced ring run compares honestly with 8 x link rate.
    ssh_check
    export NCCL_ALGO=Ring PRINT_ENV="NCCL_ALGO=Ring"
    # shellcheck disable=SC2046
    go "45-nccl-multi-ring" $(mpi_line) "$BIN" $SWEEP -g 1
    unset NCCL_ALGO PRINT_ENV
}

inject() {
    # a) One rail: NCCL may use only the first HCA. Expect busbw to fall to roughly one NIC
    #    (about 1/8 of the healthy number on an 8-rail node). This is "a cable is down" in miniature.
    ssh_check
    local all=${NCCL_IB_HCA:-} first
    first=${all%%,*}
    [ -n "$first" ] || [ "${PRINT_ONLY:-0}" = 1 ] || { echo "NCCL_IB_HCA is empty: no active HCA" >&2; exit 1; }
    export NCCL_IB_HCA=$first PRINT_ENV="NCCL_IB_HCA=$first"
    # shellcheck disable=SC2046
    go "43-nccl-inject-one-rail" $(mpi_line) "$BIN" -b 1M -e 1G -f 4 -w 5 -n 20 -c 1 -g 1
    export NCCL_IB_HCA=$all
    # b) IB disabled: NCCL falls back to TCP sockets on the Ethernet. Expect a collapse to
    #    Ethernet speed and "NET/Socket" in the log instead of "NET/IB".
    #    Sizes stop at 256M: over a slow front-end network 1 GB messages take minutes.
    export NCCL_IB_DISABLE=1 PRINT_ENV="NCCL_IB_DISABLE=1"
    # shellcheck disable=SC2046
    go "44-nccl-inject-ib-disabled" $(mpi_line) "$BIN" -b 1M -e 256M -f 4 -w 2 -n 10 -c 1 -g 1
    unset NCCL_IB_DISABLE PRINT_ENV
}

case "${1:-}" in
    build)  build ;;
    single) single ;;
    multi)  multi ;;
    ring)   ring ;;
    inject) inject ;;
    all)    single; multi; ring; inject ;;
    *)      sed -n '2,20p' "$0"; exit 2 ;;
esac

# Quick summary of what just ran: the average busbw line and the largest-message row of each file.
grep -H "Avg bus bandwidth" "$RESULTS"/4[1-5]-nccl-*.txt 2>/dev/null
