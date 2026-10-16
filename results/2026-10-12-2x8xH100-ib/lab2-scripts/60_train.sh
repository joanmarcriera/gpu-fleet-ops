#!/usr/bin/env bash
# 60_train.sh - run the ResNet-50 DDP benchmark in one of three shapes. Run on node0.
#
# Usage: NODES="node0 node1" ./60_train.sh 1gpu|1node|2node [runs]
#   1gpu   one GPU on node0                    (the baseline for scaling efficiency)
#   1node  8 GPUs on node0                     (NVLink only)
#   2node  16 GPUs, 8 per node, over InfiniBand (node1 is started over SSH)
# Output: results/<host>/61|62|63-train-<shape>-run<k>.txt, one RESULT json line per run, plus a
# GPU utilisation log (nvidia-smi, every 2 s) for the 2node shape.
# Env: STEPS=<n> timed steps (default 60), TAG=<suffix> for the output file names.
# Needs: ~/venv with torch+torchvision on both nodes (same path), SSH node0 -> node1.
set -euo pipefail
cd "$(dirname "$0")"; source ./00_env.sh >/dev/null
SHAPE=${1:?usage: $0 1gpu|1node|2node [runs]}; RUNS=${2:-2}
read -r N0 N1 <<<"$NODES"
PY=$HOME/venv/bin
export NCCL_NET_PLUGIN=none NCCL_DEBUG=${NCCL_DEBUG:-WARN} NCCL_IB_HCA=${NCCL_IB_HCA:-}
case $SHAPE in 1gpu) NP=1; ID=61;; 1node) NP=8; ID=62;; 2node) NP=8; ID=63;; *) echo "bad shape"; exit 2;; esac
for k in $(seq "$RUNS"); do
  OUT=$RESULTS/$ID-train-$SHAPE${TAG:-}-run$k.txt
  { echo "# host: $(hostname) date: $(date -Is) shape=$SHAPE run=$k"; } > "$OUT"
  if [ "$SHAPE" = 2node ]; then
    # node1 first, in the background; rendezvous is on node0:29500
    ssh "$N1" "export NCCL_NET_PLUGIN=none NCCL_DEBUG=WARN NCCL_IB_HCA='$NCCL_IB_HCA'; cd ~/lab2 && \
      $PY/torchrun --nnodes 2 --nproc-per-node $NP --node-rank 1 --master-addr $N0 --master-port 29500 \
      60_train_ddp.py ${STEPS:+--steps $STEPS} > /dev/null 2>&1" &
    ( nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv -l 2 \
        > "$RESULTS/$ID-gpu-util-$SHAPE${TAG:-}-run$k.csv" 2>&1 & echo $! > /tmp/smi.pid )
    $PY/torchrun --nnodes 2 --nproc-per-node $NP --node-rank 0 --master-addr "$N0" --master-port 29500 \
      60_train_ddp.py ${STEPS:+--steps $STEPS} >> "$OUT" 2>&1 || echo "# exit: $?" >> "$OUT"
    kill "$(cat /tmp/smi.pid)" 2>/dev/null || true; wait
  else
    $PY/torchrun --standalone --nnodes 1 --nproc-per-node $NP 60_train_ddp.py ${STEPS:+--steps $STEPS} >> "$OUT" 2>&1 \
      || echo "# exit: $?" >> "$OUT"
  fi
  grep -h RESULT "$OUT" || echo "run $k: no RESULT line, see $OUT"
done
