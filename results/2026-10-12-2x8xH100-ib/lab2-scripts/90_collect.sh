#!/usr/bin/env bash
# 90_collect.sh - gather both nodes' results, write the manifest and summary, make one tarball.
#
# Usage (on node0, after the last benchmark and BEFORE teardown):
#   ./90_collect.sh
# Then, from the Mac:
#   scp ubuntu@<node0-public-ip>:~/lab2/lab2-results-*.tgz \
#       ~/Development/Project/AI-learning/04-labs/lab2-infiniband-nccl-benchmark/
#   tar tzf lab2-results-*.tgz | head     # prove the archive opens before you delete anything
#
# What it writes into results/:
#   node1/              node1's results, copied over ssh (node0's are in results/<node0-hostname>/)
#   <host>/nccl-topology.xml  the topology NCCL detected (copied from /tmp)
#   manifest.md         provider, instance, price, meter start, versions, file checksums
#   summary.txt         the numbers you need for the note, pulled from the raw files
# The raw files are never edited; the note's numbers must trace back to one of them.

set -u
source "$(dirname "$0")/00_env.sh"
TOP=$(dirname "$RESULTS")          # results/
STAMP=$(date '+%Y%m%d-%H%M')

# 1. node1's files (its own 10_fabric_checks.sh run; the perftest server logs are on node0 already)
# A fixed folder name, so two nodes with the same hostname cannot overwrite each other.
ssh_check
mkdir -p "$TOP/node1"
scp -q -r -o BatchMode=yes "$NODE1:$LAB_DIR/results/*" "$TOP/node1/" \
    && echo "copied node1 results" || echo "WARN: could not copy node1 results (read the error above)"
# The NCCL topology dump already sits in each host's results folder (00_env.sh), so it travels
# with the scp above. Drop any symlink that came across: only regular files go into the archive.
find "$TOP" -type l -print -delete

# 2. manifest
{
    echo "# Lab 2 manifest"
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Provider | $PROVIDER |"
    echo "| Region | $REGION |"
    echo "| Instance type | $INSTANCE_TYPE |"
    echo "| Price, whole cluster | $PRICE_USD_PER_HOUR USD/h |"
    echo "| Meter start | $METER_START |"
    echo "| Collected at | $(date '+%Y-%m-%dT%H:%M:%S%z') |"
    echo "| Nodes | $NODES |"
    echo "| GPUs per node | $GPUS_PER_NODE |"
    echo "| GPU | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sort -u | paste -sd' ' -) |"
    echo "| NCCL_IB_HCA | ${NCCL_IB_HCA:-} |"
    echo
    echo "## Versions (node0)"
    echo
    echo '```'
    sed -n '4,$p' "$RESULTS/03-versions.txt" 2>/dev/null
    grep -h -m1 "NCCL version" "$RESULTS"/4[12]-nccl-*.txt 2>/dev/null | head -1
    echo '```'
    echo
    echo "## Teardown (fill by hand after terminating)"
    echo
    echo "- Meter stop (time of Terminate): "
    echo "- Hours billed x rate = "
    echo "- Balance or invoice delta = "
    echo "- Console shows zero instances, volumes, IPs: yes / no (screenshot: results/teardown.png)"
    echo "- Lab SSH key removed from the provider: yes / no"
    echo
    echo "## SHA-256 of every raw file"
    echo
    echo '```'
    (cd "$TOP" && find . -type f ! -name 'manifest*.md' ! -name summary.txt ! -name '*.tgz' -print0 | sort -z | xargs -0 sha256sum)
    echo '```'
} > "$TOP/manifest.new.md"
# Never overwrite a manifest whose teardown part you may already have filled in by hand.
if [ -e "$TOP/manifest.md" ]; then
    echo "kept $TOP/manifest.md; new version in manifest.new.md (merge by hand)"
else
    mv "$TOP/manifest.new.md" "$TOP/manifest.md"; echo "saved $TOP/manifest.md"
fi

# 3. summary: the lines the note needs
{
    echo "== NCCL all_reduce: average busbw per file (GB/s) =="
    grep -H "Avg bus bandwidth" "$RESULTS"/4[1-5]-nccl-*.txt 2>/dev/null
    echo
    echo "== NCCL largest message row per file (size count type redop root time algbw busbw #wrong ...) =="
    for f in "$RESULTS"/4[1-5]-nccl-*.txt; do
        [ -e "$f" ] || continue
        printf '%s: ' "$(basename "$f")"
        grep -E '^ +[0-9]+ +[0-9]+ +float' "$f" | tail -1
    done
    echo
    echo "== Transport NCCL chose (look for NET/IB and GDRDMA, or NET/Socket) =="
    grep -h -o -E "via NET/[A-Za-z]+(/[0-9]+)?(/GDRDMA)?" "$RESULTS"/4[1-5]-nccl-*.txt 2>/dev/null | sort | uniq -c
    echo
    echo "== perftest: last result row per file (bytes iterations BW-peak BW-average MsgRate) =="
    for f in "$RESULTS"/5[1-4]-perftest-*.txt; do
        [ -e "$f" ] || continue
        case $f in *.server.txt) continue ;; esac   # server-side logs repeat the client's numbers
        printf '%s: ' "$(basename "$f")"
        grep -E '^ +[0-9]+ +[0-9]+' "$f" | tail -1
    done
    echo
    echo "== ibdiagnet summary =="
    if grep -q -i summary "$RESULTS/24-ibdiagnet.txt" 2>/dev/null; then
        sed -n '/[Ss]ummary/,$p' "$RESULTS/24-ibdiagnet.txt" | head -30
    else
        echo "no ibdiagnet summary (not run, not installed, or not permitted: read 24-ibdiagnet.txt)"
    fi
} > "$TOP/summary.txt"
echo "saved $TOP/summary.txt"

# 4. one tarball, next to the scripts
(cd "$(dirname "$TOP")" && find "$(basename "$TOP")" -type f -print0) \
    | tar czf "$LAB_DIR/lab2-results-$STAMP.tgz" -C "$(dirname "$TOP")" --null -T -
ls -lh "$LAB_DIR/lab2-results-$STAMP.tgz"
