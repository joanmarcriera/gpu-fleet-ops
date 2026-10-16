#!/usr/bin/env python3
"""curate_lab2.py - build the scrubbed, curated lab2 results folder.

Usage:
    bin/curate_lab2.py SRC_DIR [DEST_DIR]

SRC_DIR   the "results-final" folder of the 12 Oct 2026 run (contains
          node0/<instance>/NN-*.txt and node0/lab2-scripts/).
DEST_DIR  default: results/2026-10-12-2x8xH100-ib (relative to the repo root).

What it does (read-only on SRC_DIR):
  * nccl-tests outputs: keeps the header, the result table rows, the summary
    lines and the exit marker; drops the multi-megabyte NCCL_DEBUG=INFO noise
    and appends a one-line summary of which transport NCCL reported.
    Table rows that were torn by interleaved debug output are dropped, never
    repaired (they are listed in SCRUB.md).
  * perftest, system and script files: copied whole, then scrubbed.
  * training logs: header lines + the RESULT json line only.
  * scrub: private IPs -> node0/node1, instance hostnames -> node0/node1,
    login name and home path -> "user", IB GUIDs -> 0x<guid>.
  * writes SCRUB.md, MANIFEST.md and SHA256SUMS.
Only the Python standard library is used.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

# --- scrub rules --------------------------------------------------------
# The real-to-placeholder mapping (hostnames, addresses, login) is NOT in this file.
# It is read from bin/scrub-map.local.json (git-ignored). Copy
# bin/scrub-map.example.json, fill in the real literals, and keep it private.
# Format: {"literals": [["real string", "replacement"], ...], "instance_prefix": "..."}
MAP_PATH = Path(__file__).resolve().parent / "scrub-map.local.json"


def load_rules(path: Path):
    if not path.exists():
        sys.exit(f"missing {path}: copy scrub-map.example.json to it and fill in the real values")
    m = json.loads(path.read_text())
    rules = [(re.compile(re.escape(a)), b) for a, b in m["literals"]]  # literals only, no regex
    # any other instance hostname with the same prefix
    rules.append((re.compile(re.escape(m["instance_prefix"]) + r"[a-z0-9]+"), "nodeX"))
    rules.append((re.compile(r"\b0x[0-9a-f]{16}\b"), "0x<guid>"))  # InfiniBand GUIDs
    return rules


SCRUB = load_rules(MAP_PATH)
# any other dotted quad is replaced, except loopback and tool version strings
IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


# money figures are not published: the price default in 00_env.sh is blanked
PRICE_DEFAULT = re.compile(r'(PRICE_USD_PER_HOUR:-)"[^"]*"')


def scrub(text: str) -> str:
    text = PRICE_DEFAULT.sub(r'\1"<removed>"', text)
    for rx, rep in SCRUB:
        text = rx.sub(rep, text)
    return IPV4.sub(
        lambda m: m.group(0) if m.group(0).startswith(("127.", "2.1.1.", "18.0.0."))
        else "<ip>", text)


# A complete nccl-tests result row: size count type redop root + 2x(time algbw busbw wrong)
# (the "#wrong" column is N/A for alltoall's in-place pass, and redop is "none" there)
ROW = re.compile(r"^\s*\d+\s+\d+\s+\w+\s+\w+\s+-?\d+(\s+[\d.]+){3}\s+(\d+|N/A)(\s+[\d.]+){3}\s+(\d+|N/A)\s*$")
VIA = re.compile(r"via (NET/[A-Za-z0-9_]+)(?:/\d+)?(/GDRDMA)?")


def curate_nccl(src: Path) -> tuple[str, int]:
    """Return (curated text, number of torn table rows dropped)."""
    keep, via, torn = [], Counter(), 0
    started = False
    for line in src.read_text(errors="replace").splitlines():
        m = VIA.search(line)
        if m:
            via[m.group(1) + (" +GDRDMA" if m.group(2) else "")] += 1
        if line.startswith("== NCCL_ALGO="):
            # block label written by 20_nccl.sh before each forced-algorithm run
            keep.append("# " + line[3:])
        elif line.startswith("#"):
            keep.append(line)
            started = started or "out-of-place" in line
        elif ROW.match(line):
            keep.append(line)
        elif started and re.match(r"^\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+\d+", line):
            torn += 1  # tail of a row split by interleaved debug output
    if via:
        keep.append("# [curated] NCCL_DEBUG channel lines by transport: "
                    + ", ".join(f"{k} x{v}" for k, v in sorted(via.items())))
    keep.append("# [curated] NCCL_DEBUG=INFO lines removed; "
                f"{torn} torn table row(s) dropped (see SCRUB.md)")
    return "\n".join(keep) + "\n", torn


def annotate_env(text: str) -> str:
    """Correct the 'bare metal' wording: the nodes were KVM VMs with SR-IOV VFs.
    Only the INSTANCE_TYPE label changes; an annotation comment explains why."""
    old = "2 x 8x H100 SXM, bare metal, InfiniBand)"
    assert old in text, "00_env.sh INSTANCE_TYPE default changed; update annotate_env"
    note = ("# [annotation, added when curating] Not bare metal: the nodes were KVM VMs and the\n"
            "# ConnectX HCAs are SR-IOV virtual functions (see system/08-lspci.txt).\n")
    text = text.replace(old, "2 x 8x H100 SXM, VMs with SR-IOV VFs, InfiniBand)")
    return text.replace("export INSTANCE_TYPE=", note + "export INSTANCE_TYPE=", 1)


def curate_train(src: Path) -> str:
    keep = [l for l in src.read_text(errors="replace").splitlines()
            if l.startswith("# ") or l.startswith("RESULT ")]
    return "\n".join(keep) + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).resolve()
    repo = Path(__file__).resolve().parent.parent
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else repo / "results/2026-10-12-2x8xH100-ib"
    run = next((src / "node0").glob(json.loads(MAP_PATH.read_text())["instance_prefix"] + "*"))
    scripts = src / "node0" / "lab2-scripts"
    dest.mkdir(parents=True, exist_ok=True)
    manifest, torn_report = [], []

    def put(rel: str, text: str, source: str, note: str = "") -> None:
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(scrub(text))
        manifest.append((rel, source, note))

    for f in sorted(run.glob("4[1-9]-nccl-*.txt")):
        text, torn = curate_nccl(f)
        put(f"nccl/{f.name}", text, f"node0/.../{f.name}", "table + headers only")
        if torn:
            torn_report.append(f"{f.name}: {torn}")
    put("nccl/40-nccl-tests-commit.txt", (run / "40-nccl-tests-commit.txt").read_text(),
        "node0/.../40-nccl-tests-commit.txt")
    for f in sorted(run.glob("5[1-4]-perftest-*.txt")):
        put(f"perftest/{f.name}", f.read_text(errors="replace"), f"node0/.../{f.name}", "whole file")
    for f in sorted(run.glob("6[123]-train-*.txt")):
        if f.name.endswith(".node0-copy"):
            continue
        put(f"training/{f.name}", curate_train(f), f"node0/.../{f.name}", "RESULT + header only")
    for f in sorted(run.glob("6[3]-gpu-util-*.csv")):
        put(f"training/{f.name}", f.read_text(), f"node0/.../{f.name}", "whole file")
    for n in ("01-os", "03-versions", "04-nvidia-smi", "06-topo", "08-lspci", "10-ibstat",
              "12-ibdev2netdev", "14-gdr", "19-counters-sysfs-after", "20-sminfo",
              "21-ibnetdiscover", "22-iblinkinfo", "23-ibqueryerrors", "24-ibdiagnet"):
        put(f"system/{n}.txt", (run / f"{n}.txt").read_text(errors="replace"),
            f"node0/.../{n}.txt", "whole file")
    for n in ("00_env.sh", "10_fabric_checks.sh", "20_nccl.sh", "30_perftest.sh",
              "60_train.sh", "60_train_ddp.py", "90_collect.sh"):
        text, note = (scripts / n).read_text(), "as run on node0"
        if n == "00_env.sh":
            text, note = annotate_env(text), "as run on node0; one annotation (see SCRUB.md)"
        put(f"lab2-scripts/{n}", text, f"node0/lab2-scripts/{n}", note)

    (dest / "SCRUB.md").write_text(
        "# Scrub log\n\nGenerated by `bin/curate_lab2.py`. Values were replaced, not removed.\n\n"
        "| Original | Replaced with |\n|---|---|\n"
        "| the two cloud-instance hostnames | `node0`, `node1` |\n"
        "| the two private VM addresses (both nodes' `eth0`) | `node0`, `node1` |\n"
        "| the address of a node1 that was deleted mid-run (first perftest pass) | `old-node1` |\n"
        "| any other dotted-quad address | `<ip>` (loopback and tool version strings kept) |\n"
        "| login name and home directory | `user`, `/home/user` |\n"
        "| 64-bit InfiniBand node/port/system GUIDs | `0x<guid>` |\n"
        "| the hourly price default in `00_env.sh` | `<removed>` (no money figures are published) |\n\n"
        "## Annotations (not data changes)\n\n"
        "* `lab2-scripts/00_env.sh`: the `INSTANCE_TYPE` default said \"bare metal\". The nodes were\n"
        "  VMs whose HCAs are SR-IOV virtual functions, so the label now says that and a two-line\n"
        "  comment explains it. Nothing the script measured or ran is affected.\n"
        "* `nccl/46-nccl-single-algo.txt`: the raw file marks each block with a line\n"
        "  `== NCCL_ALGO=<name>`, and the curator used to drop those markers. They are kept as\n"
        "  `# NCCL_ALGO=<name>`. Block order in the raw source is Ring, NVLS, Tree (recorded by the\n"
        "  markers, not inferred). The `cmd:` header still shows the `<algo>` placeholder.\n"
        "* `system/08-lspci.txt` is added to the published set (output already filtered to\n"
        "  Mellanox/NVIDIA lines) as the evidence that the HCAs are virtual functions.\n\n"
        "Not present in the source files and so not scrubbed: tokens, keys, account or project IDs.\n"
        "The public IPs of the VMs appear only in the source tree's own manifest, which is not\n"
        "published. IB LIDs, QPNs and PSNs are kept (session-scoped, not identifying).\n\n"
        "## Curation, not just scrubbing\n\n"
        "* nccl-tests files keep headers, result rows and summary lines. About 9 MB of\n"
        "  `NCCL_DEBUG=INFO` output was dropped; each file ends with a one-line count of the\n"
        "  transport NCCL reported.\n"
        "* A few table rows in the 16-rank files were torn by interleaved debug output in the\n"
        "  raw capture. They are dropped, not repaired. Per file: "
        + (", ".join(torn_report) or "none") + ".\n"
        "* Training logs keep header lines and the `RESULT` JSON line.\n"
        "* `63-train-2node-long-run3.txt`: the corrected copy is used (RESULT valid, hang was at\n"
        "  teardown). The uncorrected `.node0-copy` is not included.\n"
        "* Left out on purpose: `nvidia-smi -q` dump, lspci, ibv_devinfo, topology XML, the\n"
        "  side job run on the same nodes, and everything not about the fabric benchmark.\n")

    lines = ["# Manifest\n", "Source: `results-final/` of the 12 Oct 2026 run (node0 copy).\n",
             "| File | Source | Note |", "|---|---|---|"]
    lines += [f"| `{r}` | `{s}` | {n} |" for r, s, n in manifest]
    (dest / "MANIFEST.md").write_text("\n".join(lines) + "\n")

    sums = []
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS" and "plots" not in p.parts:
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}")
    (dest / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(f"wrote {len(manifest)} files to {dest}; torn rows: {torn_report or 'none'}")


if __name__ == "__main__":
    main()
