#!/usr/bin/env python3
"""plot_lab2.py - plot the 12 Oct 2026 2x8xH100 InfiniBand run from the raw outputs.

Usage:
    python3 plots/plot_lab2.py [RESULTS_DIR]

    RESULTS_DIR defaults to results/2026-10-12-2x8xH100-ib (relative to the repo root).
    Needs python3 + matplotlib (e.g. `uv venv .venv && uv pip install -p .venv matplotlib`,
    then `.venv/bin/python plots/plot_lab2.py`). The .venv is git-ignored.

Writes into RESULTS_DIR/plots/:
    busbw_vs_size.png   all_reduce bus bandwidth vs message size, log-x,
                        series: 1 node / 2 nodes / 2 nodes one rail / 2 nodes TCP
    perftest_rails.png  ib_write_bw per rail (8 HCAs) against the 400 Gb/s line rate
    degradation_ladder.png  two-node all_reduce at 256 MiB: 8 rails vs 1 rail vs InfiniBand off
    rail_topology.png   tenant-view diagram: GPU<->NIC PCIe affinity (nvidia-smi topo -m) and
                        per-rail ib_write_bw; the switch layer is NOT observable from the VM
    algorithms.png      forced NCCL_ALGO (Ring/NVLS/Tree) on 1 node, plus 2-node Ring and default
    collectives.png     16 GPUs, 4 GiB: all_reduce vs all_gather vs all-to-all busbw
    resnet_scaling.png  ResNet-50 DDP img/s at 1/8/16 GPUs and scaling efficiency

Nothing is hardcoded except spec/label text: every plotted number is parsed from the
nccl-tests / perftest / training text files (each figure function names its source files).
The one non-measured number is the 450 GB/s NVLink 4 per-direction line in algorithms.png,
which is the H100 spec (900 GB/s bidirectional) and is labelled as such.
Figures are 1600x900 px (8x4.5 in at 200 dpi), white background, one Okabe-Ito palette.
Series with several runs (1 node x3, 2 nodes x3) are plotted as the median per size,
with the min-max range shaded. Out-of-place busbw is used throughout, which is also the
column the headline figures quote. Rows torn by interleaved debug output in the raw
capture were dropped by bin/curate_lab2.py, so a size or two may be missing from a series.
A table of the parsed peaks is printed (and saved as peaks.txt) so the plots can be
checked against the headline numbers.
"""
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

# nccl-tests row: size count type redop root | time algbw busbw wrong | time algbw busbw wrong
ROW = re.compile(r"^\s*(\d+)\s+\d+\s+\w+\s+\w+\s+-?\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+\d+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+(?:\d+|N/A)\s*.*$")
# perftest result row: bytes iterations BWpeak BWaverage MsgRate
PT = re.compile(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+")

# Okabe-Ito colour-blind-safe palette
SERIES = [  # (label, files, colour, marker)
    ("1 node, 8 GPUs (NVLink/NVSwitch)", ["41-nccl-single-run1.txt", "41-nccl-single-run2.txt", "41-nccl-single-run3.txt"], "#0072B2", "o"),
    ("2 nodes, 16 GPUs, 8 rails", ["42-nccl-multi-run1.txt", "42-nccl-multi-run2.txt", "42-nccl-multi-run3.txt"], "#009E73", "s"),
    ("2 nodes, one rail only", ["43-nccl-inject-one-rail.txt"], "#E69F00", "^"),
    ("2 nodes, InfiniBand off (TCP)", ["44-nccl-inject-ib-disabled.txt"], "#D55E00", "v"),
]


FIG = (8, 4.5)   # inches; at DPI 200 -> 1600x900 px
DPI = 200
BLUE, GREEN, ORANGE, RED, SKY, PURPLE, GREY = "#0072B2", "#009E73", "#E69F00", "#D55E00", "#56B4E9", "#CC79A7", "#555555"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def titled(fig, title: str, subtitle: str, footer: str = "") -> None:
    """Title + subtitle (message size and units) at the top, optional source footer."""
    fig.text(0.03, 0.955, title, fontsize=17, fontweight="bold", va="top")
    fig.text(0.03, 0.885, subtitle, fontsize=10.5, color=GREY, va="top")
    if footer:
        fig.text(0.03, 0.015, footer, fontsize=7.5, color=GREY, va="bottom")


def save(fig, res: Path, name: str) -> None:
    out = res / "plots" / name
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print("wrote", out)


def median_at(files: list[Path], size: int) -> float:
    """Median over runs of out-of-place busbw at exactly `size`; fails loudly if a run lacks it."""
    vals = []
    for f in files:
        rows = read_rows(f)
        if size not in rows:
            raise SystemExit(f"{f.name}: no row for size {human(size)}")
        vals.append(rows[size])
    return statistics.median(vals)


def read_blocks(path: Path) -> dict[str, dict[int, float]]:
    """nccl-tests output with '# NCCL_ALGO=<name>' markers -> {algo: {size: busbw}}."""
    out: dict[str, dict[int, float]] = {}
    cur = None
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^# NCCL_ALGO=(\w+)", line)
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        m = ROW.match(line)
        if m and cur is not None and float(m.group(2)) > 0:
            cur[int(m.group(1))] = float(m.group(2))
    return out


def verify_no_wrong(res: Path) -> int:
    """Every nccl-tests file that reports validation must say 'Out of bounds values : 0 OK'
    and every parsed row must have #wrong 0 (or N/A for the in-place all-to-all column)."""
    n = 0
    for f in sorted((res / "nccl").glob("4*.txt")):
        t = f.read_text(errors="replace")
        if "Out of bounds values" not in t:
            continue  # e.g. 40-nccl-tests-commit.txt
        if "Out of bounds values : 0 OK" not in t:
            raise SystemExit(f"{f.name}: validation line is not '0 OK'")
        for line in t.splitlines():
            m = re.match(r"^\s*\d+\s+\d+\s+\w+\s+\w+\s+-?\d+\s+(?:[\d.]+\s+){3}(\S+)\s+(?:[\d.]+\s+){3}(\S+)\s*$", line)
            if m and (m.group(1) != "0" or m.group(2) not in ("0", "N/A")):
                raise SystemExit(f"{f.name}: nonzero #wrong in row: {line}")
        n += 1
    return n


def read_rows(path: Path) -> dict[int, float]:
    """size (bytes) -> out-of-place busbw (GB/s) for one nccl-tests output."""
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        m = ROW.match(line)
        if m:
            if float(m.group(2)) > 0:  # busbw 0.00 at tiny sizes cannot go on a log axis
                out[int(m.group(1))] = float(m.group(2))
    return out


def human(n: float) -> str:
    for unit, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return f"{n / div:g} {unit}"
    return f"{n:g} B"


def plot_busbw(res: Path, report: list[str]) -> None:
    """Sources: nccl/41-*, 42-*, 43-*, 44-* (median of runs per size, min-max shaded)."""
    fig, ax = plt.subplots(figsize=FIG, dpi=DPI)
    for idx, (label, files, colour, marker) in enumerate(SERIES):
        runs = [read_rows(res / "nccl" / f) for f in files]
        sizes = sorted(set().union(*runs))
        med = [statistics.median(r[s] for r in runs if s in r) for s in sizes]
        lo = [min(r[s] for r in runs if s in r) for s in sizes]
        hi = [max(r[s] for r in runs if s in r) for s in sizes]
        ax.plot(sizes, med, marker=marker, ms=4.5, lw=2, color=colour, label=f"{label}  (n={len(runs)})")
        if len(runs) > 1:
            ax.fill_between(sizes, lo, hi, color=colour, alpha=0.2, lw=0)
        # headline = out-of-place busbw at the largest size, per file
        for f, r in zip(files, runs):
            big = max(r)
            report.append(f"{f:<42} largest size {human(big):>8}  busbw {r[big]:8.2f} GB/s")
        big = max(sizes)
        ax.annotate(f"{med[-1]:g}", (big, med[-1]), textcoords="offset points",
                    xytext=(-4, 9 if idx == 0 else -17), ha="right", fontsize=10, color=colour, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(0.008, 2500)
    ticks = [2**k for k in range(10, 34, 4)]
    ax.set_xlim(2**9, 2**34)
    ax.set_xticks(ticks)
    ax.set_xticklabels([human(t) for t in ticks], fontsize=10)
    ax.set_xlabel("message size per rank")
    ax.set_ylabel("bus bandwidth, GB/s (log)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    titled(fig, "all_reduce bus bandwidth vs message size",
           "NCCL all_reduce, out-of-place busbw in GB/s, 1 KiB to 8 GiB per rank, 2 x 8 H100",
           "nccl-tests afd59ab, NCCL 2.26.2. One-rail and TCP runs used a narrower size range. busbw is not wire speed. "
           "Source: nccl/41-44.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    save(fig, res, "busbw_vs_size.png")


def plot_rails(res: Path, report: list[str]) -> list:
    """Source: perftest/51-perftest-rail{1..8}-mlx5_{0..7}-write-bw.txt (BW average column)."""
    rails = []
    for f in sorted((res / "perftest").glob("51-perftest-rail*-write-bw.txt")):
        name = re.search(r"rail(\d)-(mlx5_\d)", f.name)
        last = None
        for line in f.read_text().splitlines():
            m = PT.match(line)
            if m:
                last = float(m.group(4))  # BW average, Gb/s
        if name and last is not None:
            rails.append((int(name.group(1)), name.group(2), last))
    line_rate = 400.0
    fig, ax = plt.subplots(figsize=FIG, dpi=DPI)
    xs = [f"{dev.replace('mlx5_', 'mlx5_')}" for n, dev, _ in rails]
    vals = [v for _, _, v in rails]
    bars = ax.bar(xs, vals, color=BLUE, width=0.64)
    ax.axhline(line_rate, color=RED, ls="--", lw=1.6)
    ax.text(len(rails) - 0.5, line_rate + 6, "400 Gb/s line rate (NDR 4X)", ha="right", color=RED, fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{v:.1f}\n{100 * v / line_rate:.1f}%",
                ha="center", color="white", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 450)
    ax.set_ylabel("Gb/s")
    ax.set_xlabel("rail (HCA), node0 to node1, one rail at a time")
    ax.grid(True, axis="y", alpha=0.25)
    titled(fig, "Every rail delivers 98.8% of line rate",
           "ib_write_bw, 1 MiB messages, 4 queue pairs, Gb/s per rail",
           "Source: perftest/51-perftest-rail*-write-bw.txt (BW average). One run per rail.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    save(fig, res, "perftest_rails.png")
    for n, dev, v in rails:
        report.append(f"perftest rail{n} {dev:<7} BW average {v:7.2f} Gb/s = {100 * v / line_rate:.2f}% of {line_rate:g}")
    if rails:
        report.append(f"perftest rails min {min(vals):.2f} max {max(vals):.2f} Gb/s over {len(rails)} rails")
    return rails


def plot_ladder(res: Path, report: list[str]) -> None:
    """Sources: nccl/42-nccl-multi-run{1,2,3}.txt (8 rails, default; median of 3),
    nccl/43-nccl-inject-one-rail.txt (NCCL_IB_HCA=mlx5_0), nccl/44-nccl-inject-ib-disabled.txt
    (NCCL_IB_DISABLE, NET/Socket). All three files have a 256 MiB row, so one size is compared."""
    size = 256 * 2**20
    nccl = res / "nccl"
    healthy = median_at([nccl / f"42-nccl-multi-run{i}.txt" for i in (1, 2, 3)], size)
    one = median_at([nccl / "43-nccl-inject-one-rail.txt"], size)
    tcp = median_at([nccl / "44-nccl-inject-ib-disabled.txt"], size)
    rows = [("Eight rails\n(default)", healthy, GREEN), ("One rail\n(NCCL_IB_HCA=mlx5_0)", one, ORANGE),
            ("InfiniBand off\n(NET/Socket, TCP)", tcp, RED)]
    fig, ax = plt.subplots(figsize=FIG, dpi=DPI)
    ys = list(range(len(rows)))[::-1]
    for y, (label, v, c) in zip(ys, rows):
        ax.barh(y, v, color=c, height=0.62)
        slower = healthy / v
        txt = f"{v:.1f} GB/s" if v >= 10 else f"{v:.2f} GB/s"
        ax.text(v * 1.12, y + 0.12, txt, va="center", fontsize=13, fontweight="bold", color=c)
        ax.text(v * 1.12, y - 0.2, "healthy baseline" if v == healthy else
                f"{slower:.1f}x slower" if slower < 10 else f"{slower:.0f}x slower",
                va="center", fontsize=11, color=GREY)
        report.append(f"ladder 256 MiB {label.splitlines()[0]:<16} busbw {v:8.2f} GB/s  vs healthy {slower:6.2f}x")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.set_xlabel("all_reduce bus bandwidth, GB/s (log scale)")
    ax.grid(True, axis="x", which="both", alpha=0.25)
    n = verify_no_wrong(res)
    ax.text(0.99, 0.04, "#wrong = 0 in every run", transform=ax.transAxes, ha="right", fontsize=11,
            color=GREY, bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    titled(fig, "A degraded fabric does not fail, it gets slower",
           "Two-node all_reduce, 16 GPUs, 256 MiB per rank (same size for all three bars), busbw in GB/s",
           f"Source: nccl/42-* (median of 3 runs), nccl/43, nccl/44; out-of-place busbw. 1 run each for one-rail and TCP. "
           f"Validation checked in {n} nccl files.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    save(fig, res, "degradation_ladder.png")


def parse_topo(res: Path):
    """Parse system/06-topo.txt (nvidia-smi topo -m) -> (pix {nic: gpu}, numa {gpu: n}, nic_name {n: mlx5_x}).
    Fails loudly unless every GPU has exactly one PIX NIC, NICs map one-to-one, and GPU pairs are all NV18."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", (res / "system" / "06-topo.txt").read_text())
    pix, numa, nic_name = {}, {}, {}
    for line in text.splitlines():
        m = re.match(r"^NIC(\d+): (mlx5_\d+)", line.strip())
        if m:
            nic_name[int(m.group(1))] = m.group(2)
        if re.match(r"^GPU\d\t", line):
            f = line.split("\t")
            g = int(f[0][3:])
            pairs = f[1:9]
            if any(c.strip() not in ("X", "NV18") for c in pairs):
                raise SystemExit(f"topo: GPU{g} not all NV18 to peers: ambiguous, not drawing")
            nics = [i for i, c in enumerate(f[9:17]) if c.strip() == "PIX"]
            if len(nics) != 1:
                raise SystemExit(f"topo: GPU{g} has {len(nics)} PIX NICs: ambiguous, not drawing")
            pix[nics[0]] = g
            numa[g] = int(f[18])
    if sorted(pix) != list(range(8)) or sorted(pix.values()) != list(range(8)) or len(nic_name) != 8:
        raise SystemExit("topo: GPU<->NIC mapping is not one-to-one: ambiguous, not drawing")
    return pix, numa, nic_name


def plot_topology(res: Path, report: list[str], rails: list) -> None:
    """Sources: system/06-topo.txt (GPU<->NIC affinity, captured on node0; NIC n = mlx5_n per its
    legend), perftest/51-* (per-rail ib_write_bw, via `rails`). Switch layer is drawn dashed:
    ibnetdiscover was blocked (system/21-ibnetdiscover.txt), so it is not observed."""
    from matplotlib.patches import FancyBboxPatch
    pix, numa, nic_name = parse_topo(res)
    bw = {dev: v for _, dev, v in rails}
    fig = plt.figure(figsize=FIG, dpi=DPI)
    ax = fig.add_axes([0, 0.13, 1, 0.71])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    tint = {0: "#cfe6f5", 1: "#f1d9e6"}  # NUMA node 0 / 1

    def box(x0, x1, yc, text, fc, ec, fs=8.5, ls="-", bold=False, tc="black"):
        h = 0.46
        ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec=ec, lw=1.2, ls=ls))
        ax.text((x0 + x1) / 2, yc, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=tc)

    for x0, x1, name in ((0.2, 6.2, "node0"), (9.8, 15.8, "node1")):
        ax.add_patch(FancyBboxPatch((x0, 0.05), x1 - x0, 6.15, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc="#fafafa", ec="#888", lw=1.3))
        ax.text((x0 + x1) / 2, 5.95, f"{name} (VM): 8 x H100, 8 x ConnectX-7", ha="center", va="center",
                fontsize=8.5, fontweight="bold")
        ax.text((x0 + x1) / 2, 0.32, "GPUs fully meshed over NVLink/NVSwitch (NV18)" if name == "node0"
                else "same layout assumed (see footer)", ha="center", va="center", fontsize=8, color=GREY)
    ax.add_patch(FancyBboxPatch((7.15, 0.62), 1.7, 5.0, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="#f4f4f4", ec="#999", lw=1.2, ls="--"))
    ax.text(8, 6.7, "switch layer: not visible\nfrom the tenant VM\n(ibnetdiscover blocked;\nrail-optimised = assumed)",
            ha="center", va="center", fontsize=7.8, color=GREY, linespacing=1.15)
    for rail in range(8):
        y = 5.2 - rail * 0.6
        nic = nic_name[rail]
        g = pix[rail]
        for gx0, gx1, nx0, nx1, side in ((0.5, 2.4, 3.9, 5.4, 0), (13.6, 15.5, 10.6, 12.1, 1)):
            box(gx0, gx1, y, f"GPU{g} | NUMA{numa[g]}", tint[numa[g]], "#666", fs=8)
            box(nx0, nx1, y, nic, "#d5efe5", GREEN, fs=8.5, bold=True)
            a, b = sorted((gx1, nx0)) if side == 0 else sorted((nx1, gx0))
            ax.plot([a, b], [y, y], color="#666", lw=1.2)
            ax.text((a + b) / 2, y + 0.13, "PCIe PIX", ha="center", va="bottom", fontsize=6.5, color=GREY)
        box(7.35, 8.65, y, f"leaf {rail}?", "white", "#999", fs=7.5, ls="--", tc=GREY)
        ax.plot([5.4, 7.35], [y, y], color=BLUE, lw=2.4)
        ax.plot([8.65, 10.6], [y, y], color=BLUE, lw=2.4)
        for xm in (6.68, 9.32):
            ax.text(xm, y + 0.1, f"{bw[nic]:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=BLUE)
        report.append(f"topology rail {rail} {nic}: PIX with GPU{g} (NUMA{numa[g]}), ib_write_bw {bw[nic]:.2f} Gb/s")
    titled(fig, "Tenant view: eight rails, GPU-to-NIC PCIe affinity",
           "2 nodes x 8 H100 and 8 ConnectX-7 (NDR 400G). Blue: ib_write_bw per rail, Gb/s",
           "GPU<->NIC affinity: nvidia-smi topo -m on node0 (system/06-topo.txt). Each GPU has one PIX NIC; the other NICs are PHB or SYS.\n"
           "node1 drawn the same: topo was captured on node0 only (same PCI bus IDs in the nccl/42 device list).\n"
           "Rail speed: perftest/51-* (1 MiB, 4 QPs, rail i to rail i). Leaf switches are not observed; ibnetdiscover was blocked.")
    save(fig, res, "rail_topology.png")


def plot_algorithms(res: Path, report: list[str]) -> None:
    """Sources: nccl/46-nccl-single-algo.txt (blocks '# NCCL_ALGO=Ring|NVLS|Tree', 1 node, 8 GiB),
    nccl/45-nccl-multi-ring.txt (2 nodes, forced Ring, 8 GiB), nccl/42-nccl-multi-run{1,2,3}.txt
    (2 nodes, default, 8 GiB, median). 450 GB/s is the NVLink 4 spec (900 GB/s bidirectional), not measured."""
    size = 8 * 2**30
    nccl = res / "nccl"
    blocks = read_blocks(nccl / "46-nccl-single-algo.txt")
    report.append("algorithms block order in 46: " + ", ".join(blocks))
    two_ring = median_at([nccl / "45-nccl-multi-ring.txt"], size)
    m_runs = [nccl / f"42-nccl-multi-run{i}.txt" for i in (1, 2, 3)]
    two_def = median_at(m_runs, size)
    lo = min(read_rows(f)[size] for f in m_runs)
    hi = max(read_rows(f)[size] for f in m_runs)
    bars = [("NVLS", blocks["NVLS"][size], BLUE, ""), ("Ring", blocks["Ring"][size], BLUE, ""),
            ("Tree", blocks["Tree"][size], BLUE, ""), ("Ring", two_ring, GREEN, ""),
            ("default\n(auto)", two_def, GREEN, f"3 runs\n{lo:.0f} to {hi:.0f}")]
    fig, ax = plt.subplots(figsize=FIG, dpi=DPI)
    xs = [0, 1, 2, 3.4, 4.4]
    for x, (lab, v, c, extra) in zip(xs, bars):
        ax.bar(x, v, color=c, width=0.7)
        ax.text(x, v + 8, f"{v:.1f}", ha="center", fontsize=12, fontweight="bold", color=c)
        if extra:
            ax.text(x, v / 2, extra, ha="center", va="center", fontsize=8.5, color="white")
        report.append(f"algorithms {'1 node' if x < 3 else '2 nodes'} {lab.splitlines()[0]:<8} 8 GiB busbw {v:8.2f} GB/s")
    ax.axhline(450, color=RED, ls="--", lw=1.6)
    ax.text(2.0, 456, "NVLink 4 per-direction ceiling (ring)\n450 GB/s, spec value", ha="center", va="bottom", fontsize=9.5, color=RED)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=11)
    ax.set_ylim(0, 560)
    ax.set_xlim(-0.6, 5.0)
    ax.set_ylabel("all_reduce busbw, GB/s")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(1, -0.27, "1 node, 8 GPUs (forced NCCL_ALGO)", transform=ax.get_xaxis_transform(), ha="center", color=BLUE, fontsize=10)
    ax.text(3.9, -0.27, "2 nodes, 16 GPUs", transform=ax.get_xaxis_transform(), ha="center", color=GREEN, fontsize=10)
    titled(fig, "The algorithm matters as much as the fabric",
           "all_reduce busbw in GB/s at 8 GiB per rank; NVLS and Tree/Ring forced with NCCL_ALGO",
           "Source: nccl/46 (1 run per algorithm), nccl/45 (1 run), nccl/42-* (median of 3). The 450 GB/s line is the H100 NVLink 4 spec, not a measurement.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.84))
    save(fig, res, "algorithms.png")


def plot_collectives(res: Path, report: list[str]) -> None:
    """Sources, all 16 GPUs, 4 GiB, NCCL_NET_PLUGIN=none (NET/IB), out-of-place busbw:
    all_reduce = median of nccl/47-nccl-multi-net-plugin-none.txt, 48-*-run2, 48-*-run3;
    all_gather = nccl/49-nccl-multi-allgather.txt; all-to-all = nccl/49-nccl-multi-alltoall.txt."""
    size = 4 * 2**30
    nccl = res / "nccl"
    ar = median_at([nccl / "47-nccl-multi-net-plugin-none.txt", nccl / "48-nccl-multi-net-plugin-none-run2.txt",
                    nccl / "48-nccl-multi-net-plugin-none-run3.txt"], size)
    ag = median_at([nccl / "49-nccl-multi-allgather.txt"], size)
    a2a = median_at([nccl / "49-nccl-multi-alltoall.txt"], size)
    rows = [("all_reduce\n(median of 3)", ar, GREEN), ("all_gather", ag, BLUE), ("all-to-all", a2a, ORANGE)]
    fig, ax = plt.subplots(figsize=FIG, dpi=DPI)
    for i, (lab, v, c) in enumerate(rows):
        ax.bar(i, v, color=c, width=0.6)
        ax.text(i, v + 8, f"{v:.1f}", ha="center", fontsize=14, fontweight="bold", color=c)
        report.append(f"collectives 16 GPUs 4 GiB {lab.splitlines()[0]:<11} busbw {v:8.2f} GB/s")
    ax.set_xticks(range(3))
    ax.set_xticklabels([r[0] for r in rows], fontsize=12)
    ax.set_ylim(0, 540)
    ax.set_ylabel("busbw, GB/s")
    ax.grid(True, axis="y", alpha=0.25)
    titled(fig, "Same fabric, different collective, very different busbw",
           "2 nodes, 16 GPUs, 4 GiB message size (nccl-tests -e 4G), out-of-place busbw in GB/s",
           "Source: nccl/47, nccl/48-* (all_reduce); nccl/49-*allgather, nccl/49-*alltoall (1 run each). All with NCCL_NET_PLUGIN=none (NET/IB).\n"
           "busbw normalises each collective differently, so compare the shape, not the absolute ratio.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.84))
    save(fig, res, "collectives.png")


def plot_resnet(res: Path, report: list[str]) -> None:
    """Sources: training/61-train-1gpu-long-run{1,2}.txt, 62-train-1node-long-run{1,2}.txt,
    63-train-2node-long-run{1,2,3}.txt (RESULT json lines, 600 steps, images_per_sec; median).
    Efficiency = img/s / (n_gpus x img/s at 1 GPU)."""
    import json
    t = res / "training"

    def med(pattern: str) -> tuple[float, int, int]:
        v, ws = [], 0
        for f in sorted(t.glob(pattern)):
            for line in f.read_text().splitlines():
                if line.startswith("RESULT "):
                    j = json.loads(line[7:])
                    v.append(j["images_per_sec"])
                    ws = j["world_size"]
        if not v:
            raise SystemExit(f"no RESULT lines for {pattern}")
        return statistics.median(v), ws, len(v)

    data = [med("61-train-1gpu-long-run*.txt"), med("62-train-1node-long-run*.txt"), med("63-train-2node-long-run*.txt")]
    base = data[0][0]
    fig, ax = plt.subplots(figsize=FIG, dpi=DPI)
    for i, (v, ws, n) in enumerate(data):
        eff = 100 * v / (ws * base)
        ax.bar(i, v, color=GREEN, width=0.6)
        ax.hlines(ws * base, i - 0.38, i + 0.38, colors=GREY, linestyles="--", lw=1.4)
        ax.text(i, ws * base + 1500, f"{v:,.0f} img/s" + ("\n(baseline)" if ws == 1 else ""), ha="center",
                fontsize=13, fontweight="bold", color=GREEN, linespacing=1.1)
        if ws > 1:
            ax.text(i, v / 2, f"{eff:.1f}%", ha="center", va="center", fontsize=15, color="white", fontweight="bold")
        report.append(f"resnet {ws:>2} GPUs median {v:9.1f} img/s over {n} runs, scaling {eff:.2f}%")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["1 GPU", "8 GPUs\n(1 node)", "16 GPUs\n(2 nodes)"], fontsize=12)
    ax.set_ylim(0, 72000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylabel("images / second")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.99, 0.93, "dashed = ideal linear scaling", transform=ax.transAxes, ha="right", fontsize=9, color=GREY)
    titled(fig, "ResNet-50 DDP scales at about 95% to 16 GPUs",
           "Synthetic data, per-GPU batch 128, 600 steps, median images/s; percent = scaling efficiency vs 1 GPU",
           "Source: training/61-*, 62-*, 63-*long-run*.txt. Random tensors: a gentle fabric test, not a model run. n = 2 / 2 / 3, no confidence intervals.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.84))
    save(fig, res, "resnet_scaling.png")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "results/2026-10-12-2x8xH100-ib"
    (res / "plots").mkdir(exist_ok=True)
    report: list[str] = []
    plot_busbw(res, report)
    rails = plot_rails(res, report)
    plot_ladder(res, report)
    plot_topology(res, report, rails)
    plot_algorithms(res, report)
    plot_collectives(res, report)
    plot_resnet(res, report)
    text = "\n".join(report) + "\n"
    (res / "plots" / "peaks.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
