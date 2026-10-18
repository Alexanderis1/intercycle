"""Body-shop event simulation: run the four robot strategies over dev seeds and write the results table.

    python -m intercycle.eval.bodyshop                      # 20 dev seeds (0-19) x 6 h -> results/bodyshop/
    python -m intercycle.eval.bodyshop --quick              # 2 seeds x 1 h smoke run (< 2 min) -> out/bodyshop_quick/
    python -m intercycle.eval.bodyshop --seeds 0-9 --hours 6 --workers 4 --out results/bodyshop

Strategies (intercycle/sim/run.py):
    A0  shutdown-only: enters a machine zone only during planned line stops
    A   reactive: enters as soon as its own camera sees no motion, no prediction
    B   InterCycle: plans with the learned observer's broadcast (models learned on a separate training day)
    C   PLC oracle: plans with the scenario's true cycle models (upper bound)

Writes <out>/summary.json (aggregate means, 95 % CIs, per-run rows, seeds, hours, commit) and <out>/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from intercycle.eval.run_seeds import evaluate
from intercycle.sim.run import STRATEGIES

ROOT = Path(__file__).resolve().parents[2]
T0 = 7 * 3600                                   # shift window starts 07:00 (same as run_seeds)
NAMES = {"A0": "A0 shutdown-only", "A": "A reactive", "B": "B InterCycle (learned observer)", "C": "C PLC oracle"}
MAX_WORKERS = 6


def parse_seeds(text: str) -> list[int]:
    """'0-19' -> [0..19]; '0,3,5' -> [0, 3, 5]; '7' -> [7]."""
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            seeds.extend(range(a, b + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds in {text!r}")
    return seeds


def git_commit() -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    sha = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "intercycle", "scenario")
    return {"sha": sha or None, "code_dirty": bool(dirty)}


def table_md(ag: dict) -> str:
    c_insp = ag["C"]["machine_insp_per_prod_h"]["mean"]
    lines = ["| strategy | insp/h | trips/run | holds/run | line time lost s/run | insp/h vs C |",
             "|---|---|---|---|---|---|"]
    for s in STRATEGIES:
        m = lambda k, d: f"{ag[s][k]['mean']:.{d}f} ± {ag[s][k]['ci95']:.{d}f}"
        ratio = ag[s]["machine_insp_per_prod_h"]["mean"] / c_insp if c_insp else float("nan")
        lines.append(f"| {NAMES[s]} | {m('machine_insp_per_prod_h', 1)} | {m('trips', 2)} | {m('holds', 2)} "
                     f"| {m('line_time_lost_s', 1)} | {ratio:.3f} |")
    return "\n".join(lines)


def results_md(summary: dict) -> str:
    ag = summary["aggregate"]
    seeds, commit = summary["seeds"], summary["commit"]
    gain = ag["gain_retained_line_time"]
    return "\n".join([
        "# Body-shop event simulation results",
        "",
        f"Seeds {seeds[0]}-{seeds[-1]} (n = {len(seeds)} per strategy), {summary['hours']:g} h shift window from 07:00, "
        f"code commit `{commit['sha']}`{' (uncommitted changes in intercycle/ or scenario/)' if commit['code_dirty'] else ''}.",
        f"Command: `{summary['command']}` ({summary['wall_s']:.0f} s wall, {summary['workers']} processes).",
        "",
        "Means per run with 95 % confidence half-widths (1.96 x standard error over seeds).",
        "",
        table_md(ag),
        "",
        f"- B/C throughput (InterCycle inspections per production hour / PLC oracle): **{ag['throughput_vs_oracle']:.3f}**",
        f"- Line-time gain retained by B, (A - B) / (A - C): {'n/a' if gain is None else f'{gain:.3f}'}",
        "- insp/h = machine inspections per production hour; trips = robot crossed into an active zone "
        "(protective stop of the line); holds = inspections where the line waited for the robot; line time lost = seconds of "
        "production stopped by the robot.",
        "- Simulation only (event-based model of a body-shop cell), not measured on a real line.",
        "",
    ])


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0-19", help="seed list, e.g. 0-19 or 0,1,5 (default 0-19)")
    ap.add_argument("--hours", type=float, default=6.0, help="shift window length from 07:00 (default 6)")
    ap.add_argument("--workers", type=int, default=min(MAX_WORKERS, os.cpu_count() or 1),
                    help=f"parallel processes (default and maximum {MAX_WORKERS})")
    ap.add_argument("--out", default=None, help="output folder (default results/bodyshop, or out/bodyshop_quick with --quick)")
    ap.add_argument("--quick", action="store_true", help="smoke run: seeds 0-1 x 1 h")
    args = ap.parse_args(argv)
    if args.quick:
        args.seeds, args.hours = "0-1", 1.0
    seeds = parse_seeds(args.seeds)
    workers = max(1, min(args.workers, MAX_WORKERS))
    out = Path(args.out) if args.out else ROOT / ("out/bodyshop_quick" if args.quick else "results/bodyshop")
    out.mkdir(parents=True, exist_ok=True)

    t = time.time()
    e = evaluate(seeds, t0=T0, t1=T0 + args.hours * 3600, workers=workers)
    wall = time.time() - t
    cmd = "python -m intercycle.eval.bodyshop " + (
        "--quick" if args.quick else f"--seeds {args.seeds} --hours {args.hours:g}")
    summary = {"seeds": seeds, "hours": args.hours, "window_s": e["window"], "workers": workers,
               "wall_s": round(wall, 1), "command": cmd, "commit": git_commit(),
               "strategies": NAMES, "aggregate": e["aggregate"], "rows": e["rows"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out / "RESULTS.md").write_text(results_md(summary), encoding="utf-8")
    print(table_md(e["aggregate"]))
    print(f"B/C throughput {e['aggregate']['throughput_vs_oracle']:.3f}; wall {wall:.0f} s; wrote {out}")
    return summary


if __name__ == "__main__":
    main()
