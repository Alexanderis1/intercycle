"""Generate the pre-registered DEV worlds (PREREG v1 §2, prereg/PREREG_TWIN.md): dev seeds only, every other seed refused.

    python scripts/gen_dev_worlds.py --set robot --out-dir out/dev_worlds --suffix .day_script.json [--days 1-10]
    python scripts/gen_dev_worlds.py --set calib --out-dir out/dev_worlds_calib --suffix .day_script.json
    python scripts/gen_dev_worlds.py --seeds 0,2 --days 2-8 --out-dir ... --suffix ...      (a subset of a set)

Pre-registered dev world seeds (PREREG v1 §2; nothing else is accepted, whatever twin.rng's 0..999 dev range allows):
* `robot`: worlds 0-3 (PLAN_TWIN §5 "4 dev worlds"): robot development and tuning use days 2-8 (+ the observer's
  days 1-8 on the CPU engine); days 9-10 of dev worlds exist for the dry run and the slice's visit-count check;
* `calib`: worlds 100-131 (lead decision, Sun 18 Oct): S7b's M-W7 calibration only (CPU engine, observer, no robot).

The script never opens or imports truth itself (it is not in twin/truth_guard.py's lists): for every (seed, day) it
launches the truth producer `python -m twin.daygen --world-seed S --day D --out <out-dir>/w<S>_d<D><suffix>` in a
child process, which writes through `open_truth` and logs the write to results/truth_access.log. The file-name suffix
is passed by the caller (daygen requires the name to contain the day-script token, which this file may not name;
see tests/test_truth_allowlist.py). It refuses to run with INTERCYCLE_SEALED_RUN set, refuses an existing output file,
and writes `<out-dir>/dev_worlds.json` (seeds, days, the generator and dynamics sha256, the commands).
PREREG v1 must be committed before the first run (PLAN_TWIN §4 "PREREG v1 ... Dev worlds 0-3 are generated after it").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS: dict[str, tuple[int, ...]] = {
    "robot": tuple(range(0, 4)),          # PREREG v1 §2: dev worlds 0-3
    "calib": tuple(range(100, 132)),      # PREREG v1 §2: S7b calibration-only dev worlds 100-131
}
ALLOWED = frozenset(s for v in SETS.values() for s in v)
DAYS = tuple(range(1, 11))
HASHED = ("twin/daygen.py", "twin/rng.py", "twin/dynfit.py", "twin/dynamics.json")
CONTRACT = "intercycle.dev_worlds/1"


class Refusal(RuntimeError):
    pass


def _range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def check_seeds(seeds: list[int], set_name: str | None) -> list[int]:
    """Every seed must be a pre-registered dev seed (of `set_name` when given); duplicates refused."""
    if not seeds:
        raise Refusal("no seed given")
    if len(set(seeds)) != len(seeds):
        raise Refusal(f"repeated seeds: {seeds}")
    allowed = frozenset(SETS[set_name]) if set_name else ALLOWED
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise Refusal(f"seeds {bad} are not pre-registered dev seeds of {set_name or 'any set'} "
                      f"(robot 0-3, calib 100-131; PREREG v1 §2)")
    return sorted(seeds)


def check_days(days: list[int]) -> list[int]:
    if not days or len(set(days)) != len(days) or any(d not in DAYS for d in days):
        raise Refusal(f"days must be distinct values in 1..10, got {days}")
    return sorted(days)


def sha_lf(p: Path) -> str:
    """sha256 of the file with CRLF normalised to LF (the repo's checkout may convert line endings)."""
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def plan(seeds: list[int], days: list[int], out_dir: Path, suffix: str) -> list[dict]:
    if not suffix.endswith(".json") or "/" in suffix or "\\" in suffix:
        raise Refusal(f"suffix must be a bare file-name ending in .json, got {suffix!r}")
    jobs = []
    for s in seeds:
        for d in days:
            out = out_dir / f"w{s}_d{d}{suffix}"
            if out.exists():
                raise Refusal(f"{out} exists; dev worlds are generated once into a fresh path")
            jobs.append({"world_seed": s, "day": d, "out": str(out),
                         "cmd": [sys.executable, "-m", "twin.daygen", "--world-seed", str(s), "--day", str(d),
                                 "--out", str(out)]})
    return jobs


def run(jobs: list[dict], runner=subprocess.run) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for j in jobs:
        r = runner(j["cmd"], cwd=str(ROOT), env=env)
        if getattr(r, "returncode", 1) != 0:
            raise Refusal(f"generator failed for world {j['world_seed']} day {j['day']} (exit {r.returncode})")


def main(argv: list[str] | None = None, runner=subprocess.run) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), help="a pre-registered dev world set")
    ap.add_argument("--seeds", help="a subset of the set's seeds, e.g. 0,2 or 100-107")
    ap.add_argument("--days", default="1-10")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suffix", required=True, help="file-name suffix of each generated day (see the module doc)")
    ap.add_argument("--dry", action="store_true", help="print the jobs, generate nothing")
    a = ap.parse_args(argv)
    try:
        if os.environ.get("INTERCYCLE_SEALED_RUN"):
            raise Refusal("INTERCYCLE_SEALED_RUN is set: this script generates dev worlds only")
        if a.seeds:
            seed_list = _range(a.seeds)
        elif a.set is not None:
            seed_list = list(SETS[a.set])
        else:
            raise Refusal("give --set or --seeds")
        seeds = check_seeds(seed_list, a.set)
        days = check_days(_range(a.days))
        out_dir = Path(a.out_dir)
        jobs = plan(seeds, days, out_dir, a.suffix)
        manifest = out_dir / "dev_worlds.json"
        if manifest.exists():
            raise Refusal(f"{manifest} exists; dev worlds are generated once into a fresh directory")
    except Refusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if a.dry:
        for j in jobs:
            print(" ".join(j["cmd"]))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run(jobs, runner)
    except Refusal as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    body = {"contract": CONTRACT, "set": a.set, "seeds": seeds, "days": days,
            "inputs_sha256_lf": {p: sha_lf(ROOT / p) for p in HASHED},
            "jobs": [{k: j[k] for k in ("world_seed", "day", "out")} for j in jobs]}
    manifest.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(jobs)} days written; manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
