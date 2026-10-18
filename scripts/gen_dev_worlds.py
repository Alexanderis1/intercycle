"""Generate the pre-registered DEV worlds (PREREG v1 §2, prereg/PREREG_TWIN.md): dev seeds only, every other seed refused.

    python scripts/gen_dev_worlds.py --set robot --out-dir out/dev_worlds --suffix .day_script.json [--days 1-10]
    python scripts/gen_dev_worlds.py --set calib --out-dir out/dev_worlds_calib --suffix .day_script.json
    python scripts/gen_dev_worlds.py --set robot --seeds 0,2 --days 2-8 --out-dir ... --suffix ...   (a subset)

`--set` is always required and `--seeds` must be a subset of it, so the manifest always says whether the worlds are
robot dev worlds or calibration-only worlds.

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
`prereg_gate` checks it before anything is generated and refuses (fail closed) unless: the working tree is clean
(`git status --porcelain`, untracked files included); prereg/PREREG_TWIN.md is tracked at HEAD; the commit that added
it is an ancestor of `main`; and its blob at HEAD equals its blob on `main`. The manifest records HEAD, the main
commit, the commit that added PREREG, its git blob and its sha256, so G2 and the auditor can check the order later.
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
CONTRACT = "intercycle.dev_worlds/2"
PREREG = "prereg/PREREG_TWIN.md"
MAIN_REF = "main"


class Refusal(RuntimeError):
    pass


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _git_out(repo: Path, *args: str) -> str:
    r = _git(repo, *args)
    if r.returncode != 0:
        raise Refusal(f"git {' '.join(args)} failed: {r.stderr.strip()[:200]}")
    return r.stdout.strip()


def prereg_gate(repo: Path = ROOT, main_ref: str = MAIN_REF) -> dict:
    """PLAN_TWIN §4: PREREG v1 committed before any dev-world run. Refuses unless the tree is clean, PREREG is tracked
    and unmodified at HEAD, the commit that added it is on `main_ref`, and HEAD's PREREG equals main's (module doc)."""
    if _git(repo, "ls-files", "--error-unmatch", "--", PREREG).returncode != 0:
        raise Refusal(f"{PREREG} is not tracked in {repo}: commit PREREG v1 on {main_ref} first")
    dirty = _git_out(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise Refusal(f"working tree is not clean (first entries: {dirty.splitlines()[:3]}); commit or remove first")
    if _git(repo, "diff", "--quiet", "HEAD", "--", "prereg/").returncode != 0:
        raise Refusal("prereg/ differs from HEAD")
    head = _git_out(repo, "rev-parse", "HEAD")
    added = _git_out(repo, "log", "--diff-filter=A", "--format=%H", "HEAD", "--", PREREG).splitlines()
    if not added:
        raise Refusal(f"no commit adding {PREREG} in the history of HEAD")
    if _git(repo, "merge-base", "--is-ancestor", added[0], main_ref).returncode != 0:
        raise Refusal(f"the commit that added {PREREG} ({added[0][:10]}) is not on {main_ref}")
    blob = _git_out(repo, "rev-parse", f"HEAD:{PREREG}")
    if blob != _git_out(repo, "rev-parse", f"{main_ref}:{PREREG}"):
        raise Refusal(f"{PREREG} at HEAD differs from {main_ref}'s")
    return {"head": head, "main_ref": main_ref, "main": _git_out(repo, "rev-parse", main_ref), "tree_clean": True,
            "prereg_path": PREREG, "prereg_added_in": added[0], "prereg_git_blob": blob,
            "prereg_sha256_lf": sha_lf(repo / PREREG)}


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


def check_seeds(seeds: list[int], set_name: str) -> list[int]:
    """Every seed must be a pre-registered dev seed of `set_name` (always given); duplicates refused."""
    if set_name not in SETS:
        raise Refusal(f"unknown set {set_name!r}; give --set robot or --set calib")
    if not seeds:
        raise Refusal("no seed given")
    if len(set(seeds)) != len(seeds):
        raise Refusal(f"repeated seeds: {seeds}")
    bad = [s for s in seeds if s not in SETS[set_name]]
    if bad:
        raise Refusal(f"seeds {bad} are not pre-registered dev seeds of set {set_name} "
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


def main(argv: list[str] | None = None, runner=subprocess.run, gate=prereg_gate) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=sorted(SETS), required=True, help="a pre-registered dev world set")
    ap.add_argument("--seeds", help="a subset of the set's seeds, e.g. 0,2 or 100-107 (default: the whole set)")
    ap.add_argument("--days", default="1-10")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suffix", required=True, help="file-name suffix of each generated day (see the module doc)")
    ap.add_argument("--dry", action="store_true", help="print the jobs, generate nothing")
    a = ap.parse_args(argv)
    try:
        if os.environ.get("INTERCYCLE_SEALED_RUN"):
            raise Refusal("INTERCYCLE_SEALED_RUN is set: this script generates dev worlds only")
        seed_list = _range(a.seeds) if a.seeds else list(SETS[a.set])
        seeds = check_seeds(seed_list, a.set)
        days = check_days(_range(a.days))
        out_dir = Path(a.out_dir)
        jobs = plan(seeds, days, out_dir, a.suffix)
        manifest = out_dir / "dev_worlds.json"
        if manifest.exists():
            raise Refusal(f"{manifest} exists; dev worlds are generated once into a fresh directory")
        prereg = gate()                     # PREREG v1 committed on main, clean tree (module doc)
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
    body = {"contract": CONTRACT, "set": a.set, "seeds": seeds, "days": days, "prereg": prereg,
            "inputs_sha256_lf": {p: sha_lf(ROOT / p) for p in HASHED},
            "jobs": [{k: j[k] for k in ("world_seed", "day", "out")} for j in jobs]}
    manifest.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(jobs)} days written; manifest {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
