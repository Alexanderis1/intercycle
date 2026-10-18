"""Write the G2 generator / dynamics hash record (PLAN_TWIN §4 G2 row: "Generator and dynamics hashes committed").

    python scripts/write_g2_hashes.py            # at G2: write prereg/G2_HASHES.json from the committed files
    python scripts/write_g2_hashes.py --check    # compare the committed record with the working files; writes nothing

LEAD DECISION (Sun ~06:35): the record is prereg/G2_HASHES.json, contract twin.g2_hashes/1, holding the LF-normalised
sha256 of twin/daygen.py, twin/dynamics.json, twin/engine.py, twin/camera.py and twin/site.yaml (the sealed_run.G2_KEYS
inputs). scripts/sealed_run.py refuses a sealed run unless that record is in the FREEZE tree and every one of those
files hashes to it. The record format and the hashing live in sealed_run (`g2_record`), so writer and checker agree.

Fail closed: a missing file, or one that is modified / untracked in git (the record must describe committed content),
is refused and nothing is written. Commit the record yourself after it is written.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sealed_run():
    spec = importlib.util.spec_from_file_location("sealed_run_g2", REPO / "scripts" / "sealed_run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/sealed_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sealed_run_g2"] = mod
    spec.loader.exec_module(mod)
    return mod


def render(rec: dict) -> str:
    return json.dumps(rec, indent=1, sort_keys=True) + "\n"


def uncommitted(repo: Path, paths: list[str]) -> list[str]:
    """The paths among `paths` that git reports as modified, staged or untracked."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *paths], cwd=repo,
                         capture_output=True, check=True).stdout.decode("utf-8", "replace")
    return [ln for ln in out.splitlines() if ln.strip()]


def write(repo: Path = REPO) -> Path:
    SR = _sealed_run()
    paths = [SR.INPUTS[k] for k in SR.G2_KEYS]
    rec = SR.g2_record(repo)                                   # refuses a missing file
    dirty = uncommitted(repo, paths)
    if dirty:
        raise SR.Refusal("G2 files are not committed (commit them first):\n" + "\n".join(dirty))
    out = repo / SR.G2_HASHES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rec), encoding="utf-8", newline="\n")
    return out


def check(repo: Path = REPO) -> dict:
    SR = _sealed_run()
    p = repo / SR.G2_HASHES_FILE
    if not p.is_file():
        raise SR.Refusal(f"{SR.G2_HASHES_FILE} does not exist")
    have = json.loads(p.read_text(encoding="utf-8"))
    want = SR.g2_record(repo)
    if have != want:
        raise SR.Refusal(f"{SR.G2_HASHES_FILE} differs from the working files: {have} != {want}")
    return want


def main() -> None:
    ap = argparse.ArgumentParser(description="write / check the G2 hash record (prereg/G2_HASHES.json)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    try:
        if a.check:
            print(render(check()), end="")
        else:
            print(f"wrote {write()}")
    except Exception as e:  # noqa: BLE001 - every refusal is reported, exit 1
        sys.exit(f"REFUSED: {e}")


if __name__ == "__main__":
    main()
