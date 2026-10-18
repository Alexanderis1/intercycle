"""The sealed test run (PLAN_TWIN §5 "Sealed-test enforcement", §4 FREEZE row). Run by the CUSTODIAN only, on the laptop.

    python scripts/sealed_run.py check   [--sealed-root C:\\intercycle_sealed]    # the refusals only; writes nothing
    python scripts/sealed_run.py derive  [--sealed-root ...] [--fallback]        # custodian: seeds after FREEZE
    python scripts/sealed_run.py run --stages clips [--sealed-root ...]         # refusals, RUNLOG, then the stages

Refusals (every one fails closed, before INTERCYCLE_SEALED_RUN=1 is set and before anything is read from the sealed
root apart from the hash checks):

1. the tree is clean (`git status --porcelain --untracked-files=all` is empty);
2. prereg/FREEZE holds a 40-hex commit sha, and that commit is an ancestor of HEAD;
3. `git diff --name-only --no-renames <FREEZE> HEAD` lists nothing but prereg/FREEZE and prereg/TEST_SEED_COMMIT;
4. every input hash matches:
   * MANIFEST (sealed/MANIFEST.sha256), LABELS (sealed/LABELS.sha256), the generator (twin/daygen.py) and the dynamics
     (twin/dynamics.json) are tracked in the FREEZE tree and the working file hashes to the FREEZE blob (LF-normalised,
     so a CRLF checkout gives the same value);
   * MANIFEST and LABELS parse with the one contract parser (`twin.schemas.prereg.parse_sealed`); every clip file in
     <sealed-root>/clips hashes to its MANIFEST line, no clip is missing or extra, and every MANIFEST id is a `test`
     row of the active split with the same sha256; every LABELS file under <sealed-root> hashes to its line, and the
     LABELS sample list (`twin.label_samples/1`) carries the `#sample_list` hash;
   * TEST_SEED_COMMIT = sha256(salt ‖ seeds) for the salt and seeds revealed in <sealed-root>/seeds.json, the seeds are
     seed_i = sha256(R ‖ i) (i = 0..15 as 4 big-endian bytes) and R is the randomness of the first drand round
     published after the FREEZE commit time (the commit that wrote prereg/FREEZE), checked against api.drand.sh
     (randomness = sha256(signature), the chained default beacon); or R came from `secrets.token_hex(16)` (source
     "fallback", recorded in RUNLOG and to be disclosed in the README).

Then it sets INTERCYCLE_SEALED_RUN=1 (the loader, the generator's seed check and the observer's placeholder check
read it), writes `results/sealed/RUNLOG.json` (timestamp, HEAD, the FREEZE sha, the input hashes, the drand round,
the revealed salt and seeds) and runs the requested stages:

* `clips`: the real-clip test. Every clip is read from <sealed-root>/clips, re-hashed against MANIFEST, uploaded as
  bytes to an EPHEMERAL Modal app (`app.run()`, no volume mounted) and re-hashed in the container, which runs the SAME
  per-clip ingest code as the dev batch, `intercycle.perception.esk_ingest.ingest_clip` (imported by name; the image
  is cloud/modal_perc.py's, so the detector, tracker and versions are the dev ones). Outputs go to
  <sealed-root>/perc/ (test tracks never enter the repo); only their sha256 go into RUNLOG.
* the CPU-double and Isaac batches belong to the EVAL runner; their stages are not wired here yet (refused).

This file is an orchestrator (twin/truth_guard.py ORCHESTRATORS): it never opens truth and never spells a truth
token in a string literal. Nothing here reads the sealed root unless the custodian runs it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from twin.rng import DEV_SEED_MAX, check_world_seed  # noqa: E402
from twin.schemas.labels import SampleList  # noqa: E402
from twin.schemas.prereg import SealedHashes, active_split_path, camera_class_of, parse_sealed  # noqa: E402

SEALED_ENV = "INTERCYCLE_SEALED_RUN"
DEFAULT_SEALED_ROOT = Path(r"C:\intercycle_sealed")
FREEZE = "prereg/FREEZE"
SEED_COMMIT = "prereg/TEST_SEED_COMMIT"
ALLOWED_AFTER_FREEZE = (FREEZE, SEED_COMMIT)
INPUTS = {"manifest": "sealed/MANIFEST.sha256", "labels": "sealed/LABELS.sha256",
          "generator": "twin/daygen.py", "dynamics": "twin/dynamics.json"}
RUNLOG = "results/sealed/RUNLOG.json"
SEEDS_FILE = "seeds.json"                  # in the sealed root: the custodian's reveal (salt, R, round)
N_SEEDS = 16
INGEST_MODULE = "intercycle.perception.esk_ingest"
PERC_IMAGE_MODULE = "cloud/modal_perc.py"   # the dev batch: its image (detector weights, tracker, versions) is reused
DRAND_URL = "https://api.drand.sh"
CLIPS_PER_CALL = 16                         # one detector load per call; 126 clips -> 8 calls
STAGES = ("clips",)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")


class Refusal(RuntimeError):
    """sealed_run.py refuses to start (or to go on)."""


# ------------------------------------------------------------------------------------------------ git + hashes

def git(repo: Path, *args: str, ok_codes: Iterable[int] = (0,)) -> subprocess.CompletedProcess:
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True)
    if p.returncode not in tuple(ok_codes):
        raise Refusal(f"git {' '.join(args)} failed ({p.returncode}): {p.stderr.decode('utf-8', 'replace').strip()}")
    return p


def lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def first_value(text: str) -> str:
    vals = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(vals) != 1:
        raise Refusal(f"expected exactly one value line, found {len(vals)}")
    return vals[0]


def check_clean(repo: Path) -> None:
    out = git(repo, "status", "--porcelain", "--untracked-files=all").stdout.decode("utf-8", "replace").strip()
    if out:
        raise Refusal(f"the tree is not clean:\n{out}")


def read_freeze(repo: Path) -> str:
    p = repo / FREEZE
    if not p.is_file():
        raise Refusal(f"{FREEZE} does not exist")
    try:
        sha = first_value(p.read_text(encoding="utf-8")).lower()
    except Refusal as e:
        raise Refusal(f"{FREEZE}: {e}") from None
    if not SHA40.match(sha):
        raise Refusal(f"{FREEZE} must hold a full 40-hex commit sha, got {sha!r}")
    got = git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}", ok_codes=(0, 1))
    if got.returncode != 0 or got.stdout.decode().strip() != sha:
        raise Refusal(f"{FREEZE} names {sha}, which is not a commit of this repo")
    return sha


def check_ancestor(repo: Path, sha: str) -> None:
    p = git(repo, "merge-base", "--is-ancestor", sha, "HEAD", ok_codes=(0, 1))
    if p.returncode != 0:
        raise Refusal(f"the FREEZE commit {sha} is not an ancestor of HEAD")


def check_diff(repo: Path, sha: str) -> list[str]:
    names = [n for n in git(repo, "diff", "--name-only", "--no-renames", sha, "HEAD").stdout.decode("utf-8").splitlines()
             if n]
    extra = sorted(set(names) - set(ALLOWED_AFTER_FREEZE))
    if extra:
        raise Refusal(f"files changed since FREEZE besides {list(ALLOWED_AFTER_FREEZE)}: {extra}")
    return names


def freeze_commit_time(repo: Path) -> int:
    """Commit time (unix s) of the commit that last wrote prereg/FREEZE."""
    out = git(repo, "log", "-1", "--format=%ct", "--", FREEZE).stdout.decode().strip()
    if not out:
        raise Refusal(f"{FREEZE} was never committed")
    return int(out)


def check_inputs(repo: Path, sha: str) -> dict[str, dict]:
    """Every input is tracked in the FREEZE tree and the working file hashes to the FREEZE blob (LF-normalised)."""
    out = {}
    for name, rel in INPUTS.items():
        work = repo / rel
        if not work.is_file():
            raise Refusal(f"input {name} ({rel}) does not exist")
        blob = git(repo, "cat-file", "blob", f"{sha}:{rel}", ok_codes=(0, 128))
        if blob.returncode != 0:
            raise Refusal(f"input {name} ({rel}) is not in the FREEZE tree {sha[:12]}")
        h_work, h_frozen = sha256(lf(work.read_bytes())), sha256(lf(blob.stdout))
        if h_work != h_frozen:
            raise Refusal(f"input {name} ({rel}) hash {h_work[:16]} != FREEZE {h_frozen[:16]}")
        out[name] = {"path": rel, "sha256": h_work}
    return out


def load_sealed(repo: Path, name: str, kind: str) -> SealedHashes:
    try:
        s = parse_sealed((repo / INPUTS[name]).read_bytes().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - any parse error is a refusal
        raise Refusal(f"{INPUTS[name]} is not a twin.sealed/1 file: {e}") from None
    if s.kind != kind:
        raise Refusal(f"{INPUTS[name]} is a {s.kind} file, expected {kind}")
    return s


def _files_under(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} if root.is_dir() else set()


def check_manifest_files(repo: Path, sealed_root: Path, man: SealedHashes) -> list[dict]:
    """Every clip in <sealed_root>/clips hashes to its MANIFEST line; none missing or extra; each id is a test row of the
    active split with the same sha256. Returns the clip jobs (id, path, sha256, camera, framing)."""
    want = {f.path: f.sha256 for f in man.files}
    have = {p for p in _files_under(sealed_root / "clips")}
    have = {f"clips/{p}" for p in have}
    missing, extra = sorted(set(want) - have), sorted(have - set(want))
    if missing or extra:
        raise Refusal(f"sealed clips differ from MANIFEST: {len(missing)} missing {missing[:5]}, "
                      f"{len(extra)} extra {extra[:5]}")
    split = _split_rows(repo)
    jobs = []
    for f in man.files:
        cid = Path(f.path).stem
        row = split.get(cid)
        if row is None or row["split"] != "test":
            raise Refusal(f"MANIFEST clip {cid} is not a test row of the active split")
        if row["sha256"] != f.sha256:
            raise Refusal(f"MANIFEST sha256 of {cid} differs from the active split")
        got = sha256((sealed_root / f.path).read_bytes())
        if got != f.sha256:
            raise Refusal(f"clip {f.path} hashes to {got[:16]}, MANIFEST says {f.sha256[:16]}")
        framing = row.get("framing_id") or ""
        if not framing:
            raise Refusal(f"the active split has no framing_id for {cid}")
        jobs.append({"clip_id": cid, "path": f.path, "sha256": f.sha256, "framing": framing,
                     "camera": camera_class_of(cid)[0]})
    return jobs


def _split_rows(repo: Path) -> dict[str, dict]:
    try:
        path = active_split_path(repo)
    except Exception as e:  # noqa: BLE001
        raise Refusal(f"active split: {e}") from None
    rows = list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8"))))
    return {r["clip_id"]: r for r in rows}


def check_label_files(sealed_root: Path, lab: SealedHashes) -> None:
    """Every LABELS file under <sealed_root> hashes to its line, and the sample list among them carries #sample_list."""
    samples = []
    for f in lab.files:
        p = sealed_root / f.path
        if not p.is_file():
            raise Refusal(f"label file {f.path} is missing from the sealed root")
        data = p.read_bytes()
        if sha256(data) != f.sha256:
            raise Refusal(f"label file {f.path} hashes to {sha256(data)[:16]}, LABELS says {f.sha256[:16]}")
        if p.suffix == ".json":
            try:
                d = json.loads(data.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("contract") == "twin.label_samples/1":
                samples.append(SampleList.model_validate(d))
    if not any(s.items_sha256 == lab.sample_list_sha256 for s in samples):
        raise Refusal("no LABELS sample list (twin.label_samples/1) matches #sample_list")


# ------------------------------------------------------------------------------------------------ seeds

def seed_digest(r: bytes, i: int) -> bytes:
    return hashlib.sha256(r + int(i).to_bytes(4, "big")).digest()


def derive_seeds(r_hex: str, n: int = N_SEEDS) -> list[str]:
    r = bytes.fromhex(r_hex)
    return [seed_digest(r, i).hex() for i in range(n)]


def seed_int(seed_hex: str) -> int:
    """The integer world seed of a test seed (first 8 bytes, big-endian). A value in the dev range is refused."""
    v = int(seed_hex[:16], 16)
    if v <= DEV_SEED_MAX:
        raise Refusal(f"test seed {seed_hex[:16]} falls in the dev range 0..{DEV_SEED_MAX}")
    return v


def seed_commit(salt_hex: str, seeds_hex: list[str]) -> str:
    return sha256(bytes.fromhex(salt_hex) + b"".join(bytes.fromhex(s) for s in seeds_hex))


def drand_fetch(path: str) -> dict:
    with urllib.request.urlopen(f"{DRAND_URL}{path}", timeout=20) as r:   # noqa: S310 - fixed https URL
        return json.loads(r.read().decode("utf-8"))


def round_time(info: dict, rnd: int) -> int:
    return int(info["genesis_time"]) + (int(rnd) - 1) * int(info["period"])


def first_round_after(info: dict, t: int) -> int:
    return (int(t) - int(info["genesis_time"])) // int(info["period"]) + 2


def check_drand(reveal: dict, t_freeze: int, fetch: Callable[[str], dict]) -> dict:
    info = fetch("/info")
    rnd = reveal.get("round")
    if not isinstance(rnd, int) or rnd < 1:
        raise Refusal("seeds.json: source drand needs an integer round")
    if not (round_time(info, rnd - 1) <= t_freeze < round_time(info, rnd)):
        raise Refusal(f"drand round {rnd} is not the first round after the FREEZE commit time "
                      f"({first_round_after(info, t_freeze)} is)")
    beacon = fetch(f"/public/{rnd}")
    if int(beacon.get("round", -1)) != rnd or beacon.get("randomness") != reveal["randomness"]:
        raise Refusal(f"seeds.json randomness differs from drand round {rnd}")
    if sha256(bytes.fromhex(beacon["signature"])) != beacon["randomness"]:
        raise Refusal(f"drand round {rnd}: randomness != sha256(signature)")
    return {"source": "drand", "round": rnd, "round_time": round_time(info, rnd), "chain": info.get("hash"),
            "randomness": reveal["randomness"]}


def check_seeds(repo: Path, sealed_root: Path, t_freeze: int, fetch: Callable[[str], dict]) -> dict:
    p = repo / SEED_COMMIT
    if not p.is_file():
        raise Refusal(f"{SEED_COMMIT} does not exist")
    committed = first_value(p.read_text(encoding="utf-8")).lower()
    if not SHA64.match(committed):
        raise Refusal(f"{SEED_COMMIT} must hold one sha256 hex line")
    rp = sealed_root / SEEDS_FILE
    if not rp.is_file():
        raise Refusal(f"the seed reveal {rp} does not exist")
    reveal = json.loads(rp.read_text(encoding="utf-8"))
    try:
        r_hex, salt, seeds = str(reveal["randomness"]).lower(), str(reveal["salt"]).lower(), list(reveal["seeds"])
        for h in (r_hex, salt):
            bytes.fromhex(h)
    except (KeyError, TypeError, ValueError) as e:
        raise Refusal(f"seeds.json is malformed: {e!r}") from None
    if len(seeds) != N_SEEDS:
        raise Refusal(f"seeds.json holds {len(seeds)} seeds, need {N_SEEDS}")
    if seeds != derive_seeds(r_hex):
        raise Refusal("seeds.json seeds are not sha256(R ‖ i) of its randomness")
    if seed_commit(salt, seeds) != committed:
        raise Refusal(f"{SEED_COMMIT} != sha256(salt ‖ seeds) of the reveal")
    if reveal.get("source") == "drand":
        src = check_drand(dict(reveal, randomness=r_hex), t_freeze, fetch)
    elif reveal.get("source") == "fallback":
        if len(bytes.fromhex(r_hex)) != 16:
            raise Refusal("fallback randomness must be secrets.token_hex(16)")
        src = {"source": "fallback", "round": None, "randomness": r_hex,
               "disclose": "drand unreachable: R = secrets.token_hex(16) (README must say so)"}
    else:
        raise Refusal(f"seeds.json source {reveal.get('source')!r} is neither drand nor fallback")
    return {"test_seed_commit": committed, "salt": salt, "seeds": seeds, "seed_ints": [seed_int(s) for s in seeds],
            **src}


# ------------------------------------------------------------------------------------------------ preflight

def preflight(repo: Path, sealed_root: Path, fetch: Callable[[str], dict] = drand_fetch) -> dict:
    """Every PLAN §5 refusal, in order. Returns the RUNLOG record (without stages); raises Refusal."""
    check_clean(repo)
    sha = read_freeze(repo)
    check_ancestor(repo, sha)
    changed = check_diff(repo, sha)
    inputs = check_inputs(repo, sha)
    man = load_sealed(repo, "manifest", "MANIFEST")
    lab = load_sealed(repo, "labels", "LABELS")
    jobs = check_manifest_files(repo, sealed_root, man)
    check_label_files(sealed_root, lab)
    t_freeze = freeze_commit_time(repo)
    seeds = check_seeds(repo, sealed_root, t_freeze, fetch)
    inputs["test_seed_commit"] = {"path": SEED_COMMIT, "sha256": seeds["test_seed_commit"]}
    head = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    return {"contract": "twin.sealed_runlog/1", "head": head, "freeze_sha": sha, "freeze_commit_time": t_freeze,
            "changed_since_freeze": changed, "inputs": inputs, "manifest_aggregate": man.aggregate,
            "labels_aggregate": lab.aggregate, "n_clips": len(jobs), "seeds": seeds, "jobs": jobs}


def write_runlog(repo: Path, rec: dict) -> Path:
    p = repo / RUNLOG
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ real-clip stage

def verify_clip_bytes(clip_id: str, data: bytes, want: str) -> None:
    """Upload check (run on both sides): the bytes are the MANIFEST file."""
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise PermissionError(f"clip {clip_id}: uploaded bytes hash to {got[:16]}, MANIFEST says {want[:16]}")


def container_ingest(batch: list[dict]) -> list[dict]:
    """Body of the ephemeral Modal function: re-hash each clip, then the dev per-clip ingest, by module name."""
    import hashlib as _h
    import importlib
    import tempfile
    from pathlib import Path as _P

    ing = importlib.import_module("intercycle.perception.esk_ingest")
    from intercycle.perception.detector import RFDetrDetector

    det = RFDetrDetector(size="medium", device="cuda", precision="fp32", threshold=0.1,   # as cloud/modal_perc.py
                         keep=("person", "truck"), class_map={"truck": "truck"})
    rois = ing.load_press_rois("/root/rois.json")
    out = []
    for job in batch:
        data = job.pop("data")
        if _h.sha256(data).hexdigest() != job["sha256"]:
            raise PermissionError(f"clip {job['clip_id']}: bytes in the container differ from MANIFEST")
        with tempfile.TemporaryDirectory() as td:
            clip = _P(td) / f"{job['clip_id']}.mp4"
            clip.write_bytes(data)
            res = ing.ingest_clip(clip, job["clip_id"], "test", int(job["camera"]), job["framing"],
                                  rois.get(job["framing"], {}), det.detect)
            files = {}
            for p in ing.write_outputs(res, _P(td) / "out"):
                files[_P(p).relative_to(_P(td) / "out").as_posix()] = _P(p).read_bytes()
        out.append({"clip_id": job["clip_id"], "stats": res.stats, "files": files})
    return out


def modal_runner(batches: list[list[dict]]) -> list[dict]:
    """Ephemeral app (no volume) on cloud/modal_perc.py's image; one call per batch, all spawned up front."""
    import importlib.util

    import modal

    spec = importlib.util.spec_from_file_location("modal_perc", REPO / PERC_IMAGE_MODULE)
    if spec is None or spec.loader is None:
        raise Refusal(f"{PERC_IMAGE_MODULE} is missing (PERC not merged)")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    app = modal.App("intercycle-sealed-clips")
    fn = app.function(image=mp.image, gpu="L4", cpu=4, memory=8192, timeout=3600, serialized=True,
                      max_containers=6)(container_ingest)
    with app.run():
        calls = [fn.spawn(b) for b in batches]
        return [r for c in calls for r in c.get()]


def run_clips(sealed_root: Path, jobs: list[dict], runner: Callable[[list[list[dict]]], list[dict]]) -> dict:
    batches, cur = [], []
    for j in jobs:
        data = (sealed_root / j["path"]).read_bytes()
        verify_clip_bytes(j["clip_id"], data, j["sha256"])
        cur.append(dict(j, data=data))
        if len(cur) == CLIPS_PER_CALL:
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)
    t0 = time.time()
    results = runner(batches)
    got = sorted(r["clip_id"] for r in results)
    if got != sorted(j["clip_id"] for j in jobs):
        raise RuntimeError(f"the container returned {len(got)} clips for {len(jobs)} uploaded")
    out_root = sealed_root / "perc"
    hashes = {}
    for r in results:
        for rel, data in r["files"].items():
            dst = out_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            hashes[rel] = sha256(data)
        stats = out_root / "stats.jsonl"
        with stats.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r["stats"], default=str) + "\n")
    return {"ingest": INGEST_MODULE + ".ingest_clip", "n_clips": len(results), "wall_s": round(time.time() - t0, 1),
            "out": str(out_root), "outputs_sha256": dict(sorted(hashes.items()))}


# ------------------------------------------------------------------------------------------------ commands

def run(repo: Path, sealed_root: Path, stages: list[str], fetch: Callable[[str], dict] = drand_fetch,
        runner: Callable[[list[list[dict]]], list[dict]] | None = None) -> dict:
    bad = sorted(set(stages) - set(STAGES))
    if bad:
        raise Refusal(f"stages {bad} are not wired in sealed_run.py (known: {list(STAGES)})")
    rec = preflight(repo, sealed_root, fetch)
    os.environ[SEALED_ENV] = "1"                     # only after every refusal passed
    for s in rec["seeds"]["seed_ints"]:
        check_world_seed(s)                          # the generator's own guard now admits the test seeds
    jobs = rec.pop("jobs")
    rec["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rec["sealed_flag"] = SEALED_ENV + "=1"
    rec["stages"] = {s: {"status": "started"} for s in stages}
    write_runlog(repo, rec)
    if "clips" in stages:
        try:
            rec["stages"]["clips"] = dict(run_clips(sealed_root, jobs, runner or modal_runner), status="done")
        except Exception as e:  # noqa: BLE001 - recorded, then re-raised
            rec["stages"]["clips"] = {"status": "error", "error": repr(e)}
            write_runlog(repo, rec)
            raise
    rec["finished"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    write_runlog(repo, rec)
    return rec


def derive(repo: Path, sealed_root: Path, fallback: bool, fetch: Callable[[str], dict] = drand_fetch,
           now: Callable[[], float] = time.time) -> dict:
    """Custodian, after FREEZE is committed: R, 16 seeds, a fresh salt -> <sealed_root>/seeds.json; prints the line
    to commit as prereg/TEST_SEED_COMMIT."""
    rp = sealed_root / SEEDS_FILE
    if rp.exists():
        raise Refusal(f"{rp} exists; seeds are derived once")
    t_freeze = freeze_commit_time(repo)
    if fallback:
        rec = {"source": "fallback", "round": None, "randomness": secrets.token_hex(16)}
    else:
        info = fetch("/info")
        rnd = first_round_after(info, t_freeze)
        if now() < round_time(info, rnd):
            raise Refusal(f"drand round {rnd} is published at {round_time(info, rnd)}; wait and re-run")
        b = fetch(f"/public/{rnd}")
        rec = {"source": "drand", "round": rnd, "randomness": b["randomness"]}
    seeds = derive_seeds(rec["randomness"])
    for sd in seeds:
        seed_int(sd)                                  # refuses a seed that would land in the dev range
    salt = secrets.token_hex(16)
    rec.update(salt=salt, seeds=seeds, freeze_commit_time=t_freeze)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return {"seed_commit": seed_commit(salt, seeds), "reveal": str(rp), "source": rec["source"], "round": rec["round"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the sealed test run (PLAN_TWIN §5); custodian only")
    ap.add_argument("command", choices=("check", "derive", "run"))
    ap.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    ap.add_argument("--stages", default="clips")
    ap.add_argument("--fallback", action="store_true", help="derive: drand unreachable, R = secrets.token_hex(16)")
    a = ap.parse_args(argv)
    try:
        if a.command == "check":
            rec = preflight(REPO, a.sealed_root)
            rec.pop("jobs")
            print(json.dumps({k: rec[k] for k in ("head", "freeze_sha", "inputs", "n_clips")}, indent=1))
            print("sealed_run: every refusal passed")
        elif a.command == "derive":
            d = derive(REPO, a.sealed_root, a.fallback)
            print(f"write this line to {SEED_COMMIT} and commit only that file:\n{d['seed_commit']}")
            print(f"reveal kept at {d['reveal']} (source {d['source']}, round {d['round']})")
        else:
            rec = run(REPO, a.sealed_root, [s for s in a.stages.split(",") if s])
            print(f"sealed_run: done; {RUNLOG} written ({', '.join(rec['stages'])})")
    except Refusal as e:
        print(f"sealed_run REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
