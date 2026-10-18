"""The sealed test run (PLAN_TWIN §5 "Sealed-test enforcement", §4 FREEZE row). Run by the CUSTODIAN only, on the laptop.

    python scripts/sealed_run.py check   [--sealed-root C:\\intercycle_sealed]    # the refusals only; writes nothing
    python scripts/sealed_run.py derive  [--sealed-root ...] [--fallback]        # custodian: seeds after FREEZE
    python scripts/sealed_run.py run --stages clips [--sealed-root ...]         # refusals, RUNLOG, then the stages
    python scripts/sealed_run.py dev-dry-run --dev-root D --clips 0_dv1,3_dv2   # the clips stage on DEV clips only

Refusals (every one fails closed, before INTERCYCLE_SEALED_RUN=1 is set and before anything is read from the sealed
root apart from the hash checks):

1. the tree is clean (`git status --porcelain --untracked-files=all` is empty, except results/sealed/RUNLOG.json,
   which every attempt appends to and the custodian may commit);
2. prereg/FREEZE holds a 40-hex commit sha, and that commit is an ancestor of HEAD;
3. `git diff --name-only --no-renames <FREEZE> HEAD` lists nothing but prereg/FREEZE, prereg/TEST_SEED_COMMIT and
   sealed_run's own outputs under results/sealed/ (LEAD DECISION, Sun ~06:35: sealed_run may be invoked more than once,
   so a committed earlier attempt's results/sealed/** never blocks a re-run; every other path is refused);
3b. RUNLOG is append-only: it parses as `twin.sealed_runlog/2` (a list of records, each with attempt, timestamp,
   HEAD, stages and status), and its committed version (HEAD) is a prefix of the working file; every append re-reads
   the file and refuses unless what is there is the prefix it extends;
4. every input hash matches:
   * MANIFEST (sealed/MANIFEST.sha256), LABELS (sealed/LABELS.sha256) and the G2 files (the generator twin/daygen.py,
     the dynamics twin/dynamics.json, twin/engine.py, twin/camera.py, twin/site.yaml) are tracked in the FREEZE tree
     and the working file hashes to the FREEZE blob (LF-normalised, so a CRLF checkout gives the same value); the G2
     files' hashes also equal the committed G2 record prereg/G2_HASHES.json (LEAD DECISION: written at G2 by
     scripts/write_g2_hashes.py, contract twin.g2_hashes/1, exactly those five paths);
   * MANIFEST and LABELS parse with the one contract parser (`twin.schemas.prereg.parse_sealed`); every clip file in
     <sealed-root>/clips hashes to its MANIFEST line, no clip is missing or extra, and every MANIFEST id is a `test`
     row of the active split with the same sha256; every LABELS file under <sealed-root> hashes to its line, and the
     LABELS sample list (`twin.label_samples/1`) carries the `#sample_list` hash;
   * TEST_SEED_COMMIT = sha256(salt ‖ seeds) for the salt and seeds revealed in <sealed-root>/seeds.json, the seeds are
     seed_i = sha256(R ‖ i) (i = 0..15 as 4 big-endian bytes) and R is the randomness of the first drand round
     published after the FREEZE commit time (the commit that wrote prereg/FREEZE), checked against api.drand.sh
     (randomness = sha256(signature), the chained default beacon); or R came from `secrets.token_hex(16)` (source
     "fallback", recorded in RUNLOG and to be disclosed in the README);
   * the round is pinned to the FREEZE commit's committer date, cross-checked: the author date of that commit is within
     DATE_SKEW_S of it, and the reveal's own clock (`derived_at`, written by `derive`) lies between the round's
     publication and MAX_DERIVE_LAG_S after the FREEZE commit time.

`derive` runs refusals 1-2 (clean tree, FREEZE, ancestor) before it derives anything, and refuses to derive more than
MAX_DERIVE_LAG_S after the FREEZE commit (re-commit FREEZE instead). RESIDUAL RISK (to disclose): every clock here is
the custodian's own (git dates can be set with GIT_COMMITTER_DATE / GIT_AUTHOR_DATE, `derived_at` by the laptop
clock); the cross-checks only force a forger to backdate consistently and bound how far (a backdated FREEZE inside the
lag window could choose among the drand rounds of that window). A push time recorded by a third party (the remote's
reflog / API) would bound it independently; this repo has no such record wired.

Then it sets INTERCYCLE_SEALED_RUN=1 (the loader, the generator's seed check and the observer's placeholder check
read it), APPENDS a `started` record to `results/sealed/RUNLOG.json` (attempt number, timestamp, HEAD, stages, the
FREEZE sha, the input hashes, the drand round, the revealed salt and seeds, the attempt's perc dir) and runs the
requested stages, then appends the attempt's closing record (status `done` or `error`, the stage results). A re-run
is a new attempt with its own records; nothing earlier is rewritten.

* `clips`: the real-clip test. Every clip is read from <sealed-root>/clips, re-hashed against MANIFEST, uploaded as
  bytes to an EPHEMERAL Modal app (`app.run()`, no volume mounted) and re-hashed in the container, which runs the SAME
  per-clip ingest code as the dev batch, `intercycle.perception.esk_ingest.ingest_clip` (imported by name; the image
  is cloud/modal_perc.py's, so the detector, tracker and versions are the dev ones). The container function lives in
  scripts/sealed_ingest.py, shipped with `add_local_python_source` and imported by name in the container (not
  cloudpickled: the client and image Pythons differ). Outputs go to a FRESH <sealed-root>/perc/attempt_<NNN>/ per
  attempt (test tracks never enter the repo); only their sha256 go into RUNLOG.
* `double`: the sealed CPU-double batch (USER DECISION Sun ~07:00 CEST, the CUT design: static, learned, reactive x 16
  test worlds x days 9-10 x zones = 192 missions), launched through scripts/run_double.py `run_sealed` with the 16
  revealed test seeds (world i = seed_ints[i]) after every refusal passed (so after the robot FREEZE), into a FRESH
  <repo>/out/sealed_double/attempt_<NNN>/ (the twin_eval runs root is its runs/). run_double refuses unless the
  sealed flag is set. Isaac stays outside sealed_run (descriptive only; refused here).

`dev-dry-run` exercises the clips stage end to end on DEV clips before the sealed run: it copies the named clips (each
must be a `dev` row of the active split) from $INTERCYCLE_DATA/clips into a scratch dev root (never the sealed root),
writes a dev MANIFEST there with `make_sealed`, and runs the same jobs / upload / container path with split "dev". It
never sets INTERCYCLE_SEALED_RUN and never writes RUNLOG.

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
RUNLOG = "results/sealed/RUNLOG.json"
RUNLOG_CONTRACT = "twin.sealed_runlog/2"
RUNLOG_STATUS = ("started", "done", "error")
ALLOWED_DIRTY = (RUNLOG,)                  # the only path that may be dirty / untracked (each attempt appends to it)
ALLOWED_AFTER_FREEZE = (FREEZE, SEED_COMMIT, RUNLOG)
RESULTS_SEALED = "results/sealed/"         # sealed_run's own outputs: exempt from the FREEZE..HEAD diff rule (re-runs)
INPUTS = {"manifest": "sealed/MANIFEST.sha256", "labels": "sealed/LABELS.sha256",
          "generator": "twin/daygen.py", "dynamics": "twin/dynamics.json", "engine": "twin/engine.py",
          "camera": "twin/camera.py", "site": "twin/site.yaml"}
# The committed G2 record (LEAD DECISION, Sun ~06:35): prereg/G2_HASHES.json, written at G2 by
# scripts/write_g2_hashes.py as {"contract": "twin.g2_hashes/1", "files": {<repo path>: <LF-normalised sha256>}} for
# exactly the G2_KEYS inputs. Typed optional so a test can unset it (preflight then refuses).
G2_HASHES_FILE: str | None = "prereg/G2_HASHES.json"
G2_CONTRACT = "twin.g2_hashes/1"
G2_KEYS = ("generator", "dynamics", "engine", "camera", "site")
DATE_SKEW_S = 300                          # FREEZE commit: |committer date - author date| at most this
MAX_DERIVE_LAG_S = 3600                    # derive (the reveal's own clock) at most this long after the FREEZE commit
SEEDS_FILE = "seeds.json"                  # in the sealed root: the custodian's reveal (salt, R, round)
N_SEEDS = 16
INGEST_MODULE = "intercycle.perception.esk_ingest"
INGEST_CONTAINER_MODULE = "sealed_ingest"         # scripts/sealed_ingest.py: the container function, imported by name
PERC_IMAGE_MODULE = "cloud/modal_perc.py"   # the dev batch: its image (detector weights, tracker, versions) is reused
DRAND_URL = "https://api.drand.sh"
CLIPS_PER_CALL = 16                         # one detector load per call; 126 clips -> 8 calls
STAGES = ("clips", "double")
DOUBLE_OUT = "out/sealed_double"            # the CPU-double runs root per attempt (gitignored; the EVAL runner scores it)
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


def _porcelain_path(line: str) -> str:
    """The path of one `git status --porcelain` v1 line ("XY path"; a rename "XY a -> b" never equals a plain path)."""
    p = line[3:]
    return p[1:-1] if len(p) >= 2 and p[0] == p[-1] == '"' else p


def check_clean(repo: Path) -> None:
    out = git(repo, "status", "--porcelain", "--untracked-files=all").stdout.decode("utf-8", "replace")
    dirty = [ln for ln in out.splitlines() if ln.strip() and _porcelain_path(ln) not in ALLOWED_DIRTY]
    if dirty:
        raise Refusal("the tree is not clean:\n" + "\n".join(dirty))


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
    extra = sorted(n for n in set(names) if n not in ALLOWED_AFTER_FREEZE and not n.startswith(RESULTS_SEALED))
    if extra:
        raise Refusal(f"files changed since FREEZE besides {[FREEZE, SEED_COMMIT]} and {RESULTS_SEALED}**: {extra}")
    return names


def freeze_commit_time(repo: Path) -> int:
    """Commit time (unix s) of the commit that last wrote prereg/FREEZE."""
    return freeze_commit_dates(repo)[0]


def freeze_commit_dates(repo: Path) -> tuple[int, int]:
    """(committer date, author date), unix s, of the commit that last wrote prereg/FREEZE."""
    out = git(repo, "log", "-1", "--format=%ct %at", "--", FREEZE).stdout.decode().strip()
    if not out:
        raise Refusal(f"{FREEZE} was never committed")
    ct, at = (int(v) for v in out.split())
    return ct, at


def check_commit_dates(repo: Path) -> int:
    """The drand round is pinned to the FREEZE commit's committer date; its author date must agree within
    DATE_SKEW_S (a re-dated commit, e.g. an amend or a rebase that moved only one of them, is refused)."""
    ct, at = freeze_commit_dates(repo)
    if abs(ct - at) > DATE_SKEW_S:
        raise Refusal(f"the FREEZE commit's committer date {ct} and author date {at} differ by {abs(ct - at)} s "
                      f"(> {DATE_SKEW_S} s): re-commit prereg/FREEZE")
    return ct


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
    check_g2_hashes(repo, sha, out)
    return out


def g2_record(repo: Path) -> dict:
    """The G2 hash record of the working tree: {"contract", "files": {path: LF-normalised sha256}} for the G2_KEYS
    inputs (scripts/write_g2_hashes.py writes it at G2). A missing file is refused."""
    files = {}
    for k in G2_KEYS:
        p = repo / INPUTS[k]
        if not p.is_file():
            raise Refusal(f"G2 file {k} ({INPUTS[k]}) does not exist")
        files[INPUTS[k]] = sha256(lf(p.read_bytes()))
    return {"contract": G2_CONTRACT, "files": files}


def check_g2_hashes(repo: Path, sha: str, inputs: dict[str, dict]) -> dict[str, str]:
    """The G2 files' hashes equal the committed G2 record (G2_HASHES_FILE, tracked in the FREEZE tree): contract
    twin.g2_hashes/1, exactly the G2_KEYS paths, each a sha256 hex."""
    if G2_HASHES_FILE is None:
        raise Refusal("no committed G2 generator / dynamics hash record is named (sealed_run.G2_HASHES_FILE): the "
                      "lead names the file; refusing until then")
    blob = git(repo, "cat-file", "blob", f"{sha}:{G2_HASHES_FILE}", ok_codes=(0, 128))
    if blob.returncode != 0:
        raise Refusal(f"the G2 hash record {G2_HASHES_FILE} is not in the FREEZE tree {sha[:12]}")
    paths = sorted(INPUTS[k] for k in G2_KEYS)
    try:
        rec = json.loads(blob.stdout.decode("utf-8"))
        if rec["contract"] != G2_CONTRACT or sorted(rec["files"]) != paths:
            raise ValueError(f"contract {rec.get('contract')!r}, files {sorted(rec['files'])}")
        want = {k: str(rec["files"][INPUTS[k]]).lower() for k in G2_KEYS}
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise Refusal(f"{G2_HASHES_FILE} is not a {G2_CONTRACT} record of exactly {paths}: {e!r}") from None
    for k in G2_KEYS:
        if not SHA64.match(want[k]):
            raise Refusal(f"{G2_HASHES_FILE}: {k} is not a sha256 hex")
        if inputs[k]["sha256"] != want[k]:
            raise Refusal(f"input {k} ({inputs[k]['path']}) hash {inputs[k]['sha256'][:16]} != G2 record "
                          f"{want[k][:16]} ({G2_HASHES_FILE})")
    return want


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


def check_manifest_files(repo: Path, sealed_root: Path, man: SealedHashes, split: str = "test") -> list[dict]:
    """Every clip in <sealed_root>/clips hashes to its MANIFEST line; none missing or extra; each id is a `split` row
    (test; dev for --dev-dry-run) of the active split with the same sha256. Returns the clip jobs (id, path, sha256,
    camera, framing, split)."""
    want = {f.path: f.sha256 for f in man.files}
    have = {p for p in _files_under(sealed_root / "clips")}
    have = {f"clips/{p}" for p in have}
    missing, extra = sorted(set(want) - have), sorted(have - set(want))
    if missing or extra:
        raise Refusal(f"sealed clips differ from MANIFEST: {len(missing)} missing {missing[:5]}, "
                      f"{len(extra)} extra {extra[:5]}")
    rows = _split_rows(repo)
    jobs = []
    for f in man.files:
        cid = Path(f.path).stem
        row = rows.get(cid)
        if row is None or row["split"] != split:
            raise Refusal(f"MANIFEST clip {cid} is not a {split} row of the active split")
        if row["sha256"] != f.sha256:
            raise Refusal(f"MANIFEST sha256 of {cid} differs from the active split")
        got = sha256((sealed_root / f.path).read_bytes())
        if got != f.sha256:
            raise Refusal(f"clip {f.path} hashes to {got[:16]}, MANIFEST says {f.sha256[:16]}")
        framing = row.get("framing_id") or ""
        if not framing:
            raise Refusal(f"the active split has no framing_id for {cid}")
        jobs.append({"clip_id": cid, "path": f.path, "sha256": f.sha256, "framing": framing,
                     "camera": camera_class_of(cid)[0], "split": split})
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
    derived = reveal.get("derived_at")
    if not isinstance(derived, int):
        raise Refusal("seeds.json: source drand needs derived_at (the reveal's own clock, written by derive)")
    check_derive_clock(info, rnd, t_freeze, derived)
    return {"source": "drand", "round": rnd, "round_time": round_time(info, rnd), "chain": info.get("hash"),
            "randomness": reveal["randomness"], "derived_at": derived,
            "clock_checks": f"|ct - at| <= {DATE_SKEW_S} s; round_time <= derived_at <= t_freeze + {MAX_DERIVE_LAG_S} s",
            "disclose": "the round is pinned to the custodian's own clocks (commit dates, derive clock); see "
                        "scripts/sealed_run.py RESIDUAL RISK"}


def check_derive_clock(info: dict, rnd: int, t_freeze: int, derived_at: float) -> None:
    """The reveal's own clock: the round was already published when derive ran, and derive ran within
    MAX_DERIVE_LAG_S of the FREEZE commit (a FREEZE backdated further than that is refused)."""
    if derived_at < round_time(info, rnd):
        raise Refusal(f"derived_at {int(derived_at)} is before drand round {rnd} was published ({round_time(info, rnd)})")
    if derived_at - t_freeze > MAX_DERIVE_LAG_S:
        raise Refusal(f"derive ran {int(derived_at - t_freeze)} s after the FREEZE commit (> {MAX_DERIVE_LAG_S} s): the "
                      "commit date cannot be cross-checked; re-commit prereg/FREEZE and derive again")


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
    """Every PLAN §5 refusal, in order. Returns the RUNLOG record (without stages) plus the current RUNLOG records
    ("runlog") and the clip jobs ("jobs"); raises Refusal."""
    check_clean(repo)
    sha = read_freeze(repo)
    check_ancestor(repo, sha)
    changed = check_diff(repo, sha)
    runlog = check_runlog(repo, sha)
    inputs = check_inputs(repo, sha)
    man = load_sealed(repo, "manifest", "MANIFEST")
    lab = load_sealed(repo, "labels", "LABELS")
    jobs = check_manifest_files(repo, sealed_root, man)
    check_label_files(sealed_root, lab)
    t_freeze = check_commit_dates(repo)
    seeds = check_seeds(repo, sealed_root, t_freeze, fetch)
    inputs["test_seed_commit"] = {"path": SEED_COMMIT, "sha256": seeds["test_seed_commit"]}
    head = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    return {"head": head, "freeze_sha": sha, "freeze_commit_time": t_freeze, "changed_since_freeze": changed,
            "inputs": inputs, "manifest_aggregate": man.aggregate, "labels_aggregate": lab.aggregate,
            "n_clips": len(jobs), "seeds": seeds, "jobs": jobs, "runlog": runlog}


# ------------------------------------------------------------------------------------------------ RUNLOG (append-only)

def parse_runlog(text: str, where: str) -> list[dict]:
    """twin.sealed_runlog/2: {"contract", "records": [...]}; each record has attempt (1, 2, ... in order), timestamp,
    head, stages and status; an attempt opens with exactly one `started` record and has at most one closing record."""
    try:
        d = json.loads(text)
    except ValueError as e:
        raise Refusal(f"{where}: RUNLOG is not JSON: {e}") from None
    if not isinstance(d, dict) or d.get("contract") != RUNLOG_CONTRACT or not isinstance(d.get("records"), list):
        raise Refusal(f"{where}: RUNLOG is not a {RUNLOG_CONTRACT} record list")
    recs, opened, closed = d["records"], [], set()
    for r in recs:
        if not isinstance(r, dict) or not {"attempt", "timestamp", "head", "stages", "status"} <= set(r):
            raise Refusal(f"{where}: RUNLOG record without attempt / timestamp / head / stages / status: {r!r}"[:300])
        a, st = r["attempt"], r["status"]
        if not isinstance(a, int) or st not in RUNLOG_STATUS:
            raise Refusal(f"{where}: RUNLOG record attempt {a!r} status {st!r} is malformed")
        if st == "started":
            if a != len(opened) + 1:
                raise Refusal(f"{where}: RUNLOG attempt {a} opens out of order (expected {len(opened) + 1})")
            opened.append(a)
        elif a not in opened or a in closed:
            raise Refusal(f"{where}: RUNLOG closes attempt {a} twice or before it started")
        else:
            closed.add(a)
    return recs


def _runlog_at(repo: Path, rev: str) -> list[dict] | None:
    """The RUNLOG records committed at `rev`, or None when that tree has no RUNLOG."""
    blob = git(repo, "cat-file", "blob", f"{rev}:{RUNLOG}", ok_codes=(0, 128))
    if blob.returncode != 0:
        return None
    return parse_runlog(blob.stdout.decode("utf-8"), f"{rev[:12]}:{RUNLOG}")


def check_runlog(repo: Path, sha: str) -> list[dict]:
    """RUNLOG is append-only across history, not only against the working file. The chain is: the FREEZE tree's RUNLOG
    (absent counts as no records), then the RUNLOG of EVERY commit in FREEZE..HEAD in topological order (absent counts
    as no records, so a deletion is a truncation), then the working file. Each version must parse and be a prefix of
    the next, so no committed record is ever edited or dropped. Every commit is read (not only those `git log -- path`
    lists), so path simplification over merges hides nothing; two branches that each appended different records
    fail closed. Returns the working records."""
    prev, where = _runlog_at(repo, sha) or [], f"the FREEZE tree {sha[:12]}"
    revs = git(repo, "rev-list", "--reverse", "--topo-order", f"{sha}..HEAD").stdout.decode().split()
    p = repo / RUNLOG
    work = parse_runlog(p.read_bytes().decode("utf-8"), RUNLOG) if p.is_file() else []
    for rev, recs in [(r, _runlog_at(repo, r) or []) for r in revs] + [("working", work)]:
        name = "the working file" if rev == "working" else f"commit {rev[:12]}"
        if recs[:len(prev)] != prev:
            raise Refusal(f"{RUNLOG} is not an append-only extension of its committed version ({where} has "
                          f"{len(prev)} records, {name} {len(recs)}, not as a prefix)")
        prev, where = recs, name
    return work


def append_runlog(repo: Path, known: list[dict], rec: dict) -> list[dict]:
    """Append one record. Refused unless the file on disk still holds exactly `known` (what this attempt read or last
    wrote): the existing RUNLOG must be a prefix of the new one."""
    p = repo / RUNLOG
    cur = parse_runlog(p.read_bytes().decode("utf-8"), RUNLOG) if p.is_file() else []
    if cur != known:
        raise Refusal(f"{RUNLOG} changed under this attempt ({len(cur)} records on disk, {len(known)} expected): "
                      "the existing RUNLOG is not a prefix of the new one")
    new = known + [rec]
    parse_runlog(json.dumps({"contract": RUNLOG_CONTRACT, "records": new}), "new RUNLOG")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"contract": RUNLOG_CONTRACT, "records": new}, indent=1, sort_keys=True) + "\n",
                 encoding="utf-8", newline="\n")
    return new


# ------------------------------------------------------------------------------------------------ real-clip stage

def verify_clip_bytes(clip_id: str, data: bytes, want: str) -> None:
    """Upload check (run on both sides): the bytes are the MANIFEST file."""
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise PermissionError(f"clip {clip_id}: uploaded bytes hash to {got[:16]}, MANIFEST says {want[:16]}")


def _ingest_module():
    """scripts/sealed_ingest.py imported by its plain module name (the name Modal imports it by in the container)."""
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import importlib

    return importlib.import_module(INGEST_CONTAINER_MODULE)


def modal_function_spec() -> dict:
    """How the container function is registered (checked by the tests): imported by module name, not serialized."""
    si = _ingest_module()
    return {"module": si.container_ingest.__module__, "qualname": si.container_ingest.__qualname__,
            "serialized": False, "include_source": False, "local_python_source": INGEST_CONTAINER_MODULE}


def modal_runner(batches: list[list[dict]]) -> list[dict]:
    """Ephemeral app (no volume) on cloud/modal_perc.py's image plus scripts/sealed_ingest.py
    (add_local_python_source); one call per batch, all spawned up front. container_ingest is a global function of an
    importable module, registered WITHOUT serialized=True: the container imports it by name, so the client Python
    need not match the image's."""
    import importlib.util

    import modal

    spec = importlib.util.spec_from_file_location("modal_perc", REPO / PERC_IMAGE_MODULE)
    if spec is None or spec.loader is None or not (REPO / PERC_IMAGE_MODULE).is_file():
        raise Refusal(f"{PERC_IMAGE_MODULE} is missing (PERC not merged)")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    si = _ingest_module()
    fs = modal_function_spec()
    image = mp.image.add_local_python_source(fs["local_python_source"])
    app = modal.App("intercycle-sealed-clips")
    fn = app.function(image=image, gpu="L4", cpu=4, memory=8192, timeout=3600, max_containers=6,
                      include_source=fs["include_source"])(si.container_ingest)
    with app.run():
        calls = [fn.spawn(b) for b in batches]
        return [r for c in calls for r in c.get()]


def run_clips(sealed_root: Path, jobs: list[dict], runner: Callable[[list[list[dict]]], list[dict]],
              out_root: Path) -> dict:
    """Upload the MANIFEST-checked clips in batches, check what comes back, write the outputs under out_root (a fresh
    directory: refused if it exists)."""
    if out_root.exists():
        raise Refusal(f"the output dir {out_root} exists; every attempt writes to a fresh one")
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
    out_root.mkdir(parents=True)
    hashes = {}
    for r in results:
        for rel, data in r["files"].items():
            dst = out_root / rel
            if out_root.resolve() not in dst.resolve().parents:
                raise RuntimeError(f"output path {rel!r} leaves {out_root}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            hashes[rel] = sha256(data)
        stats = out_root / "stats.jsonl"
        with stats.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r["stats"], default=str) + "\n")
    return {"ingest": INGEST_MODULE + ".ingest_clip", "n_clips": len(results), "wall_s": round(time.time() - t0, 1),
            "out": str(out_root), "outputs_sha256": dict(sorted(hashes.items()))}


# ------------------------------------------------------------------------------------------------ commands

def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _run_double_module():
    scripts = str(REPO / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_double  # scripts/run_double.py (an orchestrator)

    return run_double


def double_ready() -> None:
    """Refused (BEFORE preflight, the drand fetch and the RUNLOG `started` record, so no attempt number is used up)
    unless run_double's sealed backend is wired (run_double.SEALED_BACKEND_READY)."""
    rd = _run_double_module()
    if not getattr(rd, "SEALED_BACKEND_READY", False):
        raise Refusal(f"the double stage's sealed backend ({getattr(rd, 'SEALED_BACKEND', '?')}) is not wired in "
                      "scripts/run_double.py: refusing before any RUNLOG record (no attempt used)")


def run_double_stage(repo: Path, attempt: int, seed_ints: list[int],
                     double_runner: Callable[[Path, list[int]], dict] | None = None) -> dict:
    """The sealed CPU-double stage: a fresh runs root per attempt, then run_double.run_sealed (16 test seeds)."""
    out = repo / DOUBLE_OUT / f"attempt_{attempt:03d}"
    if out.exists():
        raise Refusal(f"{out} exists; every attempt writes to a fresh one")
    if os.environ.get(SEALED_ENV) != "1":
        raise Refusal("the double stage runs only after every refusal passed (sealed flag not set)")
    if double_runner is None:
        double_runner = _run_double_module().run_sealed
    rec = double_runner(out, list(seed_ints))
    return {"out": str(out), "runs_root": rec.get("runs_root"), "invalid": rec.get("invalid"),
            "n_missions": rec.get("n_missions"), "usd": rec.get("usd"), "wall_s": rec.get("wall_s")}


def run(repo: Path, sealed_root: Path, stages: list[str], fetch: Callable[[str], dict] = drand_fetch,
        runner: Callable[[list[list[dict]]], list[dict]] | None = None,
        double_runner: Callable[[Path, list[int]], dict] | None = None) -> dict:
    """One sealed attempt: refusals, then a `started` RUNLOG record, the stages, and the closing record."""
    bad = sorted(set(stages) - set(STAGES))
    if bad or not stages:
        raise Refusal(f"stages {bad or stages} are not wired in sealed_run.py (known: {list(STAGES)})")
    if "double" in stages and double_runner is None:
        double_ready()
    rec = preflight(repo, sealed_root, fetch)
    known = rec.pop("runlog")
    jobs = rec.pop("jobs")
    attempt = 1 + sum(r["status"] == "started" for r in known)
    perc_rel = f"perc/attempt_{attempt:03d}"
    if (sealed_root / perc_rel).exists():
        raise Refusal(f"{sealed_root / perc_rel} exists but RUNLOG has {attempt - 1} attempts: refusing to reuse it")
    os.environ[SEALED_ENV] = "1"                     # only after every refusal passed
    for s in rec["seeds"]["seed_ints"]:
        check_world_seed(s)                          # the generator's own guard now admits the test seeds
    start = dict(rec, contract=RUNLOG_CONTRACT, attempt=attempt, status="started", timestamp=_now_iso(),
                 stages=list(stages), sealed_flag=SEALED_ENV + "=1", perc_dir=perc_rel)
    known = append_runlog(repo, known, start)
    results: dict[str, dict] = {}
    cur = ""
    try:
        if "clips" in stages:
            cur = "clips"
            results["clips"] = dict(run_clips(sealed_root, jobs, runner or modal_runner, sealed_root / perc_rel),
                                    status="done")
        if "double" in stages:
            cur = "double"
            results["double"] = dict(run_double_stage(repo, attempt, rec["seeds"]["seed_ints"], double_runner),
                                     status="done")
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised
        results[cur] = {"status": "error", "error": repr(e)}
        append_runlog(repo, known, {"attempt": attempt, "status": "error", "timestamp": _now_iso(),
                                    "head": rec["head"], "stages": results})
        raise
    known = append_runlog(repo, known, {"attempt": attempt, "status": "done", "timestamp": _now_iso(),
                                        "head": rec["head"], "stages": results})
    return {"attempt": attempt, "perc_dir": perc_rel, "stages": results, "records": known}


def derive(repo: Path, sealed_root: Path, fallback: bool, fetch: Callable[[str], dict] = drand_fetch,
           now: Callable[[], float] = time.time) -> dict:
    """Custodian, after FREEZE is committed: R, 16 seeds, a fresh salt -> <sealed_root>/seeds.json; prints the line
    to commit as prereg/TEST_SEED_COMMIT. Refuses unless the tree is clean and FREEZE names an ancestor commit, the
    FREEZE commit's dates agree, and (drand) it runs within MAX_DERIVE_LAG_S of the FREEZE commit."""
    check_clean(repo)
    sha = read_freeze(repo)
    check_ancestor(repo, sha)
    rp = sealed_root / SEEDS_FILE
    if rp.exists():
        raise Refusal(f"{rp} exists; seeds are derived once")
    t_freeze = check_commit_dates(repo)
    t_now = now()
    if fallback:
        rec = {"source": "fallback", "round": None, "randomness": secrets.token_hex(16)}
    else:
        info = fetch("/info")
        rnd = first_round_after(info, t_freeze)
        if t_now < round_time(info, rnd):
            raise Refusal(f"drand round {rnd} is published at {round_time(info, rnd)}; wait and re-run")
        check_derive_clock(info, rnd, t_freeze, t_now)
        b = fetch(f"/public/{rnd}")
        rec = {"source": "drand", "round": rnd, "randomness": b["randomness"]}
    seeds = derive_seeds(rec["randomness"])
    for sd in seeds:
        seed_int(sd)                                  # refuses a seed that would land in the dev range
    salt = secrets.token_hex(16)
    rec.update(salt=salt, seeds=seeds, freeze_sha=sha, freeze_commit_time=t_freeze, derived_at=int(t_now))
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return {"seed_commit": seed_commit(salt, seeds), "reveal": str(rp), "source": rec["source"], "round": rec["round"]}


def dev_dry_run(repo: Path, dev_root: Path, clip_ids: list[str], data_root: Path | None = None,
                runner: Callable[[list[list[dict]]], list[dict]] | None = None) -> dict:
    """The clips stage on DEV clips: a scratch dev root (never the sealed root) with the named dev clips and a dev
    MANIFEST (make_sealed), then the same jobs / upload / container path with split "dev". No sealed flag, no RUNLOG."""
    from twin.schemas.prereg import make_sealed, render_sealed

    root = dev_root.resolve()
    sealed = DEFAULT_SEALED_ROOT.resolve()
    if root == sealed or sealed in root.parents or root in sealed.parents:
        raise Refusal(f"the dev root {dev_root} must be outside the sealed root {DEFAULT_SEALED_ROOT}")
    if not clip_ids:
        raise Refusal("dev-dry-run needs --clips (dev clip ids)")
    split = _split_rows(repo)
    bad = [c for c in clip_ids if (split.get(c) or {}).get("split") != "dev"]
    if bad:
        raise Refusal(f"dev-dry-run takes dev clips only; not dev rows of the active split: {bad}")
    if data_root is None:
        env = os.environ.get("INTERCYCLE_DATA")
        if not env:
            raise Refusal("set INTERCYCLE_DATA (the dev clips are read from $INTERCYCLE_DATA/clips)")
        data_root = Path(env)
    files = []
    for cid in clip_ids:
        src = data_root / "clips" / f"{cid}.mp4"
        if not src.is_file():
            raise Refusal(f"dev clip {src} does not exist")
        data = src.read_bytes()
        dst = root / "clips" / f"{cid}.mp4"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        files.append((sha256(data), f"clips/{cid}.mp4"))
    man = make_sealed("MANIFEST", files, list(clip_ids))
    (root / "MANIFEST.dev.sha256").write_text(render_sealed(man), encoding="utf-8", newline="\n")
    man = parse_sealed((root / "MANIFEST.dev.sha256").read_text(encoding="utf-8"))
    jobs = check_manifest_files(repo, root, man, split="dev")
    out = root / "perc" / f"dry_{time.strftime('%Y%m%d_%H%M%S')}"
    res = run_clips(root, jobs, runner or modal_runner, out)
    rec = {"dev_dry_run": True, "timestamp": _now_iso(), "head": git(repo, "rev-parse", "HEAD").stdout.decode().strip(),
           "clips": list(clip_ids), "manifest_aggregate": man.aggregate, "clips_stage": res}
    (root / "dev_dry_run.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the sealed test run (PLAN_TWIN §5); custodian only")
    ap.add_argument("command", choices=("check", "derive", "run", "dev-dry-run"))
    ap.add_argument("--sealed-root", type=Path, default=DEFAULT_SEALED_ROOT)
    ap.add_argument("--stages", default="clips")
    ap.add_argument("--fallback", action="store_true", help="derive: drand unreachable, R = secrets.token_hex(16)")
    ap.add_argument("--dev-root", type=Path, default=None, help="dev-dry-run: scratch dir (never the sealed root)")
    ap.add_argument("--clips", default="", help="dev-dry-run: comma-separated DEV clip ids")
    a = ap.parse_args(argv)
    try:
        if a.command == "check":
            rec = preflight(REPO, a.sealed_root)
            rec.pop("jobs")
            print(json.dumps({k: rec[k] for k in ("head", "freeze_sha", "inputs", "n_clips")}, indent=1))
            print(f"sealed_run: every refusal passed ({len(rec['runlog'])} RUNLOG records so far)")
        elif a.command == "derive":
            d = derive(REPO, a.sealed_root, a.fallback)
            print(f"write this line to {SEED_COMMIT} and commit only that file:\n{d['seed_commit']}")
            print(f"reveal kept at {d['reveal']} (source {d['source']}, round {d['round']})")
        elif a.command == "dev-dry-run":
            if a.dev_root is None:
                raise Refusal("dev-dry-run needs --dev-root")
            d = dev_dry_run(REPO, a.dev_root, [c for c in a.clips.split(",") if c])
            print(f"sealed_run dev-dry-run: {d['clips_stage']['n_clips']} dev clips -> {d['clips_stage']['out']}")
        else:
            r = run(REPO, a.sealed_root, [s for s in a.stages.split(",") if s])
            print(f"sealed_run: attempt {r['attempt']} done; {RUNLOG} appended ({', '.join(r['stages'])})")
    except Refusal as e:
        print(f"sealed_run REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
