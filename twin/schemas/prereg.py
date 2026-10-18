"""Formats of the pre-registered split and of the sealed hash files (PLAN_TWIN §2, §3 Split, §5 Sealed-test enforcement).

* `prereg/split.csv` (S3a, run once) and amended splits such as `prereg/split_a1.csv`: CSV with the header
  `clip_id,group_id,split,sha256` (these four columns first and in this order; an amended split may append columns
  after them). One row per clip, clip ids unique, split in train/dev/test, sha256 64 lowercase hex.
* `prereg/ACTIVE_SPLIT` (optional): names the split in force. Its first line that is neither empty nor a `#` comment
  is the file name of the split CSV inside prereg/ (e.g. `split_a1.csv`; no directory part), the same rule as
  `intercycle.esk.active_split_path` (amendment A1+). Without it the active split is `prereg/split.csv`.
  `active_split_path` fails closed when the entry is not a bare `.csv` name or the named file is missing.
* `sealed/MANIFEST.sha256` and `sealed/LABELS.sha256` (custodian): `SealedHashes`, one text format for both:

      # twin.sealed/1 MANIFEST                       (or LABELS; must be the first line)
      <sha256>  <path>                               one line per file, two spaces, sorted by path, unique paths
      #ids <id>,<id>,...                             the sorted, unique test clip ids covered
      #count camera=<9|14> class=<0..7> n=<k>        one line per (camera, class) of those ids, sorted
      #sample_list <sha256>                          LABELS only: sha256 of the hashed sample list (labels.SampleList)
      #aggregate <sha256>                            sha256 of the UTF-8 text of all file lines ("<sha>  <path>\\n")

  MANIFEST paths are `clips/<clip_id>.<ext>` and their ids must equal `#ids`. The file lines keep the `sha256sum`
  layout so `sha256sum -c` can check them (it warns on the `#` lines). `parse_sealed` and `render_sealed` are the only
  parser and writer; sealed_run.py and the custodian both use them.
* `llm/window_edits.jsonl` (S3b, Gemma minimum): one `WindowEdit` per line; an edit may only shorten a window, and
  the edited window must lie inside the original (the clamp).
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from twin.schemas.common import Split, Strict, unique
from twin.schemas.tables import CAMERA_OF_CLASS

REPO = Path(__file__).resolve().parents[2]
SPLIT_COLUMNS = ("clip_id", "group_id", "split", "sha256")
SHA_RE = r"^[0-9a-f]{64}$"
CLIP_ID_RE = re.compile(r"^([0-7])_[A-Za-z0-9]+$")       # class prefix, then the authors' clip name (e.g. 0_te1)


# ------------------------------------------------------------------------------------------------ split


class SplitRow(Strict):
    clip_id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    split: Split
    sha256: str = Field(pattern=SHA_RE)


def parse_split(text: str) -> dict[str, SplitRow]:
    """Parse a split CSV; raises ValueError on a bad header, a bad row or a duplicate clip id."""
    rd = csv.reader(io.StringIO(text))
    header = next(rd, None)
    if header is None or tuple(header[: len(SPLIT_COLUMNS)]) != SPLIT_COLUMNS:
        raise ValueError(f"split header {header} must start with {list(SPLIT_COLUMNS)}")
    out: dict[str, SplitRow] = {}
    for i, row in enumerate(rd, start=2):
        if not row:
            continue
        if len(row) != len(header):
            raise ValueError(f"split line {i}: {len(row)} fields, header has {len(header)}")
        r = SplitRow.model_validate(dict(zip(SPLIT_COLUMNS, row[: len(SPLIT_COLUMNS)])))
        if r.clip_id in out:
            raise ValueError(f"split line {i}: duplicate clip id {r.clip_id}")
        out[r.clip_id] = r
    if not out:
        raise ValueError("split has no rows")
    return out


def active_split_path(repo: Path = REPO) -> Path:
    """The split in force: prereg/<name> named by prereg/ACTIVE_SPLIT, else prereg/split.csv. Fails closed if the
    entry is not a bare .csv file name or the file is missing."""
    pointer = repo / "prereg" / "ACTIVE_SPLIT"
    if pointer.exists():
        lines = [ln.strip() for ln in pointer.read_text(encoding="utf-8").splitlines()]
        named = [ln for ln in lines if ln and not ln.startswith("#")]
        if not named:
            raise ValueError(f"{pointer} names no split file")
        name = named[0]
        if "/" in name or "\\" in name or not name.endswith(".csv"):
            raise ValueError(f"bad prereg/ACTIVE_SPLIT entry {name!r}: a .csv file name inside prereg/")
        path = repo / "prereg" / name
    else:
        path = repo / "prereg" / "split.csv"
    if not path.is_file():
        raise FileNotFoundError(f"active split {path} does not exist")
    return path


def load_active_split(repo: Path = REPO) -> dict[str, SplitRow]:
    return parse_split(active_split_path(repo).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ sealed hashes


class FileHash(Strict):
    sha256: str = Field(pattern=SHA_RE)
    path: str = Field(min_length=1, pattern=r"^[^\s#][^\t\n]*$")


class ClassCount(Strict):
    camera: Literal[9, 14]
    cls: int = Field(ge=0, le=7, alias="class")
    n: int = Field(gt=0)


def camera_class_of(clip_id: str) -> tuple[int, int]:
    m = CLIP_ID_RE.match(clip_id)
    if not m:
        raise ValueError(f"clip id {clip_id!r} does not start with a class prefix 0-7 (e.g. 0_te1)")
    c = int(m.group(1))
    return CAMERA_OF_CLASS[c], c


def aggregate_sha256(files: list[FileHash]) -> str:
    return hashlib.sha256("".join(f"{f.sha256}  {f.path}\n" for f in files).encode("utf-8")).hexdigest()


class SealedHashes(Strict):
    kind: Literal["MANIFEST", "LABELS"]
    files: list[FileHash] = Field(min_length=1)
    ids: list[str] = Field(min_length=1)
    counts: list[ClassCount] = Field(min_length=1)
    sample_list_sha256: str | None = Field(default=None, pattern=SHA_RE)
    aggregate: str = Field(pattern=SHA_RE)

    @model_validator(mode="after")
    def _consistent(self):
        paths = [f.path for f in self.files]
        unique(paths, "sealed file path")
        if paths != sorted(paths):
            raise ValueError("sealed file lines must be sorted by path")
        unique(self.ids, "sealed clip")
        if self.ids != sorted(self.ids):
            raise ValueError("#ids must be sorted")
        derived = Counter(camera_class_of(i) for i in self.ids)
        want = [ClassCount.model_validate({"camera": cam, "class": c, "n": n}) for (cam, c), n in sorted(derived.items())]
        if self.counts != want:
            raise ValueError(f"#count lines {[(c.camera, c.cls, c.n) for c in self.counts]} != counts of #ids "
                             f"{[(c.camera, c.cls, c.n) for c in want]}")
        if self.aggregate != aggregate_sha256(self.files):
            raise ValueError("#aggregate does not match the file lines")
        if self.kind == "MANIFEST":
            if self.sample_list_sha256 is not None:
                raise ValueError("MANIFEST has no #sample_list")
            stems = []
            for p in paths:
                if not p.startswith("clips/") or "/" in p[len("clips/"):]:
                    raise ValueError(f"MANIFEST path {p} must be clips/<clip_id>.<ext>")
                stems.append(Path(p).stem)
            if sorted(stems) != self.ids:
                raise ValueError("MANIFEST clip files and #ids differ")
        elif self.sample_list_sha256 is None:
            raise ValueError("LABELS needs #sample_list")
        return self


def render_sealed(s: SealedHashes) -> str:
    lines = [f"# twin.sealed/1 {s.kind}"]
    lines += [f"{f.sha256}  {f.path}" for f in s.files]
    lines.append("#ids " + ",".join(s.ids))
    lines += [f"#count camera={c.camera} class={c.cls} n={c.n}" for c in s.counts]
    if s.sample_list_sha256 is not None:
        lines.append(f"#sample_list {s.sample_list_sha256}")
    lines.append(f"#aggregate {s.aggregate}")
    return "\n".join(lines) + "\n"


_COUNT_RE = re.compile(r"^#count camera=(\d+) class=(\d+) n=(\d+)$")
_FILE_RE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


def parse_sealed(text: str) -> SealedHashes:
    lines = [ln.rstrip("\r") for ln in text.split("\n")]
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        raise ValueError("empty sealed hash file")
    m = re.match(r"^# twin\.sealed/1 (MANIFEST|LABELS)$", lines[0])
    if not m:
        raise ValueError(f"first line {lines[0]!r} must be '# twin.sealed/1 MANIFEST' or '... LABELS'")
    files, ids, counts = [], None, []
    sample, agg = None, None
    for ln in lines[1:]:
        if fm := _FILE_RE.match(ln):
            files.append({"sha256": fm.group(1), "path": fm.group(2)})
        elif ln.startswith("#ids "):
            if ids is not None:
                raise ValueError("more than one #ids line")
            ids = ln[len("#ids "):].split(",")
        elif cm := _COUNT_RE.match(ln):
            counts.append({"camera": int(cm.group(1)), "class": int(cm.group(2)), "n": int(cm.group(3))})
        elif ln.startswith("#sample_list "):
            sample = ln.split(" ", 1)[1]
        elif ln.startswith("#aggregate "):
            agg = ln.split(" ", 1)[1]
        else:
            raise ValueError(f"unrecognised sealed line {ln!r}")
    return SealedHashes.model_validate({"kind": m.group(1), "files": files, "ids": ids or [], "counts": counts,
                                        "sample_list_sha256": sample, "aggregate": agg})


def make_sealed(kind: Literal["MANIFEST", "LABELS"], files: list[tuple[str, str]], ids: list[str],
                sample_list_sha256: str | None = None) -> SealedHashes:
    """Build a SealedHashes from (sha256, path) pairs and clip ids (sorting and counts are derived)."""
    fh = sorted((FileHash(sha256=s, path=p) for s, p in files), key=lambda f: f.path)
    sids = sorted(ids)
    cnt = Counter(camera_class_of(i) for i in sids)
    return SealedHashes.model_validate({
        "kind": kind, "files": [f.model_dump() for f in fh], "ids": sids,
        "counts": [{"camera": cam, "class": c, "n": n} for (cam, c), n in sorted(cnt.items())],
        "sample_list_sha256": sample_list_sha256, "aggregate": aggregate_sha256(fh)})


# ------------------------------------------------------------------------------------------------ Gemma window edits


class WindowEdit(Strict):
    """One line of llm/window_edits.jsonl: Gemma may only shorten an inspection window, never move or widen it."""
    clip_id: str = Field(min_length=1)
    press: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    t0: float
    t1: float
    t0_new: float
    t1_new: float
    mode: Literal["record", "replay"]
    model: str = Field(min_length=1)
    reason: str | None = None

    @model_validator(mode="after")
    def _clamp(self):
        if self.t1 <= self.t0:
            raise ValueError(f"window {self.window_id}: empty original window")
        if not (self.t0 <= self.t0_new < self.t1_new <= self.t1):
            raise ValueError(f"window {self.window_id}: edit [{self.t0_new}, {self.t1_new}] must lie inside "
                             f"[{self.t0}, {self.t1}] and stay non-empty (shorten only)")
        return self
