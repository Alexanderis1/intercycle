"""Custodian sealing of the TEST sample list and the TEST labels (PLAN_TWIN §5 "Sealed-test enforcement", U14).

Custodian session only. Reads the sealed store (default C:\\intercycle_sealed) and writes hash files into the repo's
sealed/ folder; it prints counts and hashes only (no clip id, no label content).

* `sample`: BEFORE any labelling. Checks C:\\intercycle_sealed\\labels\\sample.json (a valid `twin.label_samples/1`
  list of split test whose bytes match its sample.sha256, every item a test row of the active split) and writes
  sealed/SAMPLE.sha256, one line:

      <items_sha256>  sample.json#items frames=<n> stroke_windows=<n> file_sha256=<sha256 of the file bytes>

* `labels`: AFTER labelling, before FREEZE. Every other *.json in C:\\intercycle_sealed\\labels\\ is a
  `twin.labels/1` LabelSet of split test (blind, no proposals) answering that sample list, with `rules_sha256` = the
  sha256 of labels/TEST_RULES.md; together they label every sample item exactly once, with ids, clips and times of
  the sample list. Writes sealed/LABELS.sha256 (twin/schemas/prereg.py format, kind LABELS) with make_sealed /
  render_sealed: one file line per label file plus the sample list (paths `labels/<name>`), the sampled clip ids,
  `#sample_list` = the sample list's items_sha256; then checks that parse_sealed reads it back unchanged.

    python scripts/custodian_seal.py sample
    python scripts/custodian_seal.py labels

An existing hash file is never overwritten with different content (re-running with the same inputs is a no-op).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twin.schemas.labels import LabelSet, SampleList, check_complete  # noqa: E402
from twin.schemas.prereg import load_active_split, make_sealed, parse_sealed, render_sealed  # noqa: E402

SEALED_ROOT = Path(r"C:\intercycle_sealed")
SAMPLE_NAME = "sample.json"
TEST_RULES = ROOT / "labels" / "TEST_RULES.md"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_sample(labels_dir: Path, test_ids: set[str]) -> tuple[SampleList, bytes]:
    """The sealed sample list, checked: valid, split test, bytes = its .sha256 file, every item a test clip."""
    path = labels_dir / SAMPLE_NAME
    data = path.read_bytes()
    sha_file = path.with_suffix(".sha256")
    if not sha_file.is_file():
        raise SystemExit(f"refusing: no {sha_file.name} next to {path.name}")
    want = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in sha_file.read_text(encoding="utf-8").splitlines()
            if "  " in ln}
    sl = SampleList.model_validate(json.loads(data.decode("utf-8")))
    if want.get(path.name) != sha256_bytes(data) or want.get(f"{path.name}#items (twin.labels sample_items_sha256)") \
            != sl.items_sha256:
        raise SystemExit(f"refusing: {path.name} does not match {sha_file.name}")
    if sl.split != "test":
        raise SystemExit(f"refusing: sample list split is {sl.split!r}, not test")
    bad = [i.sample_id for i in sl.items if i.split != "test" or i.clip_id not in test_ids]
    if bad:
        raise SystemExit(f"refusing: {len(bad)} sample items are not test clips of the active split")
    return sl, data


def sample_line(sl: SampleList, data: bytes) -> str:
    nf = sum(i.kind == "frame" for i in sl.items)
    nw = sum(i.kind == "stroke_window" for i in sl.items)
    return (f"{sl.items_sha256}  {SAMPLE_NAME}#items frames={nf} stroke_windows={nw} "
            f"file_sha256={sha256_bytes(data)}\n")


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_bytes() == text.encode("utf-8"):
            return
        raise SystemExit(f"refusing to overwrite {path} with different content")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def cmd_sample(labels_dir: Path, out: Path, test_ids: set[str]) -> str:
    if not out.exists() and any(p.name != SAMPLE_NAME for p in labels_dir.glob("*.json")):
        raise SystemExit("refusing: label files already exist; SAMPLE.sha256 is written BEFORE labelling starts")
    sl, data = load_sample(labels_dir, test_ids)
    line = sample_line(sl, data)
    write_once(out, line)
    return line


def check_labels(labels_dir: Path, test_ids: set[str], rules_sha256: str) -> tuple[SampleList, list[tuple[str, str]],
                                                                                     dict[str, int]]:
    """Validate every label file; returns the sample list, the (sha256, path) file lines (label files + sample list)
    and counts. Raises SystemExit / ValueError on any problem."""
    sl, sdata = load_sample(labels_dir, test_ids)
    files: list[tuple[str, str]] = [(sha256_bytes(sdata), f"labels/{SAMPLE_NAME}")]
    seen: dict[str, str] = {}
    n_frames = n_windows = n_sets = 0
    names = sorted(p.name for p in labels_dir.glob("*.json") if p.name != SAMPLE_NAME)
    if not names:
        raise SystemExit("refusing: no label file in the sealed labels folder")
    for name in names:
        data = (labels_dir / name).read_bytes()
        try:
            ls = LabelSet.model_validate(json.loads(data.decode("utf-8")))
        except ValueError as e:
            raise SystemExit(f"refusing {name}: not a valid twin.labels/1 set ({type(e).__name__}: "
                             f"{str(e).splitlines()[0]})") from None
        if ls.split != "test" or not ls.blind or ls.proposals_shown:
            raise SystemExit(f"refusing {name}: test labels must be split test, blind, with no proposals")
        if ls.rules_sha256 != rules_sha256:
            raise SystemExit(f"refusing {name}: rules_sha256 is not the sha256 of labels/TEST_RULES.md")
        try:
            check_complete(ls, sl, require_all=False)
        except ValueError as e:
            raise SystemExit(f"refusing {name}: {e}") from None
        for sid in [f.sample_id for f in ls.frames] + [w.sample_id for w in ls.stroke_windows]:
            if sid in seen:
                raise SystemExit(f"refusing: a sample item is labelled twice ({seen[sid]} and {name})")
            seen[sid] = name
        n_frames += len(ls.frames)
        n_windows += len(ls.stroke_windows)
        n_sets += 1
        files.append((sha256_bytes(data), f"labels/{name}"))
    missing = [i.sample_id for i in sl.items if i.sample_id not in seen]
    if missing:
        raise SystemExit(f"refusing: {len(missing)} sample items are not labelled (every item, empty frames and "
                         f"reserve windows included)")
    return sl, files, {"label_files": n_sets, "frames": n_frames, "stroke_windows": n_windows}


def cmd_labels(labels_dir: Path, out: Path, test_ids: set[str], rules_sha256: str) -> tuple[str, dict[str, int]]:
    sl, files, counts = check_labels(labels_dir, test_ids, rules_sha256)
    ids = sorted({i.clip_id for i in sl.items})
    sealed = make_sealed("LABELS", files, ids, sample_list_sha256=sl.items_sha256)
    text = render_sealed(sealed)
    back = parse_sealed(text)
    if back != sealed or render_sealed(back) != text:
        raise SystemExit("LABELS.sha256 does not read back unchanged (parse_sealed)")
    write_once(out, text)
    if parse_sealed(out.read_text(encoding="utf-8")) != sealed:
        raise SystemExit(f"{out} does not parse back to the sealed hashes")
    return text, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("cmd", choices=["sample", "labels"])
    ap.add_argument("--sealed-root", default=str(SEALED_ROOT), help="sealed store root (labels/ inside it)")
    ap.add_argument("--out-dir", default=str(ROOT / "sealed"), help="where SAMPLE.sha256 / LABELS.sha256 go")
    a = ap.parse_args(argv)
    labels_dir = Path(a.sealed_root) / "labels"
    test_ids = {k for k, r in load_active_split().items() if r.split == "test"}
    out_dir = Path(a.out_dir)
    if a.cmd == "sample":
        line = cmd_sample(labels_dir, out_dir / "SAMPLE.sha256", test_ids)
        print(f"wrote {out_dir / 'SAMPLE.sha256'}: {line.strip()}")
        return 0
    rules = sha256_bytes(TEST_RULES.read_bytes())
    text, counts = cmd_labels(labels_dir, out_dir / "LABELS.sha256", test_ids, rules)
    agg = text.strip().splitlines()[-1]
    print(f"wrote {out_dir / 'LABELS.sha256'}: {counts['label_files']} label files, {counts['frames']} frames, "
          f"{counts['stroke_windows']} stroke windows; {agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
