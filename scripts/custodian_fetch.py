"""CUSTODIAN ONLY (laptop session). Download exactly the A1+ TEST clips and write sealed/MANIFEST.sha256.

    python scripts/custodian_fetch.py                       # download + verify + write the manifest
    python scripts/custodian_fetch.py --verify-only         # re-hash the local files against the split and manifest

Python 3.11+ (the repo's floor) with pydantic, which is required (the repo's own contract parser,
twin.schemas.prereg), no HF token. It reads the TEST rows
(clip_id, sha256) of the active split (prereg/ACTIVE_SPLIT, the one sealed_run.py checks against), downloads each clip from the pinned HF revision into C:\\intercycle_sealed\\clips\\<id>.mp4, checks
its sha256, and refuses to finish if any file is missing, extra or wrong. It never decodes or opens a video.

sealed/MANIFEST.sha256 (PLAN_TWIN §2: the sha256 per test file, the sorted ids, counts per camera and class) is written
in the one contract format, twin.sealed/1 (`twin.schemas.prereg.make_sealed` + `render_sealed`), and re-parsed with
`parse_sealed` before it is written, so scripts/sealed_run.py accepts it:
    # twin.sealed/1 MANIFEST
    <sha256>  clips/<clip_id>.mp4          one line per test clip, sorted by path
    #ids <id1>,<id2>,...                  sorted
    #count camera=<cam> class=<k> n=<n>   per (camera, class prefix): 0,1,2,4,5,6 -> 9; 3,7 -> 14
    #aggregate <sha256>                   of the file lines
The contract has no line for the split file's hash: it is printed (LF-normalised) and the MANIFEST ids and hashes are
checked against the active split's test rows by sealed_run.py instead.
The label hash (sealed/LABELS.sha256) is written later, by the test labeller step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ID = "Voxel51/Safe_and_Unsafe_Behaviours"
REV = "7369ff1830b5f6c00f7d7383eeeaa90704bfc5b5"
CAMERA_OF_PREFIX = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from twin.schemas.prereg import active_split_path, make_sealed, parse_sealed, render_sealed  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(cid: str, dst: Path) -> None:
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REV}/data/{cid}.mp4"
    tmp = dst.with_suffix(".part")
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            tmp.replace(dst)
            return
        except Exception as e:  # 429 / transient network errors: back off
            if attempt == 7:
                raise RuntimeError(f"download failed for {cid}: {e}") from e
            time.sleep(min(60, 2 ** attempt))


def test_rows(split_csv: Path) -> list[tuple[str, str]]:
    with split_csv.open(newline="", encoding="utf-8") as fh:
        rows = [(r["clip_id"], r["sha256"]) for r in csv.DictReader(fh) if r["split"] == "test"]
    return sorted(rows)


def manifest_text(rows: list[tuple[str, str]]) -> str:
    """The twin.sealed/1 MANIFEST of the test rows (clip_id, sha256); re-parsed before it is returned (fail closed)."""
    sealed = make_sealed("MANIFEST", [(s, f"clips/{c}.mp4") for c, s in rows], [c for c, _ in rows])
    text = render_sealed(sealed)
    if parse_sealed(text) != sealed:
        raise RuntimeError("the rendered MANIFEST does not parse back to itself")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="custodian: fetch the sealed TEST clips")
    ap.add_argument("--split", type=Path, default=None, help="default: the active split (prereg/ACTIVE_SPLIT)")
    ap.add_argument("--dest", type=Path, default=Path(r"C:\intercycle_sealed"))
    ap.add_argument("--manifest", type=Path, default=ROOT / "sealed" / "MANIFEST.sha256")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    a.split = a.split or active_split_path(ROOT)
    rows = test_rows(a.split)
    if not rows:
        sys.exit("no test rows in the split")
    clips = a.dest / "clips"
    if a.verify_only:
        if not clips.is_dir():
            sys.exit(f"FAILED: {clips} does not exist")
    else:
        clips.mkdir(parents=True, exist_ok=True)
    bad = []
    for k, (cid, sha) in enumerate(rows, 1):
        p = clips / f"{cid}.mp4"
        if not a.verify_only and not (p.exists() and sha256_file(p) == sha):
            download(cid, p)
        if not p.exists() or sha256_file(p) != sha:
            bad.append(cid)
        print(f"[{k}/{len(rows)}] {cid} {'ok' if cid not in bad else 'BAD'}", flush=True)
    extra = sorted(f.name for f in clips.iterdir() if f.is_file() and f.name not in {f"{c}.mp4" for c, _ in rows})
    if bad or extra:
        sys.exit(f"FAILED: {len(bad)} missing/wrong files {bad[:10]}, {len(extra)} extra files {extra[:10]}")
    # hash of the split file with LF line endings, so a CRLF checkout (core.autocrlf) gives the same value
    split_sha = hashlib.sha256(a.split.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    print(f"split {a.split.name} sha256 {split_sha}")
    text = manifest_text(rows)
    if a.verify_only:
        if not a.manifest.exists() or a.manifest.read_bytes().replace(b"\r\n", b"\n").decode("utf-8") != text:
            sys.exit(f"FAILED: {a.manifest} is missing or differs from the files")
        print(f"verified {len(rows)} files against {a.manifest}")
        return
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(rows)} test clips verified in {clips}; wrote {a.manifest} (sha256 {sha256_file(a.manifest)})")


if __name__ == "__main__":
    main()
