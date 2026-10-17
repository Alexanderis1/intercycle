"""Loader for the press-shop clip inventory (PLAN_TWIN §2 contract, owner S3a).

``$INTERCYCLE_DATA/inventory.parquet`` holds one row per clip with the columns in ``COLUMNS``. The TEST split is sealed:
``load(split="test")`` (or any request that includes test rows) raises unless ``INTERCYCLE_SEALED_RUN=1``.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_ENV = "INTERCYCLE_DATA"
SEALED_ENV = "INTERCYCLE_SEALED_RUN"
COLUMNS = ["clip_id", "camera", "framing_id", "group_id", "class", "duration", "sha256", "split"]
# extra columns (amendment A1+): dup_of = the other clip ids with a byte-identical file (comma-separated, "" if none);
# dup_classes = the distinct class prefixes of that byte-identical set when they disagree ("" otherwise)
EXTRA_COLUMNS = ["dup_of", "dup_classes"]
REPO_ROOT = Path(__file__).resolve().parents[1]


def active_split_path(repo_root: str | Path | None = None) -> Path:
    """The split in force: prereg/<name> where name is the single line of prereg/ACTIVE_SPLIT (amendment A1+).

    prereg/split.csv stays the original pre-registered record; without ACTIVE_SPLIT it is the active split.
    """
    pre = Path(repo_root or REPO_ROOT) / "prereg"
    ptr = pre / "ACTIVE_SPLIT"
    name = ptr.read_text(encoding="utf-8").strip() if ptr.exists() else "split.csv"
    if "/" in name or "\\" in name or not name.endswith(".csv"):
        raise ValueError(f"bad prereg/ACTIVE_SPLIT entry {name!r}")
    p = pre / name
    if not p.exists():
        raise FileNotFoundError(p)
    return p
SPLITS = ("train", "dev", "test")
CAMERA_OF_PREFIX = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}


class SealedSplitError(PermissionError):
    """Raised when TEST rows are requested outside a sealed run."""


def data_root(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get(DATA_ENV)
    if not env:
        raise RuntimeError(f"set {DATA_ENV} to the press-shop data root (e.g. F:\\intercycle\\data\\esk)")
    return Path(env)


def camera_of(clip_id: str) -> int:
    """Camera (9 or 14) from the class prefix of the clip id (exact, PLAN_TWIN §2)."""
    return CAMERA_OF_PREFIX[int(clip_id.split("_")[0])]


def _sealed() -> bool:
    return os.environ.get(SEALED_ENV) == "1"


def load(split: str | Iterable[str], root: str | Path | None = None) -> pd.DataFrame:
    """Rows of inventory.parquet for ``split`` ("train", "dev", "test" or a list of them)."""
    want = [split] if isinstance(split, str) else list(split)
    bad = [s for s in want if s not in SPLITS]
    if bad:
        raise ValueError(f"unknown split(s) {bad}; use {SPLITS}")
    if "test" in want and not _sealed():
        raise SealedSplitError(f"the test split is sealed: set {SEALED_ENV}=1 only inside scripts/sealed_run.py")
    inv = pd.read_parquet(data_root(root) / "inventory.parquet")
    missing = [c for c in COLUMNS if c not in inv.columns]
    if missing:
        raise ValueError(f"inventory.parquet lacks columns {missing}")
    out = inv.loc[inv["split"].isin(want)].reset_index(drop=True)
    assert isinstance(out, pd.DataFrame)
    return out


def clip_path(clip_id: str, root: str | Path | None = None) -> Path:
    """Local path of a train/dev clip copied from the Modal volume (test clips are never stored here)."""
    return data_root(root) / "clips" / f"{clip_id}.mp4"


def check_against_split(inv: pd.DataFrame, split_csv: str | Path | None = None) -> None:
    """Raise unless inventory rows match the split csv exactly on clip_id, group_id, split and sha256 (and framing_id
    when the csv has that column, as split_a1.csv does). Default: the active split (``active_split_path``)."""
    split_csv = split_csv or active_split_path()
    with open(split_csv, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        cols = ["group_id", "split", "sha256"] + (["framing_id"] if "framing_id" in (rd.fieldnames or []) else [])
        ref = {r["clip_id"]: tuple(r[c] for c in cols) for r in rd}
    got = {str(row[0]): tuple(str(v) for v in row[1:]) for row in
           zip(inv["clip_id"], *(inv[c] for c in cols))}
    if got != ref:
        only_inv = sorted(set(got) - set(ref))[:5]
        only_ref = sorted(set(ref) - set(got))[:5]
        diff = sorted(k for k in set(got) & set(ref) if got[k] != ref[k])[:5]
        raise AssertionError(f"inventory != split.csv: only_inv={only_inv} only_split={only_ref} differ={diff}")
