"""Re-derive the `stroke` column of stored PERC machines tables with the current detect_strokes (BDC emission).

    python scripts/perc_restroke.py --clips 0_dv1,3_dv2 [--data $INTERCYCLE_DATA] [--backup-dir D]

The stroke column of a machines table is a pure function of its own (t, flow_v, missing) columns:
`press_motion.ingest_clip` sets flow_v NaN on MISSING samples, then stroke = detect_strokes(t, flow_v) & ~missing with
the default PressParams (no caller passes others). So re-deriving it from the stored table gives exactly what a full
re-ingest with the new detect_strokes would write, without decoding the clip again. Only `stroke` is rewritten; every
other column is checked unchanged. DEV clips only (the active split is checked); the old file is copied to
--backup-dir first.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from intercycle.perception import press_motion as pm  # noqa: E402
from twin.schemas.prereg import active_split_path  # noqa: E402


def restroke(df: pd.DataFrame) -> pd.DataFrame:
    """The table with `stroke` re-derived from (t, flow_v, missing) by detect_strokes (default PressParams)."""
    t = df["t"].to_numpy(float)
    miss = df["missing"].to_numpy(bool)
    v = df["flow_v"].to_numpy(float).copy()
    v[miss] = np.nan
    idx, _ = pm.detect_strokes(t, v, pm.PressParams())
    st = np.zeros(len(df), bool)
    st[idx] = True
    st &= ~miss
    out = df.copy()
    out["stroke"] = st
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--clips", required=True)
    ap.add_argument("--data", default=os.environ.get("INTERCYCLE_DATA", ""))
    ap.add_argument("--backup-dir", required=True)
    a = ap.parse_args(argv)
    data, bak = Path(a.data), Path(a.backup_dir)
    rows = {r["clip_id"]: r for r in csv.DictReader(io.StringIO(active_split_path(REPO).read_bytes().decode("utf-8")))}
    clips = [c for c in a.clips.split(",") if c]
    bad = [c for c in clips if (rows.get(c) or {}).get("split") != "dev"]
    if bad:
        print(f"perc_restroke REFUSED: not dev clips: {bad}", file=sys.stderr)
        return 2
    report = []
    for c in clips:
        for f in sorted((data / "machines").glob(f"*/{c}.parquet")):
            df = pd.read_parquet(f)
            new = restroke(df)
            same = new.drop(columns="stroke").equals(df.drop(columns="stroke"))
            if not same:
                raise RuntimeError(f"{f}: a column other than stroke changed")
            dst = bak / f.parent.name / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(f, dst)
            new.to_parquet(f, index=False)
            report.append({"press": f.parent.name, "clip": c, "n_old": int(df["stroke"].to_numpy(bool).sum()),
                           "n_new": int(new["stroke"].to_numpy(bool).sum())})
    print(json.dumps({"n_files": len(report), "n_old": sum(r["n_old"] for r in report),
                      "n_new": sum(r["n_new"] for r in report), "files": report}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
