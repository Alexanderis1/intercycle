"""Write the 6-frame SYNTHETIC fixture for the Gemma verify_state harness.

    python scripts/gemma_make_fixture.py [out_dir]      (default out/gemma_fixture)

Creates <out_dir>/frames/frame_00..05.png and <out_dir>/labels.json
({file: {machine, moving, person_in_zone}}). Later, point gemma_eval_verify.py at frames rendered by the
simulator with the same labels.json layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.llm.fixtures import make_fixture  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "out" / "gemma_fixture"
    labels = make_fixture(out)
    print(f"wrote {len(labels)} synthetic frames + labels.json to {out}")
