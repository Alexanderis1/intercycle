"""Container side of scripts/sealed_run.py's clips stage (and of its --dev-dry-run).

A plain module (stdlib-only imports at top level) so Modal can IMPORT it in the container by name: sealed_run ships it
with ``Image.add_local_python_source("sealed_ingest")`` on cloud/modal_perc.py's image and registers
``container_ingest`` WITHOUT ``serialized=True`` (and with ``include_source=False``), so no cloudpickled code crosses
from the client Python to the image's Python (they differ: the image is the dev PERC image). Only plain data crosses
(lists / dicts of str, int and bytes).

Per clip it re-hashes the uploaded bytes against the MANIFEST sha256, then runs the SAME per-clip ingest as the dev
batch, ``intercycle.perception.esk_ingest.ingest_clip`` (imported by name), with the dev batch's detector settings.
"""
from __future__ import annotations

import hashlib
import importlib
import tempfile
from pathlib import Path

INGEST_MODULE = "intercycle.perception.esk_ingest"
ROIS = "/root/rois.json"                    # cloud/modal_perc.py adds the press ROIs here
SPLITS = ("test", "dev")                    # test: the sealed run; dev: --dev-dry-run


def check_batch(batch: list[dict]) -> str:
    """Every job of a batch names the same split, one of SPLITS, and carries its bytes and sha256."""
    splits = {j.get("split") for j in batch}
    if len(splits) != 1 or next(iter(splits)) not in SPLITS:
        raise PermissionError(f"a batch must hold one split of {SPLITS}, got {sorted(map(str, splits))}")
    for j in batch:
        if not isinstance(j.get("data"), (bytes, bytearray)) or not j.get("sha256"):
            raise PermissionError(f"job {j.get('clip_id')}: no bytes / sha256")
        got = hashlib.sha256(j["data"]).hexdigest()
        if got != j["sha256"]:
            raise PermissionError(f"clip {j['clip_id']}: bytes in the container differ from MANIFEST")
    return str(next(iter(splits)))


def container_ingest(batch: list[dict]) -> list[dict]:
    """Body of the ephemeral Modal function (GPU): re-hash each clip, then the dev per-clip ingest, by module name."""
    split = check_batch(batch)
    ing = importlib.import_module(INGEST_MODULE)
    from intercycle.perception.detector import RFDetrDetector  # pyright: ignore[reportMissingImports]

    det = RFDetrDetector(size="medium", device="cuda", precision="fp32", threshold=0.1,   # as cloud/modal_perc.py
                         keep=("person", "truck"), class_map={"truck": "truck"})
    rois = ing.load_press_rois(ROIS)
    out = []
    for job in batch:
        data = job["data"]
        with tempfile.TemporaryDirectory() as td:
            clip = Path(td) / f"{job['clip_id']}.mp4"
            clip.write_bytes(data)
            res = ing.ingest_clip(clip, job["clip_id"], split, int(job["camera"]), job["framing"],
                                  rois.get(job["framing"], {}), det.detect)
            files = {}
            for p in ing.write_outputs(res, Path(td) / "out"):
                files[Path(p).relative_to(Path(td) / "out").as_posix()] = Path(p).read_bytes()
        out.append({"clip_id": job["clip_id"], "stats": res.stats, "files": files})
    return out
