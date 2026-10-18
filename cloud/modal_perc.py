"""PERC batch on Modal L4 (lane S3a item 2; PLAN_TWIN §2 tracks / machines, §4 PERC gate): TRAIN + DEV clips only.

    modal run cloud/modal_perc.py --clips 0_tr1,3_tr4      # smoke test on named train/dev clips
    modal run cloud/modal_perc.py                          # every train + dev clip of $INTERCYCLE_DATA/inventory.parquet

Each clip goes through ``intercycle.perception.esk_ingest.ingest_clip`` (the same function sealed_run.py calls on test
clips): PyAV decode with container-PTS times, 12 fps sampling, RF-DETR-M (COCO, the staging model) at conf >= 0.1, a
fresh ByteTrack per clip, the explicit class map (person; forklift = truck >= 0.5 on camera 14), the P5 foot-point rule,
occlusion = MISSING on every press ROI of the clip's framing, press motion per ROI.

Outputs are mirrored on the volume ``intercycle-esk`` under ``/esk/perc/{tracks,machines/<press>,boxes}/<clip>.parquet``
and returned to the caller, which writes them under ``$INTERCYCLE_DATA`` and appends the per-clip stats to
``$INTERCYCLE_DATA/perc/stats.jsonl``.

SEALED: the batch reads ``esk.load(["train", "dev"])`` locally (the loader refuses test), and the container refuses
any clip whose split in the volume's ``split_a1.csv`` is not the train/dev split it was sent with. The volume holds
no test video (A1+ purge, ``cloud/modal_esk.py --action check``).

Licences: RF-DETR / trackers Apache-2.0, supervision MIT, PyAV BSD-3, OpenCV Apache-2.0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "s3a-perc"
VOLUME_NAME = "intercycle-esk"
VOL = "/vol"
ESK = f"{VOL}/esk"
SPLIT_FILE = "split_a1.csv"
ROIS_LOCAL = ROOT / "docs" / "probes" / "S3a-P4_rois.json"
ROIS = "/root/rois.json"
MAX_CONTAINERS = 6

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME)


def _prefetch_weights() -> None:
    import rfdetr  # pyright: ignore[reportMissingImports]

    rfdetr.RFDETRMedium(device="cpu")


# versions as in cloud/modal_esk.py (staging detector) and cloud/modal_perception.py (trackers)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("rfdetr==1.11.1", "supervision==0.30.6", "trackers==2.6.1", "transformers==5.18.0", "av==19.0.0",
                 "numpy", "pandas", "pyarrow", "pillow", "pyyaml", "pydantic")
    .run_function(_prefetch_weights)
    .apt_install("libgl1", "libglib2.0-0")
    .add_local_file(str(ROIS_LOCAL), ROIS)
    .add_local_python_source("intercycle")
)


@app.cls(image=image, gpu="L4", volumes={VOL: vol}, cpu=4, memory=16384, timeout=1800,
         max_containers=MAX_CONTAINERS)
class Perc:
    @modal.enter()
    def load(self) -> None:
        import csv

        from intercycle.perception.detector import RFDetrDetector

        # the staging model (RFDETRMedium, fp32 as staging's plain predict), raw labels person / truck
        self.det = RFDetrDetector(size="medium", device="cuda", precision="fp32", threshold=0.1,
                                  keep=("person", "truck"), class_map={"truck": "truck"})
        with open(f"{ESK}/split/{SPLIT_FILE}", newline="", encoding="utf-8") as fh:
            self.split = {r["clip_id"]: r for r in csv.DictReader(fh)}

    @modal.method()
    def run(self, cid: str, split: str, camera: int, framing: str) -> dict:
        import tempfile
        import time

        from intercycle.perception.esk_ingest import ingest_clip, load_press_rois, write_outputs

        if split not in ("train", "dev"):
            raise PermissionError(f"PERC batch refuses split={split!r} for {cid}")
        row = self.split.get(cid)
        if row is None or row["split"] != split or row["framing_id"] != framing:
            raise PermissionError(f"{cid}: not a {split}/{framing} clip of {SPLIT_FILE}")
        t0 = time.time()
        rois = load_press_rois(ROIS).get(framing, {})
        res = ingest_clip(f"{ESK}/clips/{cid}.mp4", cid, split, int(camera), framing, rois, self.det.detect)
        files: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory() as td:
            for p in write_outputs(res, td):
                rel = Path(p).relative_to(td).as_posix()
                data = Path(p).read_bytes()
                files[rel] = data
                dst = Path(ESK) / "perc" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                with open(dst, "wb") as fh:
                    fh.write(data)
        vol.commit()
        st = dict(res.stats)
        st["wall_s"] = time.time() - t0
        st["gpu"] = self.det.dev.get("name")
        return {"stats": st, "files": files}


@app.local_entrypoint()
def main(clips: str = "", limit: int = 0):
    import time

    sys.path.insert(0, str(ROOT))
    from intercycle import esk

    t0 = time.time()
    inv = esk.load(["train", "dev"])
    if clips:
        want = [c.strip() for c in clips.split(",") if c.strip()]
        bad = sorted(set(want) - set(inv["clip_id"].astype(str)))
        if bad:
            raise SystemExit(f"not train/dev clips: {bad}")
        inv = inv.loc[inv["clip_id"].astype(str).isin(want)].reset_index(drop=True)
    order = (-inv["duration"].to_numpy(float)).argsort(kind="stable")   # longest first, load balance
    inv = inv.iloc[order].reset_index(drop=True)
    if limit:
        inv = inv.head(limit)
    args = [(str(c), str(s), int(cam), str(f)) for c, s, cam, f in
            zip(inv["clip_id"], inv["split"], inv["camera"], inv["framing_id"])]
    root = esk.data_root()
    (root / "perc").mkdir(parents=True, exist_ok=True)
    ok, fail = 0, []
    with open(root / "perc" / "stats.jsonl", "a", encoding="utf-8") as log:
        for a, r in zip(args, Perc().run.starmap(args, return_exceptions=True)):
            if not isinstance(r, dict):
                fail.append(f"{a[0]}: {r!r}"[:300])
                continue
            for rel, data in r["files"].items():
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            log.write(json.dumps(r["stats"]) + "\n")
            ok += 1
    print(json.dumps({"ok": ok, "failed": len(fail), "fail_examples": fail[:5], "wall_s": round(time.time() - t0)}))
