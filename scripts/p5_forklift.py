"""Probe P5 (PLAN_TWIN §3): vehicle / forklift detections on TRAIN frames of camera 9 and camera 14.

    # 1. copy the t=0.5 s frame of every TRAIN clip (prereg/split.csv) from the Modal volume (repo venv, has modal)
    python scripts/p5_forklift.py fetch
    # 2. RF-DETR-M (COCO car/truck/bus/motorcycle/bicycle) and OWLv2 "forklift" on those frames (GPU env, heavy lock)
    python scripts/p5_forklift.py detect
    # 3. counts per camera and detector, and the frames to check by eye
    python scripts/p5_forklift.py summary

Only TRAIN clip ids are ever read: ``fetch`` takes its ids from prereg/split.csv rows with split == "train" and
refuses any other id. Frames and detections stay under ``$INTERCYCLE_DATA/p5`` (gitignored).
Licences: RF-DETR Apache-2.0, OWLv2 (google/owlv2-base-patch16-ensemble) Apache-2.0.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "prereg" / "split.csv"
CAMERA_OF_PREFIX = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}
VEHICLE_COCO = ("car", "truck", "bus", "motorcycle", "bicycle")
RFDETR_THR = 0.3
OWL_QUERIES = ["a photo of a forklift"]
OWL_THR = 0.1  # OWLv2 scores are low by design; 0.1 is the transformers docs' example threshold


def out_dir() -> Path:
    d = Path(os.environ["INTERCYCLE_DATA"]) / "p5"
    (d / "frames").mkdir(parents=True, exist_ok=True)
    return d


def train_ids() -> list[str]:
    with open(SPLIT_CSV, newline="", encoding="utf-8") as fh:
        return sorted(r["clip_id"] for r in csv.DictReader(fh) if r["split"] == "train")


def camera(cid: str) -> int:
    return CAMERA_OF_PREFIX[int(cid.split("_")[0])]


def fetch() -> None:
    import modal

    vol = modal.Volume.from_name("intercycle-esk")
    ids = train_ids()
    allowed = set(ids)
    d = out_dir() / "frames"
    n_new = 0
    for cid in ids:
        assert cid in allowed  # TRAIN only, never a dev or test id
        dst = d / f"{cid}.jpg"
        if dst.exists():
            continue
        data = b"".join(vol.read_file(f"esk/frame05/{cid}.jpg"))
        dst.write_bytes(data)
        n_new += 1
    print(json.dumps({"train_ids": len(ids), "fetched": n_new, "dir": str(d)}))


def detect() -> None:
    import numpy as np
    import rfdetr
    import torch
    from PIL import Image
    from rfdetr.assets.coco_classes import COCO_CLASSES
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    d = out_dir()
    allowed = set(train_ids())
    paths = sorted(p for p in (d / "frames").glob("*.jpg") if p.stem in allowed)
    det_model = rfdetr.RFDETRMedium(device="cuda")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to("cuda").eval()
    recs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        r = det_model.predict(img, threshold=RFDETR_THR)
        names = [COCO_CLASSES.get(int(c), str(c)) for c in np.asarray(r.class_id)]
        rf = [{"cls": n, "conf": round(float(s), 4), "xyxy": [round(float(v), 1) for v in b]}
              for n, s, b in zip(names, np.asarray(r.confidence), np.asarray(r.xyxy)) if n in VEHICLE_COCO]
        persons = [[round(float(v), 1) for v in b] for n, s, b in zip(names, np.asarray(r.confidence),
                                                                         np.asarray(r.xyxy)) if n == "person"]
        inputs = proc(text=[OWL_QUERIES], images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = owl(**inputs)
        side = max(img.width, img.height)  # OWLv2 pads to a square (bottom/right): boxes are in padded coordinates
        res = proc.post_process_grounded_object_detection(outputs=out, target_sizes=torch.tensor([[side, side]]),
                                                          threshold=OWL_THR, text_labels=[OWL_QUERIES])[0]
        ow = [{"conf": round(float(s), 4), "xyxy": [round(float(v), 1) for v in b]}
              for s, b in zip(res["scores"].cpu(), res["boxes"].cpu())]
        recs.append({"clip_id": p.stem, "camera": camera(p.stem), "w": img.width, "h": img.height,
                     "rfdetr_vehicle": rf, "owl_forklift": ow, "person_xyxy": persons})
    (d / "detections.json").write_text(json.dumps(recs), encoding="utf-8")
    print(json.dumps({"frames": len(recs)}))


def summary() -> None:
    recs = json.loads((out_dir() / "detections.json").read_text(encoding="utf-8"))
    out: dict = {}
    for cam in (9, 14):
        sel = [r for r in recs if r["camera"] == cam]
        row = {"frames": len(sel)}
        for thr in (0.3, 0.5):
            row[f"rfdetr_frames@{thr}"] = sum(any(b["conf"] >= thr for b in r["rfdetr_vehicle"]) for r in sel)
            row[f"rfdetr_boxes@{thr}"] = sum(sum(b["conf"] >= thr for b in r["rfdetr_vehicle"]) for r in sel)
        for thr in (0.1, 0.2, 0.3):
            row[f"owl_frames@{thr}"] = sum(any(b["conf"] >= thr for b in r["owl_forklift"]) for r in sel)
            row[f"owl_boxes@{thr}"] = sum(sum(b["conf"] >= thr for b in r["owl_forklift"]) for r in sel)
        cls: dict[str, int] = {}
        for r in sel:
            for b in r["rfdetr_vehicle"]:
                cls[b["cls"]] = cls.get(b["cls"], 0) + 1
        row["rfdetr_classes@0.3"] = cls
        out[str(cam)] = row
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["fetch", "detect", "summary"])
    {"fetch": fetch, "detect": detect, "summary": summary}[ap.parse_args().action]()
