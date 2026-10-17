"""Probe P8 (faces), lane S3a: face-size histogram on 100 TRAIN frames (PLAN_TWIN §3 P8).

    modal run cloud/modal_p8_faces.py            # L4: prints aggregate numbers only (JSON)

Frames: the t = 0.5 s frame (``/esk/frame05/<clip>.jpg``, 1920x1080) of 100 TRAIN clips of the committed
``prereg/split.csv`` (read from the volume copy ``/esk/split/split.csv``), drawn with a seeded RNG, stratified by camera
in proportion to the train clip counts. TEST and DEV clips are never opened.

Per frame:
* RF-DETR-M (COCO) person boxes at conf >= 0.1 (the privacy-mask threshold); the size statistics use conf >= 0.5.
* Head / face size from the person box: a standing adult (stature 1.72 m, the plan's walker height) has a head height
  of about 0.23 m and a face width of about 0.15 m, so head_px = 0.23/1.72 * h and face_w_px = 0.15/1.72 * h, and the
  pixel density on the person is h/1.72 px/m. Only "full-body" boxes (not touching the frame border, h/w >= 1.8) feed
  these estimates, because a truncated or bending person's box height is not their stature.
* A direct check: YuNet (OpenCV Zoo, MIT) face detector on the top 45 % of every conf >= 0.5 person box, upscaled 2x;
  detected face width in source pixels and the share of person boxes with a detected face (score >= 0.6).

Only aggregates leave the container (counts, quantiles, histogram bins). No image and no per-box record is returned.
Licences: RF-DETR Apache-2.0, OpenCV Apache-2.0, YuNet model MIT.
"""
from __future__ import annotations

import json

import modal

APP_NAME = "s3a-p8-faces"
VOLUME_NAME = "intercycle-esk"
VOL = "/vol"
ESK = f"{VOL}/esk"
SEED = 20261017 + 8  # P8; recorded in docs/probes/S3a-P8.md
N_FRAMES = 100
STATURE_M = 1.72
HEAD_M = 0.23
FACE_W_M = 0.15
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = "/root/yunet.onnx"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME)


def _prefetch_weights() -> None:
    import rfdetr

    rfdetr.RFDETRMedium(device="cpu")


def _fetch_yunet() -> None:
    import urllib.request

    urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)


gpu_image = (  # same chain as cloud/modal_esk.py (cache hit), plus OpenCV and the YuNet model
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("rfdetr==1.11.1", "supervision==0.30.6", "transformers==5.18.0", "numpy", "pillow")
    .run_function(_prefetch_weights)
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless==4.12.0.88")
    .run_function(_fetch_yunet)
)


def _q(v, qs=(10, 50, 90)) -> dict:
    import numpy as np

    if len(v) == 0:
        return {"n": 0}
    a = np.asarray(v, float)
    return {"n": int(a.size), **{f"p{q}": round(float(np.percentile(a, q)), 1) for q in qs},
            "min": round(float(a.min()), 1), "max": round(float(a.max()), 1)}


def _hist(v, edges) -> dict:
    import numpy as np

    c, _ = np.histogram(np.asarray(v, float), bins=edges)
    return {"edges_px": list(edges), "counts": c.astype(int).tolist()}


@app.function(image=gpu_image, gpu="L4", volumes={VOL: vol}, cpu=4, memory=16384, timeout=1200)
def p8_faces() -> dict:
    import csv

    import cv2
    import numpy as np
    import rfdetr
    from PIL import Image

    vol.reload()
    with open(f"{ESK}/split/split.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "train"]
    cam = {r["clip_id"]: (9 if int(r["clip_id"].split("_")[0]) in (0, 1, 2, 4, 5, 6) else 14) for r in rows}
    by_cam = {c: sorted(k for k, v in cam.items() if v == c) for c in (9, 14)}
    n_train = sum(len(v) for v in by_cam.values())
    rng = np.random.default_rng(SEED)
    want = {c: int(round(N_FRAMES * len(v) / n_train)) for c, v in by_cam.items()}
    want[9] = N_FRAMES - want[14]
    chosen = {c: sorted(rng.choice(by_cam[c], size=want[c], replace=False).tolist()) for c in (9, 14)}
    for c in (9, 14):  # defence in depth: every chosen id is a train id of the committed split
        assert all(cam.get(k) == c for k in chosen[c])

    model = rfdetr.RFDETRMedium(device="cuda")
    yunet = cv2.FaceDetectorYN.create(YUNET_PATH, "", (320, 320), 0.6, 0.3, 50)

    out: dict = {"seed": SEED, "n_train_clips": n_train, "frames_per_camera": {str(c): len(chosen[c]) for c in chosen}}
    for c in (9, 14):
        n_frames = n_missing = 0
        n01 = n05 = 0
        h_all, h_full, ppm, head, facew = [], [], [], [], []
        yn_w, yn_tested, yn_hit = [], 0, 0
        for cid in chosen[c]:
            try:
                im = Image.open(f"{ESK}/frame05/{cid}.jpg").convert("RGB")
            except FileNotFoundError:
                n_missing += 1
                continue
            n_frames += 1
            W, H = im.size
            d = model.predict(im, threshold=0.1)
            cls = np.asarray(d.class_id).astype(int)
            conf = np.asarray(d.confidence, float)
            xyxy = np.asarray(d.xyxy, float).reshape(-1, 4)
            person = cls == 1  # COCO id 1 = person
            n01 += int(person.sum())
            bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            for (x0, y0, x1, y1), s in zip(xyxy[person], conf[person]):
                if s < 0.5:
                    continue
                n05 += 1
                w, h = x1 - x0, y1 - y0
                h_all.append(h)
                border = x0 <= 2 or y0 <= 2 or x1 >= W - 3 or y1 >= H - 3
                if not border and h / max(w, 1.0) >= 1.8:
                    h_full.append(h)
                    ppm.append(h / STATURE_M)
                    head.append(h * HEAD_M / STATURE_M)
                    facew.append(h * FACE_W_M / STATURE_M)
                # direct face check on the top 45 % of the box, upscaled 2x
                ix0, iy0 = max(0, int(x0)), max(0, int(y0))
                ix1, iy1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y0 + 0.45 * h)))
                if ix1 - ix0 < 4 or iy1 - iy0 < 4:
                    continue
                crop = cv2.resize(bgr[iy0:iy1, ix0:ix1], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                yunet.setInputSize((crop.shape[1], crop.shape[0]))
                _, faces = yunet.detect(crop)
                yn_tested += 1
                if faces is not None and len(faces):
                    yn_hit += 1
                    yn_w.append(float(np.max(faces[:, 2])) / 2.0)  # widest face, back to source px
        out[f"camera_{c}"] = {
            "frames": n_frames, "frames_missing": n_missing,
            "person_boxes_conf_ge_0.1": n01, "person_boxes_conf_ge_0.5": n05,
            "box_height_px_all": _q(h_all),
            "full_body_boxes": len(h_full),
            "box_height_px_full_body": _q(h_full),
            "px_per_m_on_person": _q(ppm),
            "head_height_px_est": _q(head),
            "face_width_px_est": _q(facew),
            "face_width_px_est_hist": _hist(facew, [0, 5, 10, 15, 20, 25, 30, 40, 60, 100]),
            "yunet_boxes_tested": yn_tested, "yunet_boxes_with_face": yn_hit,
            "yunet_face_width_px": _q(yn_w),
            "yunet_face_width_px_hist": _hist(yn_w, [0, 5, 10, 15, 20, 25, 30, 40, 60, 100]),
        }
    return out


@app.local_entrypoint()
def main(out: str = ""):
    res = p8_faces.remote()
    txt = json.dumps(res, indent=2)
    print(txt)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(txt)
