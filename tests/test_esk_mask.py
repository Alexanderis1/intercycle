"""scripts/esk_mask.py on a synthetic clip with a stub detector: train/dev only, frames masked, PTS times."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("esk_mask", ROOT / "scripts" / "esk_mask.py")
assert spec is not None and spec.loader is not None
esk_mask = importlib.util.module_from_spec(spec)
sys.modules["esk_mask"] = esk_mask
spec.loader.exec_module(esk_mask)

W, H, FPS, N = 160, 96, 10, 20
PERSON = (60, 20, 80, 70)  # bright rectangle = the "person"


def _write_clip(path: Path) -> None:
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=FPS)
        s.width, s.height, s.pix_fmt = W, H, "yuv420p"
        for i in range(N):
            img = np.full((H, W, 3), 60, np.uint8)
            x0, y0, x1, y1 = PERSON
            img[y0:y1, x0:x1] = 250
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def _stub_detect(images):
    out = []
    for k, _ in enumerate(images):
        out.append((np.array([PERSON], float), np.array([0.12])))
    return out


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "clips").mkdir()
    _write_clip(tmp_path / "clips" / "0_tr1.mp4")
    _write_clip(tmp_path / "clips" / "0_te9.mp4")
    inv = pd.DataFrame({"clip_id": ["0_tr1", "0_te9"], "camera": [9, 9], "framing_id": ["c9_f0"] * 2,
                        "group_id": ["g1", "g2"], "class": [0, 0], "duration": [2.0, 2.0],
                        "sha256": ["a", "b"], "split": ["train", "test"]})
    inv.to_parquet(tmp_path / "inventory.parquet")
    monkeypatch.delenv("INTERCYCLE_SEALED_RUN", raising=False)
    return tmp_path


def test_masks_train_clip_and_never_touches_test(root):
    assert esk_mask.main(["--split", "train", "dev", "--fps", "4", "--root", str(root)], detect=_stub_detect) == 0
    out = root / "blurred" / "train" / "0_tr1"
    jpgs = sorted(out.glob("*.jpg"))
    assert len(jpgs) == 8  # 2 s at 4 fps on a fixed 0.25 s grid
    # names are the PTS time in ms of the first source frame at or after each grid point (10 fps source)
    assert [p.name for p in jpgs[:3]] == ["000000.jpg", "000300.jpg", "000500.jpg"]
    import cv2

    img = cv2.imread(str(jpgs[0]))
    assert img is not None
    x0, y0, x1, y1 = PERSON
    assert img[y0:y1, x0:x1].max() <= 10  # person blacked out whole (low-conf 0.12 box)
    assert img[5, 5].mean() > 30  # background kept
    assert not (root / "blurred" / "test").exists()


def test_test_split_rejected_by_parser(root):
    with pytest.raises(SystemExit):
        esk_mask.main(["--split", "test", "--root", str(root)], detect=_stub_detect)


def test_test_clip_id_rejected(root):
    with pytest.raises(SystemExit):
        esk_mask.main(["--split", "train", "--clips", "0_te9", "--root", str(root)], detect=_stub_detect)


def test_crop_is_taken_after_masking(root):
    esk_mask.main(["--split", "train", "--fps", "2", "--crop", "50,10,120,80", "--root", str(root)],
                  detect=_stub_detect)
    import cv2

    img = cv2.imread(str(sorted((root / "blurred" / "train" / "0_tr1").glob("*.jpg"))[0]))
    assert img is not None
    assert img.shape[:2] == (70, 70)
    assert img[10:60, 10:30].max() <= 10  # the person inside the crop is black
