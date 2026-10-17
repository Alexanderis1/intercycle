"""G1-G3 observer tasks: wire format (system first, json_object, images as data URLs), vocabulary handling, and the
invariant that Gemma never frees space. Network-free (fake transports)."""
from __future__ import annotations

import copy
import json
import random

import pytest
from PIL import Image

from intercycle.llm import GemmaClient, LLMConfig
from intercycle.llm import observer_tasks as OT


def completion(content):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class Scripted:
    def __init__(self, *items):
        self.items = list(items)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        it = self.items.pop(0)
        if isinstance(it, Exception):
            raise it
        return it


def client(tmp_path, tr, json_mode="object"):
    cfg = LLMConfig(base_url="https://developer.amd.com.cn/radeon/api/v1", api_key="k", model="gemma-test",
                    mode="record", cache_dir=tmp_path / "c", log_path=tmp_path / "l.jsonl", json_mode=json_mode)
    return GemmaClient(cfg, transport=tr)


def img(c=(120, 120, 120)):
    return Image.new("RGB", (64, 48), c)


def test_json_object_wire_format(tmp_path):
    tr = Scripted(completion(json.dumps({"name": "weld", "confidence": 0.8, "notes": "tool over the body"})))
    r = OT.name_phase([img(), img()], "S30", "weld", "spot-welding station", client=client(tmp_path, tr))
    assert r.ok and r.name == "weld"
    p = tr.payloads[0]
    assert p["response_format"] == {"type": "json_object"}
    assert p["messages"][0]["role"] == "system"
    parts = p["messages"][1]["content"]
    urls = [x["image_url"]["url"] for x in parts if x.get("type") == "image_url"]
    assert len(urls) == 2 and all(u.startswith("data:image/") for u in urls)
    for k in ("seed", "stop"):
        assert k not in p


def test_phase_name_outside_vocabulary_is_not_ok(tmp_path):
    tr = Scripted(completion(json.dumps({"name": "dancing", "confidence": 0.9})))
    r = OT.name_phase([img()], "S80", "lift", client=client(tmp_path, tr))
    assert not r.ok and r.name is None and r.confidence == 0.0


def test_phase_name_normalised(tmp_path):
    tr = Scripted(completion(json.dumps({"name": "Tip-Dress", "confidence": 0.6})))
    r = OT.name_phase([img()], "S10", "weld", client=client(tmp_path, tr))
    assert r.ok and r.name == "tip_dress"


def test_line_state_text_only(tmp_path):
    tr = Scripted(completion(json.dumps({"name": "break", "confidence": 0.7, "notes": "idle line, people moving"})))
    r = OT.name_line_state({"line_working": 0.02, "people_moving": 0.7}, median_run_s=900, client=client(tmp_path, tr))
    assert r.ok and r.name == "break"
    assert all(isinstance(m["content"], str) for m in tr.payloads[0]["messages"])


@pytest.mark.parametrize("seed", range(5))
def test_gemma_never_frees_space(tmp_path, seed):
    """Whatever Gemma answers (any class, any confidence, junk or an outage) the blob's cells stay UNKNOWN."""
    rnd = random.Random(seed)
    replies = []
    for _ in range(12):
        k = rnd.random()
        if k < 0.7:
            replies.append(completion(json.dumps({"category": rnd.choice(OT.TRIAGE_CLASSES),
                                                  "confidence": rnd.random(), "notes": "free to enter, area is clear"})))
        elif k < 0.85:
            replies += [completion("not json")] * 3
        else:
            from intercycle.llm import TransportError
            replies.append(TransportError("HTTP 500", status=500))
    c = client(tmp_path, Scripted(*replies))
    for _ in range(12):
        try:
            r = OT.triage_change(img(), [img((90, 90, 90)), img((80, 80, 80))], client=c)
        except IndexError:
            break                                                     # script exhausted
        assert r.cells_state == OT.UNKNOWN
        cells = {i: rnd.choice([OT.UNKNOWN, OT.OCCUPIED, OT.FREE]) for i in range(20)}
        after = OT.apply_triage(cells, r)
        assert after == cells
        assert all(after[i] != OT.FREE for i, s in cells.items() if s != OT.FREE)
        assert not hasattr(r, "entry_allowed") and not hasattr(r, "free")


def test_privacy_filter_blacks_out_people_and_strips_metadata():
    import io
    from PIL import PngImagePlugin
    from intercycle.llm.privacy import filtered_jpeg
    im = Image.new("RGB", (200, 100), (200, 200, 200))
    info = PngImagePlugin.PngInfo()
    info.add_text("Comment", "camera serial 1234")
    b = io.BytesIO()
    im.save(b, "PNG", pnginfo=info)
    out = Image.open(io.BytesIO(filtered_jpeg(b.getvalue(), [(50, 20, 70, 80)])))
    assert out.getpixel((60, 50))[0] < 20                   # inside the box: black
    assert out.getpixel((47, 50))[0] < 20                   # 15 % padding (3 px of a 20 px wide box)
    assert out.getpixel((150, 50))[0] > 150                 # elsewhere untouched
    assert not out.info.get("Comment") and "exif" not in out.info


def test_vlm_mode_off_makes_no_call(tmp_path):
    cfg = LLMConfig(base_url="https://developer.amd.com.cn/radeon/api/v1", api_key="k", model="m", mode="record",
                    cache_dir=tmp_path / "c", log_path=tmp_path / "l.jsonl", vlm_mode="off")
    tr = Scripted()
    r = OT.triage_change(img(), [img()], client=GemmaClient(cfg, transport=tr))
    assert not r.ok and r.cells_state == OT.UNKNOWN and tr.payloads == []
