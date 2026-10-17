"""observer/stream.py: snapshot encoding round-trips, size cap, index/events files, live relay, no truth import."""
import ast
import asyncio
import json
import socket
from pathlib import Path

import numpy as np
import pytest

from intercycle.observer import stream as S

GRID = dict(x0=0.0, z0=0.0, res=1.0, nx=120, nz=48)


def _layers(nz=48, nx=120, seed=0):
    rng = np.random.default_rng(seed)
    occ = rng.random((nz, nx)).astype(np.float32) * (rng.random((nz, nx)) < 0.3)
    head = np.where(occ > 0, rng.integers(0, 8, (nz, nx)), -1).astype(np.int8)
    unk = np.zeros((nz, nx), bool)
    unk[10:14, 60:70] = True
    return occ, head, unk


def _snap(sw, t, grid=GRID, layers=None):
    occ, head, unk = layers or _layers(grid["nz"], grid["nx"])
    return sw.snapshot(t_sim=t, t_obs=t, prod_state="running", grid=grid,
                       l1={"S30": [(3, 26.0, 18.5, 28.0, 21.25)], "HEM1": [(0, 16.0, 37.0, 18.0, 39.0), (16, 1, 2, 3, 4)]},
                       l2_occ=occ, l2_head=head, l3_occ=occ * 0.5, l3_head=head, unknown=unk,
                       phase_now={"S30": 3, "HEM1": np.int64(16)}, metrics={"l1_iou": np.float32(0.9), "unsafe_free": 0.0, "brier": float("nan")})


def test_encoders_roundtrip():
    occ, head, unk = _layers()
    assert np.abs(S.decode_field(S.encode_occ(occ), occ.shape) - occ).max() <= 0.5 / 255 + 1e-6
    assert (S.decode_field(S.encode_head(head), head.shape) == head).all()
    assert (S.decode_field(S.encode_mask(unk), unk.shape) == unk).all()
    m = np.ones((3, 4), bool)
    assert S.encode_mask(m)["runs"] == [0, 12]
    big = np.random.default_rng(1).random((200, 500))
    e = S.encode_occ(big)
    assert e["enc"] == "u8z"
    assert np.abs(S.decode_field(e, big.shape) - big).max() <= 0.5 / 255 + 1e-6


def test_snapshot_files_index_events(tmp_path):
    with S.StreamWriter("t1", out_root=tmp_path, meta={"scenario": "bodyshop"}) as sw:
        p = _snap(sw, 3600.0)
        sw.event(t_sim=3700.0, kind="change_detected", x=np.float64(61.0), z=40, cells=np.int32(12))
        _snap(sw, 7200.0)
    d = tmp_path / "t1"
    assert p.name == "snap_000003600.json"
    snap = json.loads(p.read_text())
    assert snap["schema"] == "intercycle.stream/1" and snap["label"] == "SYNTHETIC: simulated plant"
    assert snap["grid"]["nx"] == 120 and snap["downsample"] == 1
    assert snap["metrics"]["brier"] is None and snap["phase_now"]["HEM1"] == 16
    assert snap["l1"]["S30"][0] == [3, 26.0, 18.5, 28.0, 21.25]
    occ, head, unk = _layers()
    shape = (snap["grid"]["nz"], snap["grid"]["nx"])
    assert (S.decode_field(snap["layers"]["unknown"], shape) == unk).all()
    assert (S.decode_field(snap["layers"]["l2_head"], shape) == head).all()
    assert len(p.read_bytes()) <= S.MAX_SNAP_BYTES
    idx = json.loads((d / "index.json").read_text())
    assert idx["complete"] and [s["file"] for s in idx["snapshots"]] == ["snap_000003600.json", "snap_000007200.json"]
    assert idx["events"][0]["kind"] == "change_detected" and idx["events"][0]["cells"] == 12
    lines = (d / "events.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["x"] == 61.0


def test_transposed_input_accepted(tmp_path):
    occ, head, unk = _layers()
    sw = S.StreamWriter("t2", out_root=tmp_path)
    with pytest.warns(UserWarning):
        p = _snap(sw, 0.0, layers=(occ.T.copy(), head.T.copy(), unk.T.copy()))
    snap = json.loads(p.read_text())
    assert (S.decode_field(snap["layers"]["unknown"], (48, 120)) == unk).all()
    with pytest.raises(ValueError):
        _snap(sw, 1.0, layers=(occ[:10], head[:10], unk[:10]))
    sw.close()


def test_downsample_keeps_under_cap(tmp_path):
    g = dict(x0=0.0, z0=0.0, res=0.1, nx=1200, nz=480)       # 0.1 m grid: random noise defeats zlib, must pool
    rng = np.random.default_rng(2)
    occ = rng.random((480, 1200)).astype(np.float32)
    head = rng.integers(-1, 8, (480, 1200)).astype(np.int8)
    unk = rng.random((480, 1200)) < 0.5
    with S.StreamWriter("t3", out_root=tmp_path) as sw:
        p = _snap(sw, 0.0, grid=g, layers=(occ, head, unk))
    snap = json.loads(p.read_text())
    k = snap["downsample"]
    assert k > 1 and len(p.read_bytes()) <= S.MAX_SNAP_BYTES
    assert snap["grid"]["res"] == pytest.approx(0.1 * k)
    pooled = S.decode_field(snap["layers"]["l2_occ"], (snap["grid"]["nz"], snap["grid"]["nx"]))
    assert pooled.max() >= occ.max() - 1 / 255                 # max pooling is conservative
    assert S.decode_field(snap["layers"]["unknown"], (snap["grid"]["nz"], snap["grid"]["nx"])).mean() > 0.9


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_live_relay(tmp_path):
    pytest.importorskip("websockets")
    from websockets.asyncio.client import connect

    port = _free_port()
    sw = S.StreamWriter("live", out_root=tmp_path, live=True, port=port)
    assert sw.relay is not None
    _snap(sw, 60.0)

    async def client():
        async with connect(f"ws://127.0.0.1:{port}") as ws:
            hello = json.loads(await asyncio.wait_for(ws.recv(), 5))
            assert hello["type"] == "hello" and hello["last"]["t_sim"] == 60.0
            await asyncio.get_running_loop().run_in_executor(None, lambda: (_snap(sw, 120.0), sw.event(t_sim=130.0, kind="machine_spawned", machine="NEW1")))
            got = [json.loads(await asyncio.wait_for(ws.recv(), 5)) for _ in range(2)]
            assert got[0]["type"] == "snapshot" and got[0]["snap"]["t_sim"] == 120.0
            assert got[1]["type"] == "event" and got[1]["event"]["machine"] == "NEW1"
            await asyncio.get_running_loop().run_in_executor(None, sw.close)
            assert json.loads(await asyncio.wait_for(ws.recv(), 5))["type"] == "close"

    asyncio.run(client())


def test_stream_does_not_import_truth():
    src = Path(S.__file__).read_text(encoding="utf-8")
    mods = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    assert not [m for m in mods if m.startswith(("intercycle.truth", "intercycle.sim"))]
