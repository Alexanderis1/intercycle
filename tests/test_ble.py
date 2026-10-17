"""BLE framing, channel simulation and the verifying receiver."""
import random

import numpy as np
import pytest

from intercycle.codec import ble as B
from intercycle.codec import layout as L
from intercycle.codec import seal
from intercycle.codec.sign import demo_key, keyring_from, sign
from intercycle.scenario import load
from intercycle.types import DAY0_UNIX, FastLayer, LiveLayer, MachineModel, MachineNow, Mover, MoverClass, SlowLayer, State

SC = load()
SK = demo_key(0)
RING = keyring_from({SC.key_id: SK})
T = 8 * 3600.0                                    # sim time of the first issue (08:00 on day 0)


def unix(t: float) -> float:
    return DAY0_UNIX + t


def slow_layer(t: float = T, zones: int = len(SC.zones), machines: int = len(SC.machines), site: int = SC.site_id,
               key_id: int = SC.key_id) -> SlowLayer:
    rng = np.random.default_rng(7)
    ms = [MachineModel(i % 64, 45.0 + i, 0.5 + 0.03 * (i % 10), 0.01, 0.001, 2.0, 12) for i in range(machines)]
    return SlowLayer(site, key_id, unix(t), rng.integers(0, 16, (zones, 24)).tolist(), ms)


def fast_layer(t: float = T, seq: int = 1, ttls: tuple[float, ...] = (30.0,) * 10, site: int = SC.site_id,
               key_id: int = SC.key_id) -> FastLayer:
    ms = [MachineNow(i, State(i % 3), 5.0 + i, 8.0, ttl, 12, seq) for i, ttl in enumerate(ttls)]
    return FastLayer(site, key_id, unix(t), seq, ms)


def live_layer(t: float = T, seq: int = 1, n: int = 6, site: int = SC.site_id, key_id: int = SC.key_id) -> LiveLayer:
    ms = [Mover(i, MoverClass(i % 5), 3.0 + 4 * i, 20.0 + i, 0.3 * i, 0.25 * i) for i in range(n)]
    return LiveLayer(site, key_id, unix(t), seq, ms)


def blob(layer, sk=SK) -> bytes:
    return seal(layer, sk)


def feed(rx: B.Receiver, t: float, blob_: bytes, **kw):
    """Send every frame of a blob one after another; return the last non-None verdict."""
    out = None
    for f in B.frame(blob_, **kw):
        out = rx.ingest(t, f) or out
    return out


# ---- framing -------------------------------------------------------------------------------------------

def test_crc8_check_value():
    assert B.crc8(b"123456789") == 0xF4          # CRC-8 poly 0x07 catalogue check value
    assert B.crc8(b"") == 0


def test_scenario_layers_frame_counts(capsys):
    fast_b, slow_b = blob(fast_layer()), blob(slow_layer())
    fast_f, slow_f = B.frame(fast_b), B.frame(slow_b)
    with capsys.disabled():
        print(f"\n[ble frames] frag payload = {B.FRAG_DATA} B/frame; AdvData max = {B.ADV_MAX} B")
        print(f"[ble frames] fast {len(fast_b)} B -> {len(fast_f)} frame(s), AdvData {[len(f) for f in fast_f]}")
        print(f"[ble frames] slow {len(slow_b)} B -> {len(slow_f)} frame(s), AdvData {[len(f) for f in slow_f]} (exact fit, 0 B spare)")
    assert (len(fast_b), len(slow_b)) == (160, 242)
    assert len(fast_f) == 1 and len(fast_f[0]) == 160 + 9
    assert len(slow_f) == 1 and len(slow_f[0]) == B.ADV_MAX
    assert B.FRAG_DATA == 242
    f = slow_f[0]
    assert f[0] == len(f) - 1 and f[1] == 0xFF and f[2:4] == b"\xff\xff"
    fr = B.parse_frame(f)
    assert (fr.layer, fr.idx, fr.count, fr.data) == (L.SLOW, 0, 1, slow_b)
    assert B.parse_frame(fast_f[0]).layer == L.FAST


def test_multi_frame_split_sizes():
    big = blob(slow_layer(zones=8, machines=12))
    assert len(big) == 14 + 96 + 96 + 64 == 270
    fs = B.frame(big)
    assert len(fs) == 2 and all(len(f) <= B.ADV_MAX for f in fs)
    assert [B.parse_frame(f).idx for f in fs] == [0, 1] and {B.parse_frame(f).count for f in fs} == {2}
    assert len(B.frame(b"x" * (15 * 242), layer=0)) == 15
    with pytest.raises(L.CodecError):
        B.frame(b"x" * (15 * 242 + 1), layer=0)


def test_parse_rejects_corruption():
    f = B.frame(blob(fast_layer()))[0]
    for i in range(len(f)):                       # any single flipped byte is caught (structure or CRC)
        assert B.parse_frame(f[:i] + bytes([f[i] ^ 0x5A]) + f[i + 1:]) is None, i
    assert B.parse_frame(f[:-1]) is None and B.parse_frame(f + b"\0") is None and B.parse_frame(b"") is None


# ---- reassembly with loss ------------------------------------------------------------------------------

def test_reassembly_needs_every_fragment():
    big = blob(slow_layer(zones=10, machines=12))      # 14+120+96+64 = 294 B -> 2 frames
    fs = B.frame(big)
    r = B.Reassembler(timeout_s=5.0)
    assert r.ingest(0.0, fs[1]) is None                # missing fragment 0: nothing comes out
    assert r.ingest(0.5, fs[1]) is None                # duplicates do not complete it
    assert r.ingest(1.0, fs[0]) == big                 # complete, out of order
    r2 = B.Reassembler()
    assert [r2.ingest(0.1 * i, f) for i, f in enumerate(fs)] == [None, big]


def test_reassembly_three_fragments_any_order_interleaved_messages():
    rng = random.Random(1)
    a, b = bytes(rng.randrange(256) for _ in range(500)), bytes(rng.randrange(256) for _ in range(600))
    fa, fb = B.frame(a, layer=0), B.frame(b, layer=1)
    assert (len(fa), len(fb)) == (3, 3)
    mixed = [fa[2], fb[0], fa[0], fb[2], fb[1], fa[1]]
    r, got = B.Reassembler(), []
    for i, f in enumerate(mixed):
        out = r.ingest(0.1 * i, f)
        if out is not None:
            got.append(out)
    assert got == [b, a]


def test_incomplete_message_dropped_after_timeout():
    fs = B.frame(blob(slow_layer(zones=8, machines=12)))
    r = B.Reassembler(timeout_s=2.0)
    assert r.ingest(0.0, fs[0]) is None
    assert r.ingest(3.0, fs[1]) is None                # fragment 0 aged out, so this one starts a fresh, incomplete buffer
    assert r.dropped == 1
    assert r.ingest(3.5, fs[0]) == b"".join(B.parse_frame(f).data for f in fs)


def test_reassembler_counts_bad_frames():
    r = B.Reassembler()
    f = B.frame(blob(fast_layer()))[0]
    assert r.ingest(0.0, f[:-1] + bytes([f[-1] ^ 1])) is None and r.bad_frames == 1
    assert r.ingest(0.0, b"\x02\x01\x06") is None and r.bad_frames == 2


# ---- channel -------------------------------------------------------------------------------------------

def lossless(**kw):
    return SC.ble.model_copy(update={"packet_loss": 0.0, **kw})


def test_channel_deterministic_per_seed_and_chunk_invariant():
    frames = {b: [bytes([b]) * 3] for b in range(9)}
    ch = B.BleChannel(SC.ble, seed=5)
    a = ch.receive((36, 24), 0, 20, frames)
    assert a == B.BleChannel(SC.ble, seed=5).receive((36, 24), 0, 20, frames)
    assert a == ch.receive((36, 24), 0, 7.3, frames) + ch.receive((36, 24), 7.3, 20, frames)
    assert a != B.BleChannel(SC.ble, seed=6).receive((36, 24), 0, 20, frames)
    assert [t for t, _ in a] == sorted(t for t, _ in a) and all(0 <= t < 20 for t, _ in a)


def test_channel_range_gates_broadcasters():
    frames = {b: [bytes([b])] for b in range(9)}
    ch = B.BleChannel(lossless(), seed=0)
    for xy in [(12, 32), (36, 24), (108, 24), (60, 16)]:
        heard = {f[0] for _, f in ch.receive(xy, 0, 10, frames)}
        assert heard == set(ch.in_range(xy)) and heard
    assert ch.receive((300, 300), 0, 30, frames) == []
    assert ch.in_range((12, 32)) == [b for b, (x, z) in enumerate(ch.pos) if (x - 12) ** 2 + (z - 32) ** 2 <= 25 ** 2]


def test_channel_interval_latency_and_round_robin():
    ch = B.BleChannel(lossless(broadcasters=[(0.0, 0.0)]), seed=3)
    ev = ch.receive((0, 0), 0, 60, [b"a", b"b", b"c"])
    ts = [t for t, _ in ev]
    gaps = np.diff(ts)
    assert abs(len(ev) - 60 / 0.5) <= 1
    assert gaps.min() >= 0.5 - 0.010 - 0.060 - 1e-9 and gaps.max() <= 0.5 + 0.010 + 0.060 + 1e-9
    nominal = ch.phase[0] + 0.5 * np.arange(-1, 125)
    for t, _ in ev:                                   # delay after the nominal event time is advDelay + latency
        d = t - nominal[np.searchsorted(nominal, t - 0.0901)]
        assert 0.020 - 1e-9 <= d <= 0.090 + 1e-9
    order = [b"a", b"b", b"c"]
    tags = [f for _, f in ev]
    assert all(order[(order.index(x) + 1) % 3] == y for x, y in zip(tags, tags[1:]))   # one frame per event, cycling


def test_channel_iid_loss_rate():
    frames = {b: [bytes([b])] for b in range(9)}
    full = B.BleChannel(lossless(), seed=9).receive((36, 24), 0, 4000, frames)
    got = B.BleChannel(SC.ble, seed=9).receive((36, 24), 0, 4000, frames)
    assert set(got) <= set(full)                       # loss only removes events, never adds or moves them
    assert len(full) > 10000
    assert abs(len(got) / len(full) - (1 - SC.ble.packet_loss)) < 0.01


def test_channel_dynamic_source_and_moving_robot():
    seen = []

    def src(b, t):
        seen.append((b, t))
        return [f"{b}@{int(t)}".encode()]

    ch = B.BleChannel(lossless(), seed=1)
    ev = ch.receive(lambda t: (12.0, 32.0) if t < 5 else (108.0, 24.0), 0, 10, src)
    near = {int(f.split(b"@")[0]) for t, f in ev if t < 5}
    far = {int(f.split(b"@")[0]) for t, f in ev if t >= 5}
    assert near == set(ch.in_range((12, 32))) and far == set(ch.in_range((108, 24))) == {8}
    assert seen and all(int(t_rx) - int(f.split(b"@")[1]) in (0, 1) for t_rx, f in ev)   # source is asked at the tx time


# ---- receiver ------------------------------------------------------------------------------------------

def test_receiver_accepts_both_layers():
    rx = B.Receiver(RING, SC.site_id)
    assert rx.latest(T).reason == "no_data" and not rx.latest(T).ok
    s, f = slow_layer(), fast_layer()
    d = feed(rx, T + 0.5, blob(s))
    assert d.ok and d.slow == L.quantize_slow(s) and d.extra["layer"] == "slow"
    d = feed(rx, T + 0.5, blob(f))
    assert d.ok and d.fast == L.quantize_fast(f) and d.extra["expired_fast"] == []
    now = rx.latest(T + 1)
    assert now.ok and now.reason == "" and now.slow == L.quantize_slow(s) and now.fast == L.quantize_fast(f)
    assert now.extra["fast_age_s"] == pytest.approx(1.0) and now.extra["expired_fast"] == []


def test_receiver_tampered_byte_is_bad_signature():
    rx = B.Receiver(RING, SC.site_id)
    good = blob(fast_layer(seq=1))
    assert feed(rx, T, good).ok
    for i in (13, 20, 70, len(good) - 1):               # header, body, signature
        bad = good[:i] + bytes([good[i] ^ 0x80]) + good[i + 1:]
        d = feed(rx, T + 1, bad)                         # reframed, so the frame CRC is valid and the signature must catch it
        assert not d.ok and d.reason == "bad_signature", i
    flipped = B.frame(blob(fast_layer(seq=2)))[0]        # corrupted on the air: dropped by the frame CRC before any verify
    assert rx.ingest(T + 2, flipped[:30] + bytes([flipped[30] ^ 1]) + flipped[31:]) is None
    assert rx.reasm.bad_frames == 1 and rx.latest(T + 2).fast.seq == 1


def test_receiver_replayed_message_is_stale_seq():
    rx = B.Receiver(RING, SC.site_id)
    b5, b6 = blob(fast_layer(T, seq=5)), blob(fast_layer(T + 2, seq=6))
    assert feed(rx, T, b5).ok and feed(rx, T + 2, b6).ok
    d = feed(rx, T + 4, b5)                              # attacker replays an old, correctly signed message
    assert not d.ok and d.reason == "stale_seq" and d.extra["duplicate"] is False
    d = feed(rx, T + 4, b6)
    assert d.reason == "stale_seq" and d.extra["duplicate"] is True      # repeat of the current one (benign)
    assert rx.latest(T + 4).fast.seq == 6 and rx.stats["stale_seq"] == 2
    old_slow, new_slow = blob(slow_layer(T - 3600)), blob(slow_layer(T))
    assert feed(rx, T, new_slow).ok
    assert feed(rx, T, old_slow).reason == "stale_seq"
    assert rx.latest(T).slow.issue_time == unix(T)


def test_receiver_seq_wraps_16_bit():
    rx = B.Receiver(RING, SC.site_id)
    assert feed(rx, T, blob(fast_layer(T, seq=65535))).ok
    assert feed(rx, T + 2, blob(fast_layer(T + 2, seq=0))).ok
    assert feed(rx, T + 4, blob(fast_layer(T + 4, seq=65535))).reason == "stale_seq"


def test_receiver_expired_fast_entries_become_unknown():
    rx = B.Receiver(RING, SC.site_id)
    ttls = (4.0,) + (60.0,) * 9
    f = fast_layer(T, seq=3, ttls=ttls)
    assert feed(rx, T + 0.05, blob(f)).ok
    fresh = rx.latest(T + 3.0)
    assert fresh.fast.machines[0].state == f.machines[0].state and fresh.extra["expired_fast"] == []
    d = rx.latest(T + 5.0)                               # 0.123 s margin past ttl 4 s: entry 0 expired, others live
    assert d.ok and d.extra["expired_fast"] == [0]
    m = d.fast.machines
    assert len(m) == 10 and m[0].state == State.UNKNOWN and m[0].index == 0 and m[0].ttl_s == 0 and m[0].confidence == 0
    assert [x.state for x in m[1:]] == [State(i % 3) for i in range(1, 10)]
    d = rx.latest(T + 61)
    assert d.extra["expired_fast"] == list(range(10)) and all(x.state == State.UNKNOWN for x in d.fast.machines)
    late = B.Receiver(RING, SC.site_id)                  # arrives already past its ttl: accepted, but shown as UNKNOWN
    d = feed(late, T + 100, blob(fast_layer(T, seq=1, ttls=(10.0,) * 10)))
    assert d.ok and d.extra["expired_fast"] == list(range(10))


def test_receiver_slow_layer_expires_after_8_hours():
    rx = B.Receiver(RING, SC.site_id)
    d = feed(rx, T, blob(slow_layer(T - 8 * 3600 - 1)))
    assert not d.ok and d.reason == "expired" and d.extra["layer"] == "slow" and rx.latest(T).reason == "no_data"
    assert feed(rx, T, blob(slow_layer(T - 8 * 3600 + 60))).ok          # inside the window
    assert rx.latest(T + 30).slow is not None
    d = rx.latest(T + 120)
    assert d.slow is None and d.extra["slow_expired"] is True and d.reason == "expired" and not d.ok
    assert feed(rx, T + 130, blob(fast_layer(T + 130))).ok
    d = rx.latest(T + 131)
    assert d.ok and d.slow is None and d.fast is not None and d.extra["slow_expired"] is True


def test_receiver_unknown_key_wrong_site_malformed():
    rx = B.Receiver(RING, SC.site_id)
    other = demo_key(99)
    d = feed(rx, T, blob(fast_layer(key_id=7), other))
    assert not d.ok and d.reason == "unknown_key" and d.extra["key_id"] == 7
    d = feed(rx, T, blob(fast_layer(), other))           # known key_id, signed by someone else
    assert d.reason == "bad_signature"
    d = feed(rx, T, blob(fast_layer(site=SC.site_id + 1)))
    assert d.reason == "wrong_site" and d.extra["site_id"] == SC.site_id + 1
    garbage = sign(b"IC" + b"\x01" * 30, SK)             # validly signed, but not a layer we can parse
    assert feed(rx, T, garbage, layer=0).reason == "malformed"
    trunc = sign(L.encode_fast(fast_layer())[:-3], SK)   # validly signed, but length is not 16 + 8n
    assert feed(rx, T, trunc).reason == "malformed"
    assert rx.latest(T).reason == "no_data"              # nothing rejected may leak into the accepted state
    assert rx.stats["unknown_key"] == 1 and rx.stats["wrong_site"] == 1


def test_receiver_fragment_loss_then_recovery():
    rx = B.Receiver(RING, SC.site_id, timeout_s=5.0)
    big = slow_layer(zones=8, machines=12)
    fs = B.frame(blob(big))
    assert len(fs) == 2
    assert rx.ingest(0 + T, fs[0]) is None               # fragment 1 lost this round
    assert rx.latest(T + 1).slow is None
    assert rx.ingest(T + 6, fs[1]) is None               # next round's fragment 1 arrives after the 5 s timeout
    assert rx.latest(T + 6).slow is None and rx.reasm.dropped == 1
    assert rx.ingest(T + 12, fs[0]) is None              # ... and fragment 0 only after fragment 1 aged out as well
    assert rx.reasm.dropped == 2 and rx.latest(T + 12).slow is None
    d = rx.ingest(T + 12.5, fs[1])
    assert d.ok and d.slow == L.quantize_slow(big) and rx.latest(T + 13).slow == d.slow


@pytest.mark.parametrize("seed", range(6))
def test_end_to_end_over_lossy_channel(seed):
    """Robot parked mid-hall hears two broadcasters (10 % loss, 20-80 ms) sending 2 slow frames + 1 fast frame."""
    big = slow_layer(zones=8, machines=12)
    fast = fast_layer(T, seq=11)
    frames = B.frame(blob(big)) + B.frame(blob(fast))
    assert len(frames) == 3
    ch = B.BleChannel(SC.ble, seed=seed)
    assert len(ch.in_range((36, 24))) == 2
    rx = B.Receiver(RING, SC.site_id)
    t_slow = t_fast = None
    for k in range(60):
        for t_rx, f in ch.receive((36, 24), T + k, T + k + 1, {b: frames for b in range(9)}):
            d = rx.ingest(t_rx, f)
            if d is not None and d.ok:
                t_slow = t_rx if d.slow is not None and t_slow is None else t_slow
                t_fast = t_rx if d.fast is not None and t_fast is None else t_fast
    assert t_slow is not None and t_fast is not None and t_slow - T < 30 and t_fast - T < 30
    last = rx.latest(T + 60)
    assert last.slow == L.quantize_slow(big) and last.fast.seq == 11
    assert rx.stats["ok_slow"] == 1 and rx.stats["ok_fast"] == 1      # repeats are rejected as duplicates, not re-accepted
    assert rx.stats["stale_seq"] > 0 and rx.reasm.bad_frames == 0


def test_robot_out_of_range_hears_nothing():
    ch = B.BleChannel(SC.ble, seed=0)
    rx = B.Receiver(RING, SC.site_id)
    frames = B.frame(blob(fast_layer()))
    assert not any(rx.ingest(t, f) for t, f in ch.receive((300, 300), T, T + 30, {b: frames for b in range(9)}))
    assert rx.latest(T + 30).reason == "no_data"


# ---- live layer ----------------------------------------------------------------------------------------

def test_live_frames_and_split_messages():
    one = B.frame(blob(live_layer(n=32)))
    assert len(one) == 1 and len(one[0]) == 250 and B.parse_frame(one[0]).layer == L.LIVE
    with pytest.raises(L.CodecError):
        blob(live_layer(n=33))
    a, b = live_layer(seq=1, n=20), live_layer(seq=2, n=13)          # caller-side split: two messages, own msg_id each
    fa, fb = B.frame(blob(a)), B.frame(blob(b))
    assert len(fa) == len(fb) == 1 and B.parse_frame(fa[0]).msg_id != B.parse_frame(fb[0]).msg_id


def test_receiver_accepts_live_and_each_layer_has_its_own_seq():
    rx = B.Receiver(RING, SC.site_id)
    lv = live_layer(seq=3)
    d = feed(rx, T + 0.1, blob(lv))
    assert d.ok and d.live == L.quantize_live(lv) and d.fast is None and d.extra["layer"] == "live"
    assert feed(rx, T + 0.1, blob(fast_layer(seq=3))).ok             # same seq number on another layer is fine
    assert feed(rx, T + 0.1, blob(fast_layer(seq=2))).reason == "stale_seq"
    assert feed(rx, T + 0.5, blob(live_layer(T + 0.4, seq=4))).ok
    now = rx.latest(T + 0.6)
    assert now.ok and now.live.seq == 4 and now.fast.seq == 3 and now.slow is None
    assert now.extra["live_age_s"] == pytest.approx(0.2) and "live_expired" not in now.extra
    assert rx.stats["ok_live"] == 2 and rx.stats["ok_fast"] == 1


def test_receiver_live_tamper_replay_and_wrap():
    rx = B.Receiver(RING, SC.site_id)
    b5, b6 = blob(live_layer(T, seq=5)), blob(live_layer(T + 0.5, seq=6))
    assert feed(rx, T, b5).ok and feed(rx, T + 0.5, b6).ok
    for i in (14, 18, 30, len(b6) - 1):
        d = feed(rx, T + 0.6, b6[:i] + bytes([b6[i] ^ 0x04]) + b6[i + 1:])
        assert d.reason == "bad_signature", i
    d = feed(rx, T + 0.7, b5)
    assert not d.ok and d.reason == "stale_seq" and d.extra["layer"] == "live" and d.extra["duplicate"] is False
    assert feed(rx, T + 0.7, b6).extra["duplicate"] is True
    assert rx.latest(T + 0.8).live.seq == 6
    wrap = B.Receiver(RING, SC.site_id)
    assert feed(wrap, T, blob(live_layer(T, seq=65535))).ok
    assert feed(wrap, T + 0.5, blob(live_layer(T + 0.5, seq=0))).ok
    assert feed(wrap, T + 0.6, blob(live_layer(T + 0.6, seq=65535))).reason == "stale_seq"
    forged = feed(rx, T + 0.8, blob(live_layer(T + 0.8, seq=9), demo_key(5)))
    assert forged.reason == "bad_signature" and rx.latest(T + 0.8).live.seq == 6


def test_receiver_live_expires_after_1_5_seconds():
    rx = B.Receiver(RING, SC.site_id)
    lv = live_layer(T, seq=1)
    assert feed(rx, T + 0.05, blob(lv)).ok
    d = rx.latest(T + 1.4)
    assert d.ok and d.live == L.quantize_live(lv) and "live_expired" not in d.extra
    d = rx.latest(T + 1.6)
    assert not d.ok and d.reason == "expired" and d.live is None and d.extra["live_expired"] is True
    assert feed(rx, T + 1.6, blob(fast_layer(T + 1.6, seq=1))).ok   # other layers keep working
    d = rx.latest(T + 1.7)
    assert d.ok and d.live is None and d.fast is not None and d.extra["live_expired"] is True
    stale = B.Receiver(RING, SC.site_id)                            # already too old on arrival: rejected, seq not consumed
    d = feed(stale, T + 2.0, blob(live_layer(T, seq=9)))
    assert not d.ok and d.reason == "expired" and d.extra["layer"] == "live"
    assert feed(stale, T + 2.0, blob(live_layer(T + 2.0, seq=1))).ok


def test_live_stream_over_channel():
    """Observer re-issues the live layer every 0.5 s; a robot in range of two broadcasters keeps a fresh view."""
    ch = B.BleChannel(SC.ble, seed=2)
    issued = {}

    def src(b, t):
        k = int((t - T) // 0.5)
        if k not in issued:
            issued[k] = B.frame(blob(live_layer(T + 0.5 * k, seq=k + 1, n=10)))
        return issued[k]

    rx = B.Receiver(RING, SC.site_id)
    ages = []
    for step in range(100):
        t0 = T + 0.1 * step
        for t_rx, f in ch.receive((36, 24), t0, t0 + 0.1, src):
            rx.ingest(t_rx, f)
        d = rx.latest(t0 + 0.1)
        if step >= 20:
            assert d.ok and d.live is not None, step                 # never goes stale despite 10 % loss
            ages.append(d.extra["live_age_s"])
    assert max(ages) < 1.5 and rx.latest(T + 10).live.seq >= 18
    assert rx.stats["ok_live"] >= 18
