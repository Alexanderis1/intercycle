"""Bit layout, quantization error and Ed25519 signing of the two broadcast layers."""
import math

import numpy as np
import pytest

from intercycle.codec import layout as L
from intercycle.codec.sign import SIG_LEN, demo_key, keyring_from, sign, verify
from intercycle.scenario import load
from intercycle.types import DAY0_UNIX, FastLayer, MachineModel, MachineNow, SlowLayer, State

SC = load()
T0 = DAY0_UNIX + 8 * 3600


def slow_layer(t: float = T0) -> SlowLayer:
    """Scenario-sized slow layer (7 zones, 10 machines) with the scenario's own machine parameters."""
    rng = np.random.default_rng(7)
    traffic = rng.integers(0, 16, (len(SC.zones), 24)).tolist()
    ms = []
    for i, m in enumerate(SC.machines):
        if m.on_line:
            ms.append(MachineModel(i, SC.line.takt_s, m.busy, SC.line.jitter, SC.line.drift_per_h, SC.line.andon_stops_per_h, 12))
        else:
            ms.append(MachineModel(i, m.period_s, m.busy, m.jitter, -0.0003, m.stops_per_h, 9))
    return SlowLayer(SC.site_id, SC.key_id, t, traffic, ms)


def fast_layer(t: float = T0 + 0.123, seq: int = 41, ttl: float = 30.0) -> FastLayer:
    rng = np.random.default_rng(11)
    ms = [MachineNow(i, State(i % 4), float(rng.uniform(0, 60)), float(rng.uniform(1, 30)), ttl,
                     int(rng.integers(0, 16)), seq) for i in range(len(SC.machines))]
    return FastLayer(SC.site_id, SC.key_id, t, seq, ms)


def rel(a: float, b: float) -> float:
    return abs(a - b) / b


def test_slow_roundtrip_within_quantization():
    s = slow_layer()
    b = L.encode_slow(s)
    assert len(b) == L.slow_len(7, 10) == 14 + 84 + 80
    d = L.decode_slow(b)
    assert (d.site_id, d.key_id, d.traffic) == (s.site_id, s.key_id, s.traffic)
    assert d.issue_time == s.issue_time
    for a, m in zip(d.machines, s.machines):
        assert a.index == m.index and a.confidence == m.confidence
        assert abs(a.period_s - m.period_s) <= L.TOL["period_s"]
        assert abs(a.busy_fraction - m.busy_fraction) <= L.TOL["busy_fraction"]
        assert rel(a.jitter, m.jitter) <= L.TOL["jitter_rel"] + 1e-9
        assert abs(a.drift_per_h - m.drift_per_h) <= L.TOL["drift_per_h"] + 1e-12
        assert rel(a.stops_per_h, m.stops_per_h) <= L.TOL["stops_rel"] + 1e-9
    assert d == L.quantize_slow(s)       # the bit path and the arithmetic path agree exactly


def test_slow_quantization_bounds_random():
    rng = np.random.default_rng(3)
    for _ in range(2000):
        m = MachineModel(int(rng.integers(0, 64)), float(rng.uniform(0.5, 131.0)), float(rng.uniform(0, 1)),
                         float(10 ** rng.uniform(math.log10(1.2e-3), math.log10(0.18))), float(rng.uniform(-0.0032, 0.0031)),
                         float(2 ** rng.uniform(-2, 5)), int(rng.integers(0, 16)))
        q = L.quantize_model(m)
        assert abs(q.period_s - m.period_s) <= L.TOL["period_s"] + 1e-12
        assert abs(q.busy_fraction - m.busy_fraction) <= L.TOL["busy_fraction"] + 1e-12
        assert rel(q.jitter, m.jitter) <= L.TOL["jitter_rel"] + 1e-9
        assert abs(q.drift_per_h - m.drift_per_h) <= L.TOL["drift_per_h"] + 1e-12
        s = m.stops_per_h
        assert (rel(q.stops_per_h, s) <= L.TOL["stops_rel"] + 1e-9) if s >= 0.25 else abs(q.stops_per_h - s) <= L.TOL["stops_abs_low"]
        sl = SlowLayer(1, 1, T0, [[0] * 24], [m])
        assert L.decode_slow(L.encode_slow(sl)).machines == [q]


def test_slow_low_and_saturating_values():
    lo = MachineModel(0, 45.0, 0.5, 0.0004, -0.0040, 0.05, 15)
    hi = MachineModel(1, 45.0, 0.5, 0.5, 0.0040, 99.0, 15)
    d = L.decode_slow(L.encode_slow(SlowLayer(1, 1, T0, [], [lo, hi]))).machines
    assert d[0].jitter == 5e-4 and d[0].stops_per_h == 0.0           # "<0.1 %" and "none" codes
    assert d[0].drift_per_h == pytest.approx(-0.0032)                # saturates at -32 steps
    assert d[1].jitter == pytest.approx(1e-3 * 2 ** 7.5) and d[1].stops_per_h == pytest.approx(32.0)
    assert d[1].drift_per_h == pytest.approx(0.0031)
    assert L.jitter_val(L.jitter_code(0.00105)) == pytest.approx(0.0014142, rel=1e-4)   # lowest real bin overestimates


def test_slow_long_period_uses_x10_flag():
    ms = [MachineModel(0, 131.0, 0.5, 0.01, 0, 1, 8), MachineModel(1, 200.123, 0.5, 0.01, 0, 1, 8)]
    d = L.decode_slow(L.encode_slow(SlowLayer(1, 1, T0, [], ms))).machines
    assert abs(d[0].period_s - 131.0) <= 0.0005                      # ms mode
    assert abs(d[1].period_s - 200.123) <= 0.005 and d[1].period_s == 200.12   # 10 ms mode
    assert L.period_code(131.071) == (0, 131071) and L.period_code(131.072) == (1, 13107)
    with pytest.raises(L.CodecError):
        L.period_code(1400.0)


def test_fast_roundtrip_within_quantization():
    f = fast_layer()
    b = L.encode_fast(f)
    assert len(b) == L.fast_len(10) == 16 + 80
    d = L.decode_fast(b)
    assert (d.site_id, d.key_id, d.seq) == (f.site_id, f.key_id, f.seq)
    assert abs(d.issue_time - f.issue_time) <= L.TOL["fast_issue_s"]
    for a, m in zip(d.machines, f.machines):
        assert (a.index, a.state, a.confidence, a.seq) == (m.index, m.state, m.confidence, f.seq)
        assert abs(a.next_idle_in_s - m.next_idle_in_s) <= L.TOL["next_idle_in_s"] + 1e-12
        assert abs(a.next_idle_len_s - m.next_idle_len_s) <= L.TOL["next_idle_len_s"] + 1e-12
        assert 0 <= m.ttl_s - a.ttl_s < L.TOL["ttl_s"]          # never longer than intended
    assert d == L.quantize_fast(f)


def test_fast_ttl_rounds_down_and_saturates():
    for ttl, want in [(0.0, 0), (0.1, 0), (2.0, 1), (2.01, 1), (3.9, 1), (4.0, 2), (30.0, 15), (510.0, 255), (9999.0, 255)]:
        assert L.ttl_code(ttl) == want
    big = MachineNow(0, State.IDLE, 500.0, 500.0, 1.0, 15, 1)
    d = L.decode_fast(L.encode_fast(FastLayer(1, 1, T0, 1, [big]))).machines[0]
    assert d.next_idle_in_s == pytest.approx(131.071) and d.next_idle_len_s == pytest.approx(204.7)


def test_fast_issue_milliseconds_and_seq_wire_bytes():
    b = L.encode_fast(FastLayer(SC.site_id, SC.key_id, T0 + 0.987, 0xBEEF, []))
    assert len(b) == 16
    assert b[:2] == b"IC" and b[2] == 0x11                           # magic, version 1, layer 1 (fast)
    assert b[3:6] == SC.site_id.to_bytes(3, "big") and b[6:8] == SC.key_id.to_bytes(2, "big")
    assert int.from_bytes(b[8:12], "big") == T0 and int.from_bytes(b[12:14], "big") == 987
    assert b[14:16] == b"\xbe\xef"
    assert L.encode_slow(SlowLayer(1, 1, T0, [], []))[2] == 0x10     # layer 0 (slow)


def test_machine_records_are_bit_exact():
    n = MachineNow(5, State.STOPPED, 12.345, 6.7, 9.0, 11, 1)
    rec = L.encode_fast(FastLayer(1, 1, T0, 1, [n]))[16:]
    want = (5 << 58) | (2 << 56) | (12345 << 39) | (67 << 28) | (4 << 20) | (11 << 16)
    assert rec == want.to_bytes(8, "big")
    m = MachineModel(33, 60.0, 0.8, 0.01, 0.001, 2.0, 12)
    rec = L.encode_slow(SlowLayer(1, 1, T0, [], [m]))[14:]
    want = (33 << 58) | (60000 << 41) | (50 << 35) | (7 << 31) | (10 << 25) | (7 << 21) | (12 << 17)
    assert rec == want.to_bytes(8, "big")
    neg = L.encode_slow(SlowLayer(1, 1, T0, [], [MachineModel(0, 1.0, 0, 0, -0.0003, 0, 0)]))[14:]
    assert (int.from_bytes(neg, "big") >> 25) & 0x3F == 0b111101     # -3 in 6-bit two's complement
    big = L.encode_slow(SlowLayer(1, 1, T0, [], [MachineModel(0, 200.0, 0, 0, 0, 0, 0)]))[14:]
    assert (int.from_bytes(big, "big") >> 16) & 1 == 1               # x10 flag = first reserved bit


def test_decode_rejects_malformed():
    good = L.encode_slow(slow_layer())
    for bad in (good[:-1], good + b"\0", b"XX" + good[2:], good[:2] + b"\x20" + good[3:]):
        with pytest.raises(L.CodecError):
            L.decode_slow(bad)
    with pytest.raises(L.CodecError):
        L.decode_fast(good)                                          # wrong layer
    with pytest.raises(L.CodecError):
        L.decode_fast(L.encode_fast(fast_layer())[:-3])
    with pytest.raises(L.CodecError):
        L.encode_slow(SlowLayer(1, 1, T0, [[16] + [0] * 23], []))    # traffic level out of 4 bits
    with pytest.raises(L.CodecError):
        L.encode_fast(FastLayer(1, 1, T0, 1, [MachineNow(64, State.IDLE, 0, 0, 0, 0, 1)]))   # index out of 6 bits


def test_payload_sizes_for_this_scenario(capsys):
    assert (len(SC.zones), len(SC.machines)) == (7, 10)
    sk = demo_key()
    slow_b, fast_b = sign(L.encode_slow(slow_layer()), sk), sign(L.encode_fast(fast_layer()), sk)
    with capsys.disabled():
        print(f"\n[codec sizes] {len(SC.machines)} machines, {len(SC.zones)} zones")
        print(f"[codec sizes] fast = 16 + 80 + 64 = {len(fast_b)} B")
        print(f"[codec sizes] slow = 14 + 84 + 80 + 64 = {len(slow_b)} B")
    assert len(fast_b) == 160
    assert len(slow_b) == 242


def test_sign_verify_roundtrip_and_determinism():
    sk = demo_key(0)
    assert demo_key(0).encode() == sk.encode() and demo_key(1).encode() != sk.encode()
    payload = L.encode_fast(fast_layer())
    blob = sign(payload, sk)
    assert len(blob) == len(payload) + SIG_LEN and sign(payload, sk) == blob
    ok, p, key_id, why = verify(blob, keyring_from({SC.key_id: sk}))
    assert (ok, p, key_id, why) == (True, payload, SC.key_id, "")


def test_tamper_any_single_byte_fails():
    sk = demo_key()
    ring = keyring_from({SC.key_id: sk})
    blob = sign(L.encode_slow(slow_layer()), sk)
    for i in range(len(blob)):
        bad = blob[:i] + bytes([blob[i] ^ 0x01]) + blob[i + 1:]
        ok, p, _, why = verify(bad, ring)
        assert not ok and p == b"" and why in {"bad_signature", "malformed", "unknown_key"}, i
    body = blob[:40] + bytes([blob[40] ^ 0xFF]) + blob[41:]          # inside the payload proper
    assert verify(body, ring)[3] == "bad_signature"
    sig = blob[:-1] + bytes([blob[-1] ^ 0x01])
    assert verify(sig, ring)[3] == "bad_signature"


def test_unknown_key_wrong_key_and_short_blob():
    sk = demo_key(0)
    blob = sign(L.encode_fast(fast_layer()), sk)
    assert verify(blob, {})[3] == "unknown_key"
    assert verify(blob, keyring_from({SC.key_id + 1: sk}))[3] == "unknown_key"
    assert verify(blob, keyring_from({SC.key_id: demo_key(1)}))[3] == "bad_signature"
    assert verify(b"\0" * 20, {})[3] == "malformed"
    assert verify(b"\0" * 100, keyring_from({SC.key_id: sk}))[3] == "malformed"   # bad magic
