"""C-W5: the robot must never trust a key it was not explicitly given (fail closed), plus the PLAN_TWIN truth guard.

    .venv/Scripts/python.exe -m pytest -q ros2/intercycle_nav/test/test_keyring_failclosed.py
"""
from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from intercycle import scenario as S  # noqa: E402
from intercycle.codec import ble  # noqa: E402
from intercycle.codec import trust as TR  # noqa: E402
from intercycle.codec.layout import FAST, encode_fast  # noqa: E402
from intercycle.codec.sign import demo_key, sign  # noqa: E402
from intercycle.types import DAY0_UNIX, State  # noqa: E402

from intercycle_nav import feed as F  # noqa: E402
from intercycle_nav import masks as K  # noqa: E402
from intercycle_nav.broadcast import SyntheticSchedule  # noqa: E402

SITE = K.Site.from_scenario()
T0 = 10 * 3600.0


def _sched(key_id: int | None = None):
    scn = S.load()
    return SyntheticSchedule(scn.site_id, scn.key_id if key_id is None else key_id, [60.0] * 8 + [45.0, 60.0],
                             [m.busy for m in scn.machines])


def _forged_idle_fast(t: float, key, key_id: int | None = None):
    """A fast layer saying "every machine IDLE for 200 s" signed with `key` (an attacker's choice)."""
    f = _sched(key_id).fast(t)
    f = replace(f, machines=[replace(m, state=State.IDLE, next_idle_len_s=200.0, ttl_s=300.0) for m in f.machines])
    return [F.FRAME + fr for fr in ble.frame(sign(encode_fast(f), key), layer=FAST)]


def _feed(st, t, recs):
    for r in recs:
        st.ingest(t, r)


# ------------------------------------------------------------------------------------------------ today's fail-open path
def test_no_keyring_never_trusts_the_public_demo_key():
    """Before C-W5: ObserverState(site, None) (what the bridge built when no `keyring` parameter was set) silently trusted
    the PUBLIC demo key, so anyone could compute it and forge "idle" windows. Now it refuses to build."""
    with pytest.raises(K.KeyringError):
        st = K.ObserverState(SITE, None)          # noqa: F841
        _feed(st, T0, _forged_idle_fast(T0, demo_key(SITE.key_id)))
        assert st.all_stats()["bridge"].get("ok", 0) == 0, "forged layer signed with the PUBLIC demo key was accepted"


def test_bridge_keyring_resolution_fails_closed():
    # a condition that consumes the feed refuses to start without a keyring unless dev=true
    for cond in ("learned", "oracle"):
        with pytest.raises(K.KeyringError):
            K.resolve_keyring(SITE, cond)
    # dev=true: the public demo key, explicitly
    kr = K.resolve_keyring(SITE, "learned", dev=True)
    assert isinstance(kr, dict) and set(kr) == {SITE.key_id}
    # none / static open no feed: an empty keyring (nothing verifies)
    for cond in ("none", "static"):
        assert K.resolve_keyring(SITE, cond) == {}
    # malformed explicit keyrings are errors, never a silent fallback
    for bad in ("1", "1:zz", "1:abcd", "x:" + "00" * 32, " , "):
        with pytest.raises(K.KeyringError):
            K.resolve_keyring(SITE, "learned", keyring_hex=bad)
    vk = demo_key(7).verify_key.encode().hex()
    kr7 = K.resolve_keyring(SITE, "learned", keyring_hex=f"7:{vk}")
    assert isinstance(kr7, dict) and set(kr7) == {7}
    with pytest.raises(K.KeyringError):              # ambiguous: both a pinned keyring and a cert keyring file
        K.resolve_keyring(SITE, "learned", keyring_hex=f"7:{vk}", keyring_file="x.json")
    with pytest.raises(K.KeyringError):              # a keyring file that cannot be read is an error
        K.resolve_keyring(SITE, "learned", keyring_file=str(PKG / "does_not_exist.json"))


def test_explicit_dev_keyring_still_works_and_rejects_other_keys():
    st = K.ObserverState(SITE, K.demo_keyring(SITE.key_id))
    _feed(st, T0, _forged_idle_fast(T0, demo_key(4242)))
    assert st.all_stats()["layers"].get("bad_signature", 0) == 1
    _feed(st, T0 + 0.1, _forged_idle_fast(T0 + 0.1, demo_key(SITE.key_id)))
    assert st.all_stats()["bridge"].get("ok", 0) == 1


# ------------------------------------------------------------------------------------------------ cert keyring end to end
def _store(t_unix: float, life_s: float = 3600.0):
    root = TR.SigningKey.generate()
    op = TR.SigningKey.generate()
    kid = TR.key_id_of(op.verify_key)
    cert = TR.issue_cert(root, op.verify_key, SITE.site_id, nbf=int(t_unix) - 10, exp=int(t_unix + life_s), serial=1)
    st = TR.TrustStore(root.verify_key, SITE.site_id)
    st.add_cert(cert)
    return root, op, kid, st


def test_cert_keyring_accepts_valid_key_and_fails_closed_on_expiry():
    now = DAY0_UNIX + T0
    _, op, kid, store = _store(now, life_s=60.0)
    st = K.ObserverState(SITE, store)
    _feed(st, T0, _forged_idle_fast(T0, op, key_id=kid))
    assert st.all_stats()["bridge"].get("ok", 0) == 1
    assert st.snapshot(T0 + 1.0).machines[0].state == "IDLE"
    # the cert expires at T0 + 60: the accepted layer is no longer trusted, a new one is rejected
    assert all(m.state == "UNKNOWN" for m in st.snapshot(T0 + 61.0).machines)
    _feed(st, T0 + 62.0, _forged_idle_fast(T0 + 62.0, op, key_id=kid))
    assert st.all_stats()["layers"].get("unknown_key", 0) == 1
    # a key the root never certified is rejected
    _feed(st, T0 + 1.0, _forged_idle_fast(T0 + 1.0, demo_key(SITE.key_id), key_id=kid))
    assert st.all_stats()["layers"].get("bad_signature", 0) >= 1


def test_cert_keyring_revocation_drops_the_key():
    now = DAY0_UNIX + T0
    root, op, kid, store = _store(now)
    st = K.ObserverState(SITE, store)
    _feed(st, T0, _forged_idle_fast(T0, op, key_id=kid))
    assert st.snapshot(T0 + 1.0).machines[0].state == "IDLE"
    store.apply_crl(TR.make_crl(root, SITE.site_id, serial=1, issue_time=int(now) + 2, revoked=[op.verify_key]))
    assert all(m.state == "UNKNOWN" for m in st.snapshot(T0 + 3.0).machines)
    _feed(st, T0 + 4.0, _forged_idle_fast(T0 + 4.0, op, key_id=kid))
    assert st.all_stats()["layers"].get("unknown_key", 0) == 1


def _map_frames(key, key_id: int, t: float, version: int = 1):
    from intercycle.codec import maplayer as ML
    from intercycle.observer import mapping as M

    m = M.LayeredMap(S.load())
    m.spec["key_id"] = key_id                       # the manifest header carries the signing key's id
    return [F.FRAME + fr for fr in ML.pack_map(m, key, version=version, issue_time=DAY0_UNIX + t - 60.0).carousel()]


def test_map_layer_follows_the_cert_keyring():
    now = DAY0_UNIX + T0
    root, op, kid, store = _store(now)
    st = K.ObserverState(SITE, store)
    _feed(st, T0, _map_frames(TR.SigningKey.generate(), kid, T0))          # right key_id, uncertified key
    assert st.map_view(T0 + 1.0) is None
    _feed(st, T0 + 1.0, _map_frames(op, kid, T0))
    assert st.map_view(T0 + 2.0) is not None, st.all_stats()
    store.apply_crl(TR.make_crl(root, SITE.site_id, serial=1, issue_time=int(now) + 3, revoked=[op.verify_key]))
    assert st.map_view(T0 + 4.0) is None                                    # revoked after delivery: map dropped


def test_map_layer_with_pinned_keyring():
    st = K.ObserverState(SITE, K.demo_keyring(SITE.key_id))
    _feed(st, T0, _map_frames(demo_key(SITE.key_id), SITE.key_id, T0))
    assert st.map_view(T0 + 1.0) is not None, st.all_stats()


# ------------------------------------------------------------------------------------------------ truth guard (PLAN_TWIN §2)
def test_assert_no_truth_forbids_twin_modules(monkeypatch):
    K.assert_no_truth()                                       # clean robot process
    monkeypatch.setitem(sys.modules, "twin.engine", types.ModuleType("twin.engine"))
    with pytest.raises(RuntimeError, match="twin.engine"):
        K.assert_no_truth()


def test_assert_no_truth_still_forbids_truth_and_sim(monkeypatch):
    for name in ("intercycle.truth.fake", "intercycle.sim.fake"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        with pytest.raises(RuntimeError):
            K.assert_no_truth()
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "twinkle", types.ModuleType("twinkle"))   # prefix is "twin.", not "twin"
    K.assert_no_truth()
