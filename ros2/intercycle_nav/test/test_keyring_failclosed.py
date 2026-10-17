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
DEMO_KID = TR.key_id_of(demo_key(SITE.key_id).verify_key)     # the PUBLIC demo key under its DERIVED key_id


def _sched(key_id: int | None = None):
    scn = S.load()
    return SyntheticSchedule(scn.site_id, DEMO_KID if key_id is None else key_id, [60.0] * 8 + [45.0, 60.0],
                             [m.busy for m in scn.machines])


def _forged_idle_fast(t: float, key, key_id: int | None = None):
    """A fast layer saying "every machine IDLE for 200 s" signed with `key` (an attacker's choice)."""
    f = _sched(key_id).fast(t)
    f = replace(f, machines=[replace(m, state=State.IDLE, next_idle_len_s=200.0, ttl_s=300.0) for m in f.machines])
    return [F.FRAME + fr for fr in ble.frame(sign(encode_fast(f), key), layer=FAST)]


class _MemWriter:
    def __init__(self, tr):
        self.tr = tr

    def frames(self, frs):
        for f in frs:
            self.tr.push(F.FRAME + f)


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
    assert isinstance(kr, dict) and set(kr) == {DEMO_KID}
    # none / static open no feed: an empty keyring (nothing verifies)
    for cond in ("none", "static"):
        assert K.resolve_keyring(SITE, cond) == {}
    # malformed explicit keyrings are errors, never a silent fallback
    for bad in ("1", "1:zz", "1:abcd", "x:" + "00" * 32, " , "):
        with pytest.raises(K.KeyringError):
            K.resolve_keyring(SITE, "learned", keyring_hex=bad)
    vk = demo_key(7).verify_key.encode().hex()
    kid7 = TR.key_id_of(demo_key(7).verify_key)
    kr7 = K.resolve_keyring(SITE, "learned", keyring_hex=f"{kid7}:{vk}")
    assert isinstance(kr7, dict) and set(kr7) == {kid7}
    with pytest.raises(K.KeyringError):              # ambiguous: both a pinned keyring and a cert keyring file
        K.resolve_keyring(SITE, "learned", keyring_hex=f"{kid7}:{vk}", keyring_file="x.json")
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
    _feed(st, T0, _map_frames(demo_key(SITE.key_id), DEMO_KID, T0))
    assert st.map_view(T0 + 1.0) is not None, st.all_stats()


# ------------------------------------------------------------------------------------------------ derived key_id (C-W5)
def test_pinned_keyring_refuses_a_key_id_not_derived_from_its_key():
    """C-W5 "key_id is a constant, not derived from the key": a pinned key under any id other than sha256(vk)[:2] (e.g.
    the scenario's constant key_id 1) is a startup error, never a silent trust."""
    op = TR.SigningKey.generate()
    kid, vk = TR.key_id_of(op.verify_key), op.verify_key.encode().hex()
    bad = (kid + 1) & 0xFFFF
    with pytest.raises(K.KeyringError, match="derived"):
        K.keyring_from_hex(f"{bad}:{vk}")
    with pytest.raises(K.KeyringError, match="derived"):
        K.ObserverState(SITE, {bad: op.verify_key})
    assert set(K.keyring_from_hex(f"{kid}:{vk}")) == {kid}
    assert set(K.demo_keyring(SITE.key_id)) == {DEMO_KID}                   # the dev key too


def test_broadcaster_puts_the_derived_key_id_in_every_header():
    """Producers: whatever key_id the observer model carries (here the scenario's constant), the Broadcaster signs with
    key_id_of(its key), so a robot holding only the certified key accepts both layers."""
    from intercycle_nav.broadcast import Broadcaster

    now = DAY0_UNIX + T0
    _, op, kid, store = _store(now)
    st = K.ObserverState(SITE, store)
    sch = _sched(key_id=SITE.key_id)                                         # constant id from the scenario
    assert SITE.key_id != kid
    tr = F.MemoryTransport()
    bc = Broadcaster(_MemWriter(tr), op, None, sch.slow, sch.fast)
    for k in range(4):
        bc.tick(T0 + 0.5 * k)
        for r in tr.poll():
            st.ingest(T0 + 0.5 * k, r)
    assert st.all_stats()["bridge"].get("ok", 0) >= 3, st.all_stats()
    assert st.snapshot(T0 + 1.6).slow_ok


def test_broadcaster_refuses_a_map_bundle_under_another_key_id():
    from intercycle.codec import maplayer as ML
    from intercycle.observer import mapping as M
    from intercycle_nav.broadcast import Broadcaster, pack_map_for_key

    now = DAY0_UNIX + T0
    _, op, kid, store = _store(now)
    m = M.LayeredMap(S.load())
    wrong = ML.pack_map(m, op, version=1, issue_time=now - 60.0)            # spec key_id = the scenario constant
    with pytest.raises(ValueError, match="key_id"):
        Broadcaster(_MemWriter(F.MemoryTransport()), op, wrong)
    bc = Broadcaster(_MemWriter(F.MemoryTransport()), op)
    with pytest.raises(ValueError, match="key_id"):
        bc.set_map(wrong)
    good = pack_map_for_key(m, op, version=1, issue_time=now - 60.0)
    assert good.manifest.key_id == kid and m.spec["key_id"] == SITE.key_id  # the caller's map is not modified
    st = K.ObserverState(SITE, store)
    _feed(st, T0, [F.FRAME + fr for fr in good.carousel()])
    assert st.map_view(T0 + 1.0) is not None, st.all_stats()


def test_eval_broadcaster_episode_key_line_uses_the_derived_key_id():
    """intercycle_eval.broadcaster writes observer_vk.txt, which eval/batch.py passes as the bridge's `keyring`."""
    sys.path.insert(0, str(PKG.parents[0] / "intercycle_eval"))
    from intercycle_eval.broadcaster import episode_key

    key, line = episode_key()
    kid = TR.key_id_of(key.verify_key)
    kr = K.keyring_from_hex(line)
    assert set(kr) == {kid} and kr[kid].encode() == key.verify_key.encode()


# ------------------------------------------------------------------------------------------------ key rotation (map cache)
def _two_key_store(now: float, life_a: float, life_b: float):
    root = TR.SigningKey.generate()
    a, b = TR.SigningKey.generate(), TR.SigningKey.generate()
    store = TR.TrustStore(root.verify_key, SITE.site_id)
    store.add_cert(TR.issue_cert(root, a.verify_key, SITE.site_id, nbf=int(now) - 10, exp=int(now + life_a), serial=1))
    store.add_cert(TR.issue_cert(root, b.verify_key, SITE.site_id, nbf=int(now) - 10, exp=int(now + life_b), serial=2))
    return a, b, store


def test_map_reattested_by_a_rotated_key_survives_the_old_key():
    """Audit S3b #3: map v1 delivered under key A (cert expires at +60 s) and re-announced at +30 s by certified key B
    (same version, same content): after A expires the map is still trusted, because B vouches for it."""
    now = DAY0_UNIX + T0
    a, b, store = _two_key_store(now, 60.0, 3600.0)
    st = K.ObserverState(SITE, store)
    _feed(st, T0, _map_frames(a, TR.key_id_of(a.verify_key), T0))
    assert st.map_view(T0 + 1.0) is not None
    _feed(st, T0 + 30.0, _map_frames(b, TR.key_id_of(b.verify_key), T0 + 30.0))
    assert st.map_rx.stats["cached"] >= 1
    assert st.map_view(T0 + 70.0) is not None, st.all_stats()


def test_map_reattestation_of_other_content_does_not_extend_trust():
    """A manifest from key B for the SAME version but OTHER content (different sha) does not vouch for the cached map:
    once A expires the map is dropped (fail closed)."""
    from intercycle.codec import maplayer as ML
    from intercycle.observer import mapping as M

    now = DAY0_UNIX + T0
    a, b, store = _two_key_store(now, 60.0, 3600.0)
    st = K.ObserverState(SITE, store)
    _feed(st, T0, _map_frames(a, TR.key_id_of(a.verify_key), T0))
    m2 = M.LayeredMap(S.load())
    m2.spec["key_id"] = TR.key_id_of(b.verify_key)
    m2.period[0] = 77.0                                                     # other content, same version number
    b2 = ML.pack_map(m2, b, version=1, issue_time=now + 30.0 - 60.0)
    first = st.map_rx.manifest
    assert first is not None and b2.manifest.sha != first.sha
    _feed(st, T0 + 30.0, [F.FRAME + fr for fr in b2.carousel()])
    assert st.map_view(T0 + 70.0) is None


def test_map_trust_ends_when_every_attesting_key_is_gone():
    now = DAY0_UNIX + T0
    a, b, store = _two_key_store(now, 60.0, 90.0)
    st = K.ObserverState(SITE, store)
    _feed(st, T0, _map_frames(a, TR.key_id_of(a.verify_key), T0))
    _feed(st, T0 + 30.0, _map_frames(b, TR.key_id_of(b.verify_key), T0 + 30.0))
    assert st.map_view(T0 + 70.0) is not None
    assert st.map_view(T0 + 95.0) is None


# ------------------------------------------------------------------------------------------------ truth guard (PLAN_TWIN §2)
def test_assert_no_truth_forbids_twin_modules(monkeypatch):
    K.assert_no_truth()                                       # clean robot process
    monkeypatch.setitem(sys.modules, "twin.engine", types.ModuleType("twin.engine"))
    with pytest.raises(RuntimeError, match="twin.engine"):
        K.assert_no_truth()


def test_assert_no_truth_still_forbids_truth_and_sim(monkeypatch):
    for name in ("intercycle.truth.fake", "intercycle.sim.fake",
                 "twin",                                       # the bare package (e.g. `import twin`)
                 "intercycle.eval.conformal_margin"):          # S7b: the offline margin calibration reads truth
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        with pytest.raises(RuntimeError):
            K.assert_no_truth()
        monkeypatch.delitem(sys.modules, name)
    for name in ("twinkle", "intercycle.eval", "intercycle.eval.conformal_margin_doc"):   # exact names, not prefixes
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    K.assert_no_truth()
