"""C-W5 key management: derived key_id, site-root certs (nbf/exp), revocation, keyring file, private signing-key files."""
from __future__ import annotations

import os
import subprocess

import pytest

from intercycle.codec import trust as TR
from intercycle.codec.layout import FastLayer, encode_fast
from intercycle.codec.sign import sign, verify

SITE = 0x4A3D11
NOW = 1_800_000_000


def _keys():
    return TR.SigningKey.generate(), TR.SigningKey.generate()


def _store(root, *certs):
    st = TR.TrustStore(root.verify_key, SITE)
    for c in certs:
        st.add_cert(c)
    return st


def test_key_id_is_derived_from_the_key():
    import hashlib

    _, op = _keys()
    kid = TR.key_id_of(op.verify_key)
    assert kid == int.from_bytes(hashlib.sha256(op.verify_key.encode()).digest()[:2], "big")
    assert 0 <= kid < 1 << 16


def test_valid_cert_gives_a_keyring_inside_nbf_exp_only():
    root, op = _keys()
    st = _store(root, TR.issue_cert(root, op.verify_key, SITE, nbf=NOW, exp=NOW + 100, serial=1))
    kid = TR.key_id_of(op.verify_key)
    assert st.keyring_at(NOW - 1) == {}
    assert set(st.keyring_at(NOW)) == {kid} and set(st.keyring_at(NOW + 100)) == {kid}
    assert st.keyring_at(NOW + 101) == {}
    # a layer signed by the certified key verifies against keyring_at(now), with the derived key_id in its header
    blob = sign(encode_fast(FastLayer(SITE, kid, float(NOW + 5), 1, [])), op)
    assert verify(blob, st.keyring_at(NOW + 5))[0]
    assert verify(blob, st.keyring_at(NOW + 500))[3] == "unknown_key"


@pytest.mark.parametrize("case", ["other_root", "wrong_site", "tampered", "truncated", "bad_window"])
def test_bad_certs_are_refused(case):
    root, op = _keys()
    st = TR.TrustStore(root.verify_key, SITE)
    if case == "other_root":
        blob = TR.issue_cert(TR.SigningKey.generate(), op.verify_key, SITE, NOW, NOW + 10)
    elif case == "wrong_site":
        blob = TR.issue_cert(root, op.verify_key, SITE + 1, NOW, NOW + 10)
    elif case == "tampered":                                           # extend exp without the root
        b = bytearray(TR.issue_cert(root, op.verify_key, SITE, NOW, NOW + 10))
        b[18] ^= 0xFF
        blob = bytes(b)
    elif case == "truncated":
        blob = TR.issue_cert(root, op.verify_key, SITE, NOW, NOW + 10)[:40]
    else:
        with pytest.raises(TR.TrustError):
            TR.issue_cert(root, op.verify_key, SITE, NOW, NOW)
        return
    with pytest.raises(TR.TrustError):
        st.add_cert(blob)
    assert st.certs == {} and st.keyring_at(NOW + 1) == {}


def test_cert_with_a_constant_key_id_is_refused():
    """A root-signed cert whose key_id is not sha256(vk)[:2] (e.g. the old constant key_id 1) is refused."""
    root, op = _keys()
    body = TR._CERT.pack(TR.CERT_MAGIC, SITE, 1 if TR.key_id_of(op.verify_key) != 1 else 2, NOW, NOW + 10, 0,
                         op.verify_key.encode())
    with pytest.raises(TR.TrustError, match="key_id"):
        TR.TrustStore(root.verify_key, SITE).add_cert(body + root.sign(body).signature)


def test_revocation_and_crl_ordering():
    root, op = _keys()
    _, op2 = _keys()
    c1 = TR.issue_cert(root, op.verify_key, SITE, NOW, NOW + 1000)
    c2 = TR.issue_cert(root, op2.verify_key, SITE, NOW, NOW + 1000)
    st = _store(root, c1, c2)
    assert len(st.keyring_at(NOW + 1)) == 2
    st.apply_crl(TR.make_crl(root, SITE, serial=5, issue_time=NOW + 2, revoked=[op.verify_key]))
    assert set(st.keyring_at(NOW + 3)) == {TR.key_id_of(op2.verify_key)}
    with pytest.raises(TR.TrustError):                                 # a revoked key cannot be re-admitted
        st.add_cert(c1)
    with pytest.raises(TR.TrustError):                                 # replaying an older / equal CRL is refused
        st.apply_crl(TR.make_crl(root, SITE, serial=5, issue_time=NOW + 3, revoked=[]))
    with pytest.raises(TR.TrustError):                                 # a CRL not signed by the root is refused
        st.apply_crl(TR.make_crl(TR.SigningKey.generate(), SITE, serial=9, issue_time=NOW + 3, revoked=[op2.verify_key]))
    assert set(st.keyring_at(NOW + 3)) == {TR.key_id_of(op2.verify_key)}


def test_key_id_collision_is_refused():
    root = TR.SigningKey.generate()
    seen: dict[int, TR.SigningKey] = {}
    while True:                                                        # birthday search for two keys with one key_id
        k = TR.SigningKey.generate()
        kid = TR.key_id_of(k.verify_key)
        if kid in seen:
            break
        seen[kid] = k
    st = _store(root, TR.issue_cert(root, seen[kid].verify_key, SITE, NOW, NOW + 10))
    with pytest.raises(TR.TrustError, match="collision"):
        st.add_cert(TR.issue_cert(root, k.verify_key, SITE, NOW, NOW + 10))
    assert st.keyring_at(NOW + 1)[kid].encode() == seen[kid].verify_key.encode()


def test_keyring_file_roundtrip_and_errors(tmp_path):
    root, op = _keys()
    cert = TR.issue_cert(root, op.verify_key, SITE, NOW, NOW + 10)
    p = tmp_path / "keyring.json"
    p.write_text(TR.keyring_file_json(root.verify_key, SITE, [cert]), encoding="utf-8")
    st = TR.load_trust_store(p, SITE)
    assert set(st.keyring_at(NOW + 1)) == {TR.key_id_of(op.verify_key)}
    with pytest.raises(TR.TrustError):
        TR.load_trust_store(p, SITE + 1)                               # wrong site
    p.write_text(TR.keyring_file_json(TR.SigningKey.generate().verify_key, SITE, [cert]), encoding="utf-8")
    with pytest.raises(TR.TrustError):
        TR.load_trust_store(p, SITE)                                   # cert not signed by the provisioned root
    p.write_text(TR.keyring_file_json(root.verify_key, SITE, []), encoding="utf-8")
    with pytest.raises(TR.TrustError):
        TR.load_trust_store(p, SITE)                                   # no certs
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(TR.TrustError):
        TR.load_trust_store(p, SITE)
    crl = TR.make_crl(root, SITE, 1, NOW, [op.verify_key])
    p.write_text(TR.keyring_file_json(root.verify_key, SITE, [cert], crl), encoding="utf-8")
    with pytest.raises(TR.TrustError):
        TR.load_trust_store(p, SITE)                                   # the only cert is revoked by the file's CRL


def test_keyring_file_crl_refuses_only_the_revoked_certs(tmp_path):
    """Audit S3b #2: revoking a key by adding a CRL to the keyring file must drop that key only, not the whole file
    (which would stop the robot until someone also edits the cert list)."""
    root, op = _keys()
    _, op2 = _keys()
    good = TR.issue_cert(root, op.verify_key, SITE, NOW, NOW + 100)
    bad = TR.issue_cert(root, op2.verify_key, SITE, NOW, NOW + 100)
    crl = TR.make_crl(root, SITE, 1, NOW, [op2.verify_key])
    p = tmp_path / "keyring.json"
    p.write_text(TR.keyring_file_json(root.verify_key, SITE, [bad, good], crl), encoding="utf-8")
    st = TR.load_trust_store(p, SITE)
    assert set(st.keyring_at(NOW + 1)) == {TR.key_id_of(op.verify_key)}
    assert st.stats["cert_revoked_skipped"] == 1 and st.stats["cert_ok"] == 1


def test_keyring_file_revoked_but_invalid_cert_is_still_an_error(tmp_path):
    """Only a GENUINE cert of a revoked key is skipped: a forged / foreign-site cert is a provisioning error even when
    its key is on the CRL (fail closed)."""
    root, op = _keys()
    _, op2 = _keys()
    good = TR.issue_cert(root, op.verify_key, SITE, NOW, NOW + 100)
    crl = TR.make_crl(root, SITE, 1, NOW, [op2.verify_key])
    p = tmp_path / "keyring.json"
    for forged in (TR.issue_cert(TR.SigningKey.generate(), op2.verify_key, SITE, NOW, NOW + 100),   # other root
                   TR.issue_cert(root, op2.verify_key, SITE + 1, NOW, NOW + 100)):                 # other site
        p.write_text(TR.keyring_file_json(root.verify_key, SITE, [good, forged], crl), encoding="utf-8")
        with pytest.raises(TR.TrustError):
            TR.load_trust_store(p, SITE)


# ------------------------------------------------------------------------------------------------ signing-key files
def test_signing_key_file_roundtrip(tmp_path):
    sk = TR.SigningKey.generate()
    p = tmp_path / "observer.key"
    TR.write_signing_key(p, sk)
    assert TR.load_signing_key(p).encode() == sk.encode()
    with pytest.raises(FileExistsError):
        TR.write_signing_key(p, sk)                                    # never overwrites


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
def test_signing_key_group_or_world_readable_is_refused(tmp_path):
    sk = TR.SigningKey.generate()
    p = tmp_path / "observer.key"
    TR.write_signing_key(p, sk)
    assert (p.stat().st_mode & 0o777) == 0o600
    for mode in (0o640, 0o604, 0o660):
        os.chmod(p, mode)
        with pytest.raises(TR.KeyPermissionError):
            TR.load_signing_key(p)
    os.chmod(p, 0o400)
    assert TR.load_signing_key(p).encode() == sk.encode()


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL")
@pytest.mark.parametrize("sid", ["S-1-1-0", "S-1-5-11", "S-1-5-32-545"])     # Everyone, Authenticated Users, Users
def test_signing_key_shared_by_acl_is_refused_on_windows(tmp_path, sid):
    sk = TR.SigningKey.generate()
    p = tmp_path / "observer.key"
    TR.write_signing_key(p, sk)
    r = subprocess.run(["icacls", str(p), "/grant", f"*{sid}:R"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with pytest.raises(TR.KeyPermissionError, match=sid):
        TR.load_signing_key(p)


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL")
def test_signing_key_inherited_acl_is_checked_on_windows(tmp_path):
    """A key file written by hand inherits its folder's ACL: an inherited grant to Everyone is refused."""
    sk = TR.SigningKey.generate()
    d = tmp_path / "shared"
    d.mkdir()
    r = subprocess.run(["icacls", str(d), "/grant", "*S-1-1-0:(OI)(CI)R"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    q = d / "inherits_everyone.key"
    q.write_text(sk.encode().hex(), encoding="ascii")
    with pytest.raises(TR.KeyPermissionError, match="S-1-1-0"):
        TR.load_signing_key(q)
