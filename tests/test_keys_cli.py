"""scripts/keys.py: provisioning CLI (keygen, cert, crl, keyring) end to end through codec.trust (C-W5)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from intercycle.codec import trust as TR

SITE = 0x4A3D11
NOW = 1_800_000_000
_SPEC = importlib.util.spec_from_file_location("keys_cli", Path(__file__).resolve().parents[1] / "scripts" / "keys.py")
assert _SPEC is not None and _SPEC.loader is not None
keys = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(keys)


def _run(capsys, *argv) -> str:
    assert keys.main(list(argv)) == 0
    return capsys.readouterr().out.strip()


def test_keygen_prints_only_public_material(tmp_path, capsys):
    out = _run(capsys, "keygen", str(tmp_path / "op.key"))
    sk = TR.load_signing_key(tmp_path / "op.key")
    kid, vk = out.split(":")
    assert int(kid) == TR.key_id_of(sk.verify_key) and vk == sk.verify_key.encode().hex()
    assert sk.encode().hex() not in out                                     # never prints the private seed
    with pytest.raises(FileExistsError):
        keys.main(["keygen", str(tmp_path / "op.key")])                     # never overwrites a key


def test_issue_revoke_and_keyring_end_to_end(tmp_path, capsys):
    root_line = _run(capsys, "keygen", str(tmp_path / "root.key"))
    a_line = _run(capsys, "keygen", str(tmp_path / "a.key"))
    b_line = _run(capsys, "keygen", str(tmp_path / "b.key"))
    ca = _run(capsys, "cert", "--root", str(tmp_path / "root.key"), "--vk", a_line, "--site", str(SITE),
              "--nbf", str(NOW), "--exp", str(NOW + 3600), "--serial", "1")
    cb = _run(capsys, "cert", "--root", str(tmp_path / "root.key"), "--vk", b_line.split(":")[1], "--site", str(SITE),
              "--nbf", str(NOW), "--exp", str(NOW + 3600), "--serial", "2")
    kr = tmp_path / "keyring.json"
    _run(capsys, "keyring", "--root-vk", root_line, "--site", str(SITE), "--cert", ca, "--cert", cb, "--out", str(kr))
    st = TR.load_trust_store(kr, SITE)
    kid_a, kid_b = int(a_line.split(":")[0]), int(b_line.split(":")[0])
    assert set(st.keyring_at(NOW + 1)) == {kid_a, kid_b}
    # revoke A: a CRL added to the same keyring file drops A only
    crl = _run(capsys, "crl", "--root", str(tmp_path / "root.key"), "--site", str(SITE), "--serial", "1",
               "--issue-time", str(NOW + 10), "--vk", a_line)
    kr2 = tmp_path / "keyring2.json"
    _run(capsys, "keyring", "--root-vk", root_line, "--site", str(SITE), "--cert", ca, "--cert", cb, "--crl", crl,
         "--out", str(kr2))
    assert set(TR.load_trust_store(kr2, SITE).keyring_at(NOW + 11)) == {kid_b}
    assert json.loads(kr2.read_text(encoding="utf-8"))["format"] == TR.KEYRING_FORMAT


def test_keyring_refuses_bad_input_and_never_overwrites(tmp_path, capsys):
    root_line = _run(capsys, "keygen", str(tmp_path / "root.key"))
    a_line = _run(capsys, "keygen", str(tmp_path / "a.key"))
    other = _run(capsys, "keygen", str(tmp_path / "other_root.key"))
    forged = _run(capsys, "cert", "--root", str(tmp_path / "other_root.key"), "--vk", a_line, "--site", str(SITE),
                  "--nbf", str(NOW), "--exp", str(NOW + 60))
    kr = tmp_path / "keyring.json"
    with pytest.raises(TR.TrustError):                                      # cert not signed by this root
        keys.main(["keyring", "--root-vk", root_line, "--site", str(SITE), "--cert", forged, "--out", str(kr)])
    assert not kr.exists()
    good = _run(capsys, "cert", "--root", str(tmp_path / "root.key"), "--vk", a_line, "--site", str(SITE),
                "--nbf", str(NOW), "--exp", str(NOW + 60))
    _run(capsys, "keyring", "--root-vk", root_line, "--site", str(SITE), "--cert", good, "--out", str(kr))
    with pytest.raises(FileExistsError):
        keys.main(["keyring", "--root-vk", other, "--site", str(SITE), "--cert", good, "--out", str(kr)])
    with pytest.raises(TR.TrustError):                                      # exp must be after nbf
        keys.main(["cert", "--root", str(tmp_path / "root.key"), "--vk", a_line, "--site", str(SITE),
                   "--nbf", str(NOW), "--exp", str(NOW)])
