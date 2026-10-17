"""Key provisioning for the signed broadcast (WEAKNESSES C-W5): keys, site-root certificates, revocation, keyring file.

    python scripts/keys.py keygen OUT.key
        New Ed25519 key in a private file (POSIX 0600 / Windows owner-only DACL; never overwrites). Prints only the
        PUBLIC "key_id:verify_key_hex" (key_id = sha256(vk)[:2]). Use it for the site ROOT key (kept offline) and for
        each operational (observer) key.
    python scripts/keys.py cert --root ROOT.key --vk [KID:]HEX --site N --nbf UNIX --exp UNIX [--serial S]
        Root-signed certificate for an operational verify key; prints the cert hex.
    python scripts/keys.py crl --root ROOT.key --site N --serial S [--issue-time UNIX] --vk [KID:]HEX [--vk ...]
        Root-signed revocation list (serial must rise with every new list); prints the CRL hex.
    python scripts/keys.py keyring --root-vk [KID:]HEX --site N --cert HEX [--cert ...] [--crl HEX] --out FILE
        Writes the robot's keyring file (public material only; observer_bridge keyring_file:=FILE). The file is
        checked with codec.trust.load_trust_store before it is put in place, and an existing file is never replaced.

Nothing here prints or logs a private key. Library: intercycle/codec/trust.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intercycle.codec import trust as TR  # noqa: E402


def _vk(spec: str) -> TR.VerifyKey:
    """'HEX' or 'KID:HEX' (as keygen prints it); a KID that is not key_id_of(vk) is refused."""
    kid, _, h = spec.rpartition(":")
    try:
        vk = TR.VerifyKey(bytes.fromhex(h))
    except ValueError as e:
        raise TR.TrustError(f"bad verify key {spec!r}: {e}") from None
    if kid and int(kid) != TR.key_id_of(vk):
        raise TR.TrustError(f"key_id {kid} is not derived from the key (expected {TR.key_id_of(vk)})")
    return vk


def _line(vk: TR.VerifyKey) -> str:
    return f"{TR.key_id_of(vk)}:{vk.encode().hex()}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("keygen")
    g.add_argument("out")
    c = sub.add_parser("cert")
    c.add_argument("--root", required=True)
    c.add_argument("--vk", required=True)
    c.add_argument("--site", type=int, required=True)
    c.add_argument("--nbf", type=int, required=True)
    c.add_argument("--exp", type=int, required=True)
    c.add_argument("--serial", type=int, default=0)
    r = sub.add_parser("crl")
    r.add_argument("--root", required=True)
    r.add_argument("--site", type=int, required=True)
    r.add_argument("--serial", type=int, required=True)
    r.add_argument("--issue-time", type=int, default=None)
    r.add_argument("--vk", action="append", default=[])
    k = sub.add_parser("keyring")
    k.add_argument("--root-vk", required=True)
    k.add_argument("--site", type=int, required=True)
    k.add_argument("--cert", action="append", required=True)
    k.add_argument("--crl", default=None)
    k.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "keygen":
        sk = TR.SigningKey.generate()
        TR.write_signing_key(a.out, sk)
        print(_line(sk.verify_key))
    elif a.cmd == "cert":
        root = TR.load_signing_key(a.root)
        print(TR.issue_cert(root, _vk(a.vk), a.site, a.nbf, a.exp, a.serial).hex())
    elif a.cmd == "crl":
        root = TR.load_signing_key(a.root)
        issue = a.issue_time if a.issue_time is not None else int(time.time())
        print(TR.make_crl(root, a.site, a.serial, issue, [_vk(v) for v in a.vk]).hex())
    else:
        out = Path(a.out)
        if out.exists():
            raise FileExistsError(f"{out} exists; write a new file and swap it in deliberately")
        try:
            certs = [bytes.fromhex(x) for x in a.cert]
            crl = bytes.fromhex(a.crl) if a.crl else None
        except ValueError as e:
            raise TR.TrustError(f"bad hex: {e}") from None
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(TR.keyring_file_json(_vk(a.root_vk), a.site, certs, crl), encoding="utf-8")
        try:
            st = TR.load_trust_store(tmp, a.site)               # the robot's own check, before the file is in place
            os.rename(tmp, out)                                  # fails if `out` appeared meanwhile: never replaces
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        print(f"{out}: {len(st.certs)} cert(s) {sorted(st.certs)}; stats {dict(st.stats)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
