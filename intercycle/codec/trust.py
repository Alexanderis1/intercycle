"""Key management for the signed broadcast (WEAKNESSES C-W5): fail-closed keys, site-root certificates, revocation.

Trust model
  * Each site has an offline ROOT Ed25519 key. Its verify key is the trust anchor provisioned on every robot (in the
    keyring file below). The root key never signs broadcast layers; it signs only certificates and revocation lists.
  * An OPERATIONAL key (the observer's signing key) is trusted only through a root-signed certificate that binds
    (site_id, key_id, verify key, nbf, exp, serial). key_id is DERIVED from the key: key_id = first 2 bytes of
    sha256(verify key), big endian (`key_id_of`), so it is never a configured constant.
  * A root-signed revocation list (CRL: site_id, serial, issue_time, sha256 fingerprints of revoked verify keys) removes
    keys. A CRL is accepted only with a higher serial than the one held.
  * `TrustStore.keyring_at(now_unix)` is the keyring valid at that instant: certified, nbf <= now <= exp, not revoked.
    Anything else is not in it, so `sign.verify` rejects it ("unknown_key"): every check fails closed.

Wire formats (big endian; blob = body || Ed25519 signature by the root over body, 64 B)
  cert body (66 B): b"ICC1" | site_id u32 | key_id u16 | nbf u64 | exp u64 | serial u32 | verify key 32 B
  crl body:         b"ICR1" | site_id u32 | serial u32 | issue_time u64 | n u16 | n x sha256(verify key) 32 B

Keyring file (JSON, provisioned read-only on the robot; it holds only PUBLIC material):
  {"format": "intercycle-keyring/1", "site_id": <int>, "root_vk": "<hex>", "certs": ["<hex blob>", ...], "crl": "<hex>"|null}

Signing-key files (`load_signing_key` / `write_signing_key`) hold a 32-byte Ed25519 seed as 64 hex characters and must be
private to the owner:
  * POSIX: the file is owned by the current user and its mode grants nothing to group/other (mode & 0o077 == 0, i.e.
    0600 or 0400), as ssh does for private keys.
  * Windows (no meaningful mode bits): the Windows-aware equivalent is the file's DACL. Every ACE that grants access and
    applies to the file itself must name the current user, LocalSystem (S-1-5-18) or BUILTIN\\Administrators
    (S-1-5-32-544) — the accounts that can read any file anyway, like root on POSIX. A NULL DACL (everyone), any other
    trustee (e.g. Everyone, Authenticated Users, Users) or an ACE type we cannot parse is refused. Deny ACEs and
    inherit-only ACEs are ignored (they grant nothing on this file). `write_signing_key` creates such a file by removing
    inheritance and granting Full control to the current user only (icacls, by SID so it is locale independent) before
    the seed is written.
Only the standard library and PyNaCl are used (Isaac dependency rule).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from intercycle.codec.sign import SIG_LEN, Keyring

__all__ = ["Cert", "Crl", "KeyPermissionError", "SigningKey", "TrustError", "TrustStore", "VerifyKey", "fingerprint",
           "issue_cert", "key_id_of", "keyring_file_json", "load_signing_key", "load_trust_store", "make_crl",
           "write_signing_key"]

CERT_MAGIC, CRL_MAGIC = b"ICC1", b"ICR1"
_CERT = struct.Struct(">4sIHQQI32s")
_CRL_HEAD = struct.Struct(">4sIIQH")
KEYRING_FORMAT = "intercycle-keyring/1"


class TrustError(ValueError):
    """A certificate, revocation list or keyring file that must not be trusted (the message says why)."""


class KeyPermissionError(PermissionError):
    """A signing-key file readable or writable by someone other than its owner (or root/SYSTEM/Administrators)."""


def fingerprint(vk: VerifyKey) -> bytes:
    return hashlib.sha256(vk.encode()).digest()


def key_id_of(vk: VerifyKey) -> int:
    """key_id = sha256(verify key)[:2] as a big-endian u16 (the 16-bit header field)."""
    return int.from_bytes(fingerprint(vk)[:2], "big")


# ------------------------------------------------------------------------------------------------ certificates / CRL
@dataclass(frozen=True)
class Cert:
    site_id: int
    key_id: int
    nbf: int
    exp: int
    serial: int
    vk: VerifyKey

    def valid_at(self, now_unix: float) -> bool:
        return self.nbf <= now_unix <= self.exp


@dataclass(frozen=True)
class Crl:
    site_id: int
    serial: int
    issue_time: int
    revoked: frozenset


def issue_cert(root: SigningKey, vk: VerifyKey, site_id: int, nbf: int, exp: int, serial: int = 0) -> bytes:
    if exp <= nbf:
        raise TrustError("exp must be after nbf")
    body = _CERT.pack(CERT_MAGIC, site_id, key_id_of(vk), int(nbf), int(exp), serial, vk.encode())
    return body + root.sign(body).signature


def make_crl(root: SigningKey, site_id: int, serial: int, issue_time: int, revoked) -> bytes:
    """revoked: VerifyKeys or 32-byte fingerprints."""
    fps = sorted({fingerprint(r) if isinstance(r, VerifyKey) else bytes(r) for r in revoked})
    if any(len(f) != 32 for f in fps):
        raise TrustError("a fingerprint is not 32 bytes")
    body = _CRL_HEAD.pack(CRL_MAGIC, site_id, serial, int(issue_time), len(fps)) + b"".join(fps)
    return body + root.sign(body).signature


def _root_verified(blob: bytes, root: VerifyKey, what: str) -> bytes:
    if len(blob) <= SIG_LEN:
        raise TrustError(f"{what}: malformed")
    body, sig = blob[:-SIG_LEN], blob[-SIG_LEN:]
    try:
        root.verify(body, sig)
    except BadSignatureError:
        raise TrustError(f"{what}: not signed by the site root") from None
    return body


def parse_cert(blob: bytes, root: VerifyKey) -> Cert:
    body = _root_verified(blob, root, "cert")
    if len(body) != _CERT.size:
        raise TrustError("cert: malformed")
    magic, site_id, key_id, nbf, exp, serial, raw = _CERT.unpack(body)
    if magic != CERT_MAGIC:
        raise TrustError("cert: bad magic")
    vk = VerifyKey(raw)
    if key_id != key_id_of(vk):
        raise TrustError("cert: key_id is not sha256(verify key)[:2]")
    if exp <= nbf:
        raise TrustError("cert: exp is not after nbf")
    return Cert(site_id, key_id, nbf, exp, serial, vk)


def parse_crl(blob: bytes, root: VerifyKey) -> Crl:
    body = _root_verified(blob, root, "crl")
    if len(body) < _CRL_HEAD.size:
        raise TrustError("crl: malformed")
    magic, site_id, serial, issue, n = _CRL_HEAD.unpack(body[:_CRL_HEAD.size])
    rest = body[_CRL_HEAD.size:]
    if magic != CRL_MAGIC or len(rest) != 32 * n:
        raise TrustError("crl: malformed")
    return Crl(site_id, serial, issue, frozenset(rest[i:i + 32] for i in range(0, len(rest), 32)))


class TrustStore:
    """Root-anchored set of operational keys for one site. `keyring_at(now)` is what the receivers verify against."""

    def __init__(self, root_vk: VerifyKey, site_id: int) -> None:
        self.root, self.site_id = root_vk, int(site_id)
        self.certs: dict[int, Cert] = {}
        self.revoked: frozenset = frozenset()
        self.crl_serial = -1
        self.stats: Counter[str] = Counter()

    def add_cert(self, blob: bytes) -> Cert:
        """Admit a root-signed cert for this site. Raises TrustError (and counts the reason) otherwise."""
        try:
            c = parse_cert(blob, self.root)
            if c.site_id != self.site_id:
                raise TrustError("cert: wrong site")
            if fingerprint(c.vk) in self.revoked:
                raise TrustError("cert: key is revoked")
            old = self.certs.get(c.key_id)
            if old is not None and old.vk.encode() != c.vk.encode():
                raise TrustError("cert: key_id collision with a different certified key")
        except TrustError as e:
            self.stats[str(e)] += 1
            raise
        if old is None or c.serial > old.serial:          # renewal of the same key: the newer cert wins
            self.certs[c.key_id] = c
        self.stats["cert_ok"] += 1
        return c

    def apply_crl(self, blob: bytes) -> Crl:
        try:
            crl = parse_crl(blob, self.root)
            if crl.site_id != self.site_id:
                raise TrustError("crl: wrong site")
            if crl.serial <= self.crl_serial:
                raise TrustError("crl: not newer than the one held")
        except TrustError as e:
            self.stats[str(e)] += 1
            raise
        self.revoked, self.crl_serial = crl.revoked, crl.serial
        self.certs = {k: c for k, c in self.certs.items() if fingerprint(c.vk) not in crl.revoked}
        self.stats["crl_ok"] += 1
        return crl

    def keyring_at(self, now_unix: float) -> Keyring:
        return {k: c.vk for k, c in self.certs.items() if c.valid_at(now_unix) and fingerprint(c.vk) not in self.revoked}


def keyring_file_json(root_vk: VerifyKey, site_id: int, certs: list[bytes], crl: bytes | None = None) -> str:
    return json.dumps({"format": KEYRING_FORMAT, "site_id": int(site_id), "root_vk": root_vk.encode().hex(),
                       "certs": [c.hex() for c in certs], "crl": crl.hex() if crl else None}, indent=1)


def load_trust_store(path: str | os.PathLike, site_id: int) -> TrustStore:
    """Read a keyring file. Every malformed field, a site mismatch, or any cert / CRL that fails to verify raises
    TrustError: a provisioning error must stop the robot, never degrade into trusting less-checked keys."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TrustError(f"keyring file {path}: {e}") from None
    if not isinstance(d, dict) or d.get("format") != KEYRING_FORMAT:
        raise TrustError(f"keyring file {path}: format is not {KEYRING_FORMAT}")
    if d.get("site_id") != int(site_id):
        raise TrustError(f"keyring file {path}: site_id {d.get('site_id')} != this site's {site_id}")
    try:
        root = VerifyKey(bytes.fromhex(d["root_vk"]))
        certs = [bytes.fromhex(c) for c in d.get("certs") or []]
        crl = bytes.fromhex(d["crl"]) if d.get("crl") else None
    except (KeyError, TypeError, ValueError) as e:
        raise TrustError(f"keyring file {path}: {e}") from None
    if not certs:
        raise TrustError(f"keyring file {path}: no certificates")
    st = TrustStore(root, site_id)
    if crl is not None:
        st.apply_crl(crl)
    for c in certs:
        st.add_cert(c)
    return st


# ------------------------------------------------------------------------------------------------ signing-key files
def check_key_file_private(path: str | os.PathLike) -> None:
    """Raise KeyPermissionError unless only the owner (plus root / SYSTEM / Administrators) can access the file."""
    p = os.fspath(path)
    if os.name == "nt":
        _check_windows_dacl(p)
        return
    st = os.stat(p)
    if st.st_uid != os.getuid():
        raise KeyPermissionError(f"{p}: owned by uid {st.st_uid}, not by the current user")
    if st.st_mode & 0o077:
        raise KeyPermissionError(f"{p}: mode {st.st_mode & 0o777:o} grants access to group/other (need 0600 or 0400)")


def load_signing_key(path: str | os.PathLike) -> SigningKey:
    """The observer's Ed25519 signing key from a private file (see the module docstring for the permission rule)."""
    check_key_file_private(path)
    txt = Path(path).read_text(encoding="ascii").strip()
    try:
        seed = bytes.fromhex(txt)
    except ValueError:
        raise TrustError(f"{os.fspath(path)}: not a hex seed") from None
    if len(seed) != 32:
        raise TrustError(f"{os.fspath(path)}: seed is {len(seed)} bytes, need 32")
    return SigningKey(seed)


def write_signing_key(path: str | os.PathLike, sk: SigningKey) -> None:
    """Create a NEW private key file (never overwrites): POSIX mode 0600; Windows DACL = current user only, set while
    the file is still empty."""
    p = os.fspath(path)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    try:
        if os.name == "nt":
            sid = _windows_user_sid()
            r = subprocess.run(["icacls", p, "/inheritance:r", "/grant:r", f"*{sid}:F"], capture_output=True, text=True)
            if r.returncode != 0:
                raise KeyPermissionError(f"{p}: icacls failed ({r.returncode})")
            _check_windows_dacl(p)
        os.write(fd, sk.encode().hex().encode("ascii") + b"\n")
    except BaseException:
        os.close(fd)
        os.unlink(p)
        raise
    os.close(fd)


# ---- Windows DACL (ctypes; no pywin32 dependency)
_SYSTEM_SID, _ADMINS_SID = "S-1-5-18", "S-1-5-32-544"
_DENY_TYPES = {1, 6, 10, 12}           # ACCESS_DENIED(_OBJECT)(_CALLBACK)(_OBJECT): grant nothing
_SIMPLE_ALLOW_TYPES = {0, 9}           # ACCESS_ALLOWED, ACCESS_ALLOWED_CALLBACK: SID at offset 8
_INHERIT_ONLY_ACE = 0x08


def _win():
    import ctypes
    from ctypes import wintypes

    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    adv.GetNamedSecurityInfoW.argtypes = [wintypes.LPCWSTR, ctypes.c_int, wintypes.DWORD, ctypes.c_void_p,
                                          ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_void_p)]
    adv.GetNamedSecurityInfoW.restype = wintypes.DWORD
    adv.GetAce.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    adv.GetAce.restype = wintypes.BOOL
    adv.GetAclInformation.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.c_int]
    adv.GetAclInformation.restype = wintypes.BOOL
    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    adv.ConvertSidToStringSidW.restype = wintypes.BOOL
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    adv.OpenProcessToken.restype = wintypes.BOOL
    adv.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD)]
    adv.GetTokenInformation.restype = wintypes.BOOL
    k32.LocalFree.argtypes = [ctypes.c_void_p]
    k32.LocalFree.restype = ctypes.c_void_p
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return ctypes, wintypes, adv, k32


def _sid_str(ctypes, wintypes, adv, k32, psid) -> str:
    s = wintypes.LPWSTR()
    if not adv.ConvertSidToStringSidW(psid, ctypes.byref(s)):
        raise KeyPermissionError(f"ConvertSidToStringSid failed ({ctypes.get_last_error()})")
    try:
        return str(s.value)
    finally:
        k32.LocalFree(s)


def _windows_user_sid() -> str:
    ctypes, wintypes, adv, k32 = _win()
    h = wintypes.HANDLE()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x0008, ctypes.byref(h)):       # TOKEN_QUERY
        raise KeyPermissionError(f"OpenProcessToken failed ({ctypes.get_last_error()})")
    try:
        n = wintypes.DWORD()
        adv.GetTokenInformation(h, 1, None, 0, ctypes.byref(n))                         # TokenUser: size query
        buf = ctypes.create_string_buffer(n.value)
        if not adv.GetTokenInformation(h, 1, buf, n, ctypes.byref(n)):
            raise KeyPermissionError(f"GetTokenInformation failed ({ctypes.get_last_error()})")
        psid = ctypes.c_void_p.from_buffer(buf).value                                  # TOKEN_USER.User.Sid
        return _sid_str(ctypes, wintypes, adv, k32, psid)
    finally:
        k32.CloseHandle(h)


def _check_windows_dacl(path: str) -> None:
    ctypes, wintypes, adv, k32 = _win()
    allowed = {_windows_user_sid(), _SYSTEM_SID, _ADMINS_SID}
    dacl, sd = ctypes.c_void_p(), ctypes.c_void_p()
    err = adv.GetNamedSecurityInfoW(path, 1, 0x4, None, None, ctypes.byref(dacl), None, ctypes.byref(sd))  # FILE, DACL
    if err:
        raise KeyPermissionError(f"{path}: cannot read the security descriptor (error {err})")
    try:
        if not dacl.value:
            raise KeyPermissionError(f"{path}: NULL DACL (everyone has full access)")

        class AclSize(ctypes.Structure):
            _fields_ = [("AceCount", wintypes.DWORD), ("AclBytesInUse", wintypes.DWORD), ("AclBytesFree", wintypes.DWORD)]

        info = AclSize()
        if not adv.GetAclInformation(dacl, ctypes.byref(info), ctypes.sizeof(info), 2):     # AclSizeInformation
            raise KeyPermissionError(f"{path}: GetAclInformation failed ({ctypes.get_last_error()})")
        bad = []
        for i in range(info.AceCount):
            ace = ctypes.c_void_p()
            if not adv.GetAce(dacl, i, ctypes.byref(ace)):
                raise KeyPermissionError(f"{path}: GetAce failed ({ctypes.get_last_error()})")
            assert ace.value is not None
            ace_type, ace_flags = ctypes.c_ubyte.from_address(ace.value).value, ctypes.c_ubyte.from_address(ace.value + 1).value
            if ace_type in _DENY_TYPES or ace_flags & _INHERIT_ONLY_ACE:
                continue
            if ace_type not in _SIMPLE_ALLOW_TYPES:
                raise KeyPermissionError(f"{path}: unsupported ACE type {ace_type} (refused)")
            sid = _sid_str(ctypes, wintypes, adv, k32, ace.value + 8)
            if sid not in allowed:
                bad.append(sid)
        if bad:
            raise KeyPermissionError(f"{path}: access granted to {sorted(set(bad))}; only the owner, SYSTEM and "
                                     "Administrators may hold the signing key (Windows equivalent of mode 0600)")
    finally:
        k32.LocalFree(sd)
