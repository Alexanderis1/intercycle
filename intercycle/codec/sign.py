"""Ed25519 (PyNaCl) sign/verify over the encoded layer payload: blob = payload || signature(64)."""
from __future__ import annotations

import hashlib

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from intercycle.codec.layout import HDR_LEN, CodecError, peek_header

SIG_LEN = 64
Keyring = dict[int, VerifyKey]


def demo_key(seed: int = 0) -> SigningKey:
    """Deterministic key from a seed. TEST/DEMO ONLY: the seed is public, so the key is not secret."""
    return SigningKey(hashlib.sha256(b"intercycle-demo-key/%d" % seed).digest())


def keyring_from(signing_keys: dict[int, SigningKey]) -> Keyring:
    return {k: sk.verify_key for k, sk in signing_keys.items()}


def sign(payload: bytes, signing_key: SigningKey) -> bytes:
    return payload + signing_key.sign(payload).signature


def verify(blob: bytes, keyring: Keyring) -> tuple[bool, bytes, int, str]:
    """(ok, payload, key_id, reason). reason in {"", "malformed", "unknown_key", "bad_signature"}.
    payload is returned only when ok; key_id is -1 if the header could not be parsed."""
    if len(blob) < HDR_LEN + SIG_LEN:
        return False, b"", -1, "malformed"
    payload, sig = blob[:-SIG_LEN], blob[-SIG_LEN:]
    try:
        key_id = peek_header(payload).key_id
    except CodecError:
        return False, b"", -1, "malformed"
    vk = keyring.get(key_id)
    if vk is None:
        return False, b"", key_id, "unknown_key"
    try:
        vk.verify(payload, sig)
    except BadSignatureError:
        return False, b"", key_id, "bad_signature"
    return True, payload, key_id, ""
