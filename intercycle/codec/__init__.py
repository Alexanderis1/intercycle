"""Wire format, Ed25519 signing and BLE transport for the three broadcast layers (slow, fast, live). Never imports ground truth."""
from __future__ import annotations

from nacl.signing import SigningKey

from intercycle.codec.ble import BleChannel, Receiver, Reassembler, frame, parse_frame
from intercycle.codec.layout import (CodecError, decode_fast, decode_live, decode_slow, encode_fast, encode_live,
                                     encode_slow, quantize_fast, quantize_live, quantize_slow)
from intercycle.codec.sign import Keyring, demo_key, keyring_from, sign, verify
from intercycle.types import FastLayer, LiveLayer, SlowLayer

__all__ = ["BleChannel", "CodecError", "Keyring", "Reassembler", "Receiver", "decode_fast", "decode_live", "decode_slow",
           "demo_key", "encode_fast", "encode_live", "encode_slow", "frame", "keyring_from", "parse_frame", "quantize_fast",
           "quantize_live", "quantize_slow", "seal", "sign", "verify"]


def seal(layer: SlowLayer | FastLayer | LiveLayer, signing_key: SigningKey) -> bytes:
    """Encode + sign one layer -> blob ready for frame()."""
    enc = encode_slow if isinstance(layer, SlowLayer) else encode_live if isinstance(layer, LiveLayer) else encode_fast
    return sign(enc(layer), signing_key)
