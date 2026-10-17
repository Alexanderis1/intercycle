"""Wire format, Ed25519 signing and BLE transport for the two broadcast layers. Never imports ground truth."""
from __future__ import annotations

from nacl.signing import SigningKey

from intercycle.codec.ble import BleChannel, Receiver, Reassembler, frame, parse_frame
from intercycle.codec.layout import (CodecError, decode_fast, decode_slow, encode_fast, encode_slow,
                                     quantize_fast, quantize_slow)
from intercycle.codec.sign import Keyring, demo_key, keyring_from, sign, verify
from intercycle.types import FastLayer, SlowLayer

__all__ = ["BleChannel", "CodecError", "Keyring", "Reassembler", "Receiver", "decode_fast", "decode_slow", "demo_key",
           "encode_fast", "encode_slow", "frame", "keyring_from", "parse_frame", "quantize_fast", "quantize_slow",
           "seal", "sign", "verify"]


def seal(layer: SlowLayer | FastLayer, signing_key: SigningKey) -> bytes:
    """Encode + sign one layer -> blob ready for frame()."""
    return sign(encode_slow(layer) if isinstance(layer, SlowLayer) else encode_fast(layer), signing_key)
