"""Transport for the observer broadcast (transport-agnostic; no ROS, stdlib only).

The bridge never cares how bytes arrive. A `Transport` yields raw *records*; the bridge verifies everything itself
(Ed25519 via intercycle.codec), so a transport is untrusted by design: junk, replays and forgeries are allowed through and
rejected downstream.

Record = 1 kind byte + data:
  b"F" + frame   one BLE extended-advertising frame (`codec.ble.frame` fragment of a slow/fast/live blob, or a map-layer
                 manifest fragment / chunk from `codec.maplayer`)  <- what BLE (and the local episode feed) carries
  b"B" + blob    one whole signed blob (payload || 64 B signature) of the slow/fast/live layer  <- what Zenoh would carry

Transports here:
  unix:///path   Unix datagram socket; the bridge binds it, the broadcaster sends one record per datagram (episode feed)
  file:///path   append-only file of u16-BE-length-prefixed records; the bridge tails it (replays, debugging)
  memory         in-process queue (tests)
  none           no feed (condition "none" / "static")
Zenoh and BLE (workstream C, Sunday) plug in as further `Transport` subclasses.
"""
from __future__ import annotations

import collections
import os
import socket
import struct

FRAME, BLOB = b"F", b"B"
MAX_DGRAM = 4096


class Transport:
    """Non-blocking source of records. `poll()` returns every record that arrived since the last call."""

    def poll(self) -> list[bytes]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullTransport(Transport):
    def poll(self) -> list[bytes]:
        return []


class MemoryTransport(Transport):
    def __init__(self) -> None:
        self.q: collections.deque[bytes] = collections.deque()

    def push(self, record: bytes) -> None:
        self.q.append(record)

    def poll(self) -> list[bytes]:
        out = list(self.q)
        self.q.clear()
        return out


class UnixDgramTransport(Transport):
    """Receiver side: binds `path` (removing a stale socket file) and drains it without blocking."""

    def __init__(self, path: str) -> None:
        self.path = path
        if os.path.exists(path):
            os.unlink(path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sock.bind(path)
        self.sock.setblocking(False)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        except OSError:
            pass

    def poll(self, max_n: int = 10000) -> list[bytes]:
        out = []
        for _ in range(max_n):
            try:
                out.append(self.sock.recv(MAX_DGRAM))
            except (BlockingIOError, InterruptedError):
                break
        return out

    def close(self) -> None:
        self.sock.close()
        try:
            os.unlink(self.path)
        except OSError:
            pass


class FileTailTransport(Transport):
    """Tails an append-only record file (u16 BE length + record). Tolerates a file that does not exist yet."""

    def __init__(self, path: str) -> None:
        self.path, self.pos, self.buf = path, 0, b""

    def poll(self) -> list[bytes]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read()
        except FileNotFoundError:
            return []
        self.pos += len(data)
        self.buf += data
        out = []
        while len(self.buf) >= 2:
            n = struct.unpack_from(">H", self.buf)[0]
            if len(self.buf) < 2 + n:
                break
            out.append(self.buf[2:2 + n])
            self.buf = self.buf[2 + n:]
        return out


def open_transport(url: str) -> Transport:
    if not url or url == "none":
        return NullTransport()
    if url == "memory":
        return MemoryTransport()
    if url.startswith("unix://"):
        return UnixDgramTransport(url[len("unix://"):])
    if url.startswith("file://"):
        return FileTailTransport(url[len("file://"):])
    raise ValueError(f"unknown transport {url!r} (unix:///path | file:///path | memory | none)")


# ------------------------------------------------------------------------------------------------ sender side
class FeedWriter:
    """Broadcaster side of the local episode feed (used by the episode's broadcaster process, never by the robot).

    FeedWriter("unix:///tmp/ic/observer.sock").frames(list_of_ble_frames) / .blob(signed_blob).
    Send errors (bridge not up yet, buffer full) are counted and dropped: the broadcast is a carousel, like the radio."""

    def __init__(self, url: str) -> None:
        self.url, self.sent, self.dropped = url, 0, 0
        if url.startswith("unix://"):
            self.path = url[len("unix://"):]
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            self.f = None
        elif url.startswith("file://"):
            self.sock = None
            self.f = open(url[len("file://"):], "ab")
        else:
            raise ValueError(f"unknown feed url {url!r}")

    def record(self, rec: bytes) -> bool:
        try:
            if self.sock is not None:
                self.sock.sendto(rec, self.path)
            else:
                self.f.write(struct.pack(">H", len(rec)) + rec)
                self.f.flush()
            self.sent += 1
            return True
        except OSError:
            self.dropped += 1
            return False

    def frames(self, frames) -> int:
        return sum(self.record(FRAME + bytes(fr)) for fr in frames)

    def blob(self, blob: bytes) -> bool:
        return self.record(BLOB + bytes(blob))

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
        if self.f is not None:
            self.f.close()
