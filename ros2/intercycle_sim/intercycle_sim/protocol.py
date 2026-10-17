"""Lockstep wire protocol between the simulator process (Isaac, python 3.11) and the ROS 2 bridge node (python 3.10).

Transport: one Unix-domain stream socket (local only).  The ROS bridge node listens; the simulator connects.
Each message is a 4-byte big-endian length followed by a UTF-8 JSON object.  Standard library only, so both
interpreters import this same file.

Lockstep, once per physics step of the policy (50 Hz for ANYmal-D):
  sim  -> ros   {"type": "frame", "t": sim_time_s, "imu": {...}, "odom": {...}, "scan": {...} (only on scan ticks),
                 "contacts": [...] (optional)}
  ros  -> sim   {"type": "cmd", "vx": .., "vy": .., "wz": .., "age": seconds since the last /cmd_vel (sim time)}
The simulator does not advance until it has the reply, so ROS sees a gap-free /clock and Nav2's latest command is
applied on the next step.  The frame NEVER carries ground truth: only the noisy sensor outputs of `sensors.py`.
"""
from __future__ import annotations

import json
import socket
import struct
import time

_HDR = struct.Struct(">I")
MAX_MSG = 16 * 1024 * 1024


def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj, separators=(",", ":"), allow_nan=True).encode("utf-8")
    sock.sendall(_HDR.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the bridge socket")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict:
    (n,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    if n > MAX_MSG:
        raise ValueError(f"bridge message too large: {n} bytes")
    return json.loads(_recv_exact(sock, n).decode("utf-8"))


def connect(path: str, timeout_s: float = 600.0) -> socket.socket:
    """Connect to the bridge node's socket, retrying until it exists (the ROS side may start later)."""
    t0 = time.time()
    while True:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(path)
            return s
        except (FileNotFoundError, ConnectionRefusedError):
            s.close()
            if time.time() - t0 > timeout_s:
                raise
            time.sleep(0.2)


def listen(path: str) -> socket.socket:
    import os

    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(path)
    s.listen(1)
    return s


class SimClient:
    """Simulator side of the lockstep: `exchange(frame)` sends one frame and blocks for the command reply."""

    def __init__(self, path: str, timeout_s: float = 600.0):
        self.sock = connect(path, timeout_s)
        self.cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "age": 1e9}

    def exchange(self, frame: dict) -> dict:
        frame = dict(frame, type="frame")
        send_msg(self.sock, frame)
        msg = recv_msg(self.sock)
        if msg.get("type") == "cmd":
            self.cmd = msg
        return self.cmd

    def close(self) -> None:
        try:
            send_msg(self.sock, {"type": "bye"})
        except OSError:
            pass
        self.sock.close()
