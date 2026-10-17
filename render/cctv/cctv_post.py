"""CCTV look for clean Blender frames, with ground truth kept aligned.

Pipeline per frame (all parametrised by :class:`CctvParams`):

  a. barrel distortion (+ zoom so no black corners), vignetting, lateral
     chromatic aberration                      -> one sampling map, also used for GT
  c. lens softness, exposure / white balance, Poisson-Gaussian noise in linear
     light, colour noise in shadows, CCTV sharpening halo
  d. burned-in overlay (timestamp, camera name, SYNTHETIC label)
  e. real H.264 (libx264) through ffmpeg: a clip encode/decode, or a one-frame
     encode/decode round trip for stills

Python API::

    base = camera_params("CAM03", seed=0, cam_label="BODY SHOP S30")
    p = frame_params(base, frame_idx, seed=0)
    img, gt = process_frame(img_rgb_u8, p, t=frame_idx / 25, gt=GtFrame(...))
    process_sequence(frames_dir, out_dir, base, fps=25, seed=0, gt_dir=..., video=True)

CLI::

    python -m render.cctv.cctv_post --in <dir|png> --out <dir> --cam CAM03 \
        --cam-label "BODY SHOP S30" --t0 "2026-10-17 10:00:30" --fps 25 --seed 0 \
        [--gt <dir>] [--video] [--size 1920x1080]
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import functools
import json
import math
import re
import shutil
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .gt import GtFrame, load_gt, save_gt, warp_gt, write_coco

# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------


@dataclass
class CctvParams:
    # identity / overlay
    cam_id: str = "CAM01"
    cam_label: str = "BODY SHOP"
    t0: str = "2026-10-17 10:00:30"  # wall-clock time of t = 0
    overlay: bool = True
    # output geometry
    size: tuple = (1920, 1080)  # (W, H)
    overscan: float = 0.0  # fraction of the input removed on EACH side to get the nominal view
    # a. optics (fixed per camera). r is normalised by RADIUS_NORM x the output half-diagonal.
    k1: float = -0.25  # x_d = x_u (1 + k1 r_u^2 + k2 r_u^4)   (OpenCV convention, see RADIUS_NORM)
    k2: float = 0.0
    vignette: float = 0.18  # linear-light gain loss at the corner (0.18 = -0.29 stops)
    ca: float = 0.0010  # lateral CA: R/B radial scale = 1 +/- ca * r^2 (r = 1 at corner)
    # c. sensor / ISP (fixed per camera)
    gain: float = 2.5  # analog gain multiplier (noise grows with it)
    noise_a: float = 1.0 / 1100.0  # shot-noise variance per unit linear signal at gain 1
    read_noise: float = 0.0015  # read-noise sigma (linear units) at gain 1
    noise_corr: float = 0.6  # spatial correlation (demosaic/ISP) sigma in px
    chroma_noise: float = 0.014  # extra low-frequency colour noise, shadow weighted (sRGB units)
    soften_sigma: float = 0.55  # lens softness (px, output)
    sharpen_amount: float = 0.7
    sharpen_sigma: float = 1.2
    wb: tuple = (1.0, 1.0)  # (r gain, b gain), static part
    exposure: float = 1.0  # static exposure multiplier (linear)
    # e. compression (per clip)
    crf: int = 32
    keyint: int = 30
    preset: str = "veryfast"
    # GT
    min_box_pixels: int = 24
    # per-frame fields (set by frame_params; neutral by default)
    f_exposure: float = 1.0
    f_wb: tuple = (1.0, 1.0)
    f_shift: tuple = (0.0, 0.0)  # sub-pixel camera shake in OUTPUT px (applied to image and GT)
    f_noise_seed: int = 0
    # randomisation amplitudes for frame_params
    ae_flicker: float = 0.025
    wb_drift: float = 0.008
    shake_px: float = 0.35


def _cam_rng(cam_id: str, seed: int, *extra) -> np.random.Generator:
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(cam_id.encode()), *[int(e) & 0xFFFFFFFF for e in extra]])


def camera_params(cam_id: str, seed: int = 0, cam_label: str = "BODY SHOP", t0: str = "2026-10-17 10:00:30",
                  size: tuple = (1920, 1080), overscan: float = 0.0, **overrides) -> CctvParams:
    """Fixed per-camera parameters, deterministic for (cam_id, seed). ``overrides`` replace any field."""
    r = _cam_rng(cam_id, seed, 1)
    u = r.uniform
    p = CctvParams(
        cam_id=cam_id,
        cam_label=cam_label,
        t0=t0,
        size=tuple(size),
        overscan=overscan,
        k1=float(u(-0.35, -0.20)),
        k2=float(u(-0.03, 0.03)),
        vignette=float(u(0.12, 0.25)),
        ca=float(u(0.0006, 0.0016)),
        gain=float(math.exp(u(math.log(1.5), math.log(5.0)))),
        soften_sigma=float(u(0.45, 0.8)),
        sharpen_amount=float(u(0.5, 1.0)),
        sharpen_sigma=float(u(1.0, 1.6)),
        wb=(float(1 + u(-0.03, 0.03)), float(1 + u(-0.03, 0.03))),
        exposure=float(u(0.95, 1.05)),
        crf=int(r.integers(28, 37)),
        keyint=int(r.integers(25, 51)),
    )
    return dataclasses.replace(p, **overrides) if overrides else p


def frame_params(base: CctvParams, frame_idx: int, seed: int = 0) -> CctvParams:
    """Small per-frame randomisation (AE flicker, WB drift, shake, noise seed). Deterministic."""
    ph = _cam_rng(base.cam_id, seed, 2)  # slow, per-camera phases -> temporally smooth drift
    period_ae = ph.uniform(40, 160)
    phase_ae = ph.uniform(0, 2 * math.pi)
    period_wb = ph.uniform(300, 900, size=2)
    phase_wb = ph.uniform(0, 2 * math.pi, size=2)
    fr = _cam_rng(base.cam_id, seed, 3, frame_idx)  # white part, per frame
    ae = 1.0 + base.ae_flicker * math.sin(2 * math.pi * frame_idx / period_ae + phase_ae) + 0.2 * base.ae_flicker * fr.standard_normal()
    wbr = 1.0 + base.wb_drift * math.sin(2 * math.pi * frame_idx / period_wb[0] + phase_wb[0]) + 0.15 * base.wb_drift * fr.standard_normal()
    wbb = 1.0 + base.wb_drift * math.sin(2 * math.pi * frame_idx / period_wb[1] + phase_wb[1]) + 0.15 * base.wb_drift * fr.standard_normal()
    shift = (float(fr.normal(0, base.shake_px / 2.0)), float(fr.normal(0, base.shake_px / 2.0)))
    shift = (max(-base.shake_px, min(base.shake_px, shift[0])), max(-base.shake_px, min(base.shake_px, shift[1])))
    return dataclasses.replace(base, f_exposure=float(ae), f_wb=(float(wbr), float(wbb)), f_shift=shift,
                               f_noise_seed=int(fr.integers(0, 2**31 - 1)))


def display_cam(cam_id: str) -> str:
    """'CAM03' -> 'CAM 03'."""
    m = re.fullmatch(r"([A-Za-z]+)[ _-]?(\d+)", cam_id.strip())
    return f"{m.group(1).upper()} {m.group(2)}" if m else cam_id.upper()


def recommended_overscan(k1: float, k2: float = 0.0, size=(1920, 1080)) -> float:
    """Overscan fraction (per side) that makes the auto zoom 1.0, i.e. no crop of the nominal view."""
    tab = _inverse_table(float(k1), float(k2))
    rd = min(1.0 / RADIUS_NORM, 0.98 * float(tab[0][-1]))  # distorted radius of the output corner
    ratio = float(np.interp(rd, tab[0], tab[1])) / rd  # source / output radius there
    return max(0.0, (1.0 - 1.0 / ratio) / 2.0) if ratio > 1.0 else 0.0


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------


# Distortion radius r is |offset from centre| / (RADIUS_NORM * output half-diagonal), so r = 1/RADIUS_NORM at the
# corner. 1.7 ~ a 55 deg horizontal FOV in OpenCV's focal-length units; it keeps k1 in -0.2..-0.35 invertible.
RADIUS_NORM = 1.7


@dataclass
class Geometry:
    map_x: np.ndarray  # (H, W) float32, source x for the green channel / GT (pixel centres = integers)
    map_y: np.ndarray
    rad: np.ndarray  # (H, W) float32, output radius, 1 at the corner
    zoom: float  # fraction of the nominal distorted view kept (1.0 = no crop)
    scale: float  # source px per output px at the centre
    ca_maps: Optional[tuple]  # (rx, ry, bx, by) or None


@functools.lru_cache(maxsize=16)
def _inverse_table(k1: float, k2: float):
    ru = np.linspace(0.0, 3.0, 6001)
    rd = ru * (1.0 + k1 * ru**2 + k2 * ru**4)
    bad = np.nonzero(np.diff(rd) <= 0)[0]
    n = (bad[0] + 1) if bad.size else rd.size
    return rd[:n].copy(), ru[:n].copy()


def _src_coords(px, py, s, tab, d_src, cx, cy):
    rd = s * np.hypot(px, py) / RADIUS_NORM
    with np.errstate(invalid="ignore", divide="ignore"):
        ru = np.interp(rd, tab[0], tab[1], right=np.inf)
        ratio = np.where(rd > 1e-9, ru / np.maximum(rd, 1e-9), 1.0)
        sx = cx + s * px * ratio * d_src
        sy = cy + s * py * ratio * d_src
    return sx, sy


@functools.lru_cache(maxsize=8)
def build_geometry(src_wh: tuple, out_wh: tuple, overscan: float, k1: float, k2: float, ca: float,
                   margin_px: float = 2.0) -> Geometry:
    """Sampling map output->source. Zoom is chosen as the largest view with no black corners."""
    wi, hi = src_wh
    w, h = out_wh
    d_out = math.hypot(w, h) / 2.0
    d_src = (1.0 - 2.0 * overscan) * math.hypot(wi, hi) / 2.0
    cx, cy = (wi - 1) / 2.0, (hi - 1) / 2.0
    tab = _inverse_table(k1, k2)
    scale0 = d_src / d_out
    m = margin_px + abs(ca) * d_src * 1.0 + 1.0

    # boundary points of the output rectangle
    xs = np.linspace(-(w - 1) / 2, (w - 1) / 2, 2 * w) / d_out
    ys = np.linspace(-(h - 1) / 2, (h - 1) / 2, 2 * h) / d_out
    bx = np.concatenate([xs, xs, np.full_like(ys, xs[0]), np.full_like(ys, xs[-1])])
    by = np.concatenate([np.full_like(xs, ys[0]), np.full_like(xs, ys[-1]), ys, ys])

    def fits(s):
        sx, sy = _src_coords(bx, by, s, tab, d_src, cx, cy)
        # CA moves R/B along the radius: include the largest factor
        f = 1.0 + abs(ca) * (bx**2 + by**2)
        sx2, sy2 = cx + (sx - cx) * f, cy + (sy - cy) * f
        ok = np.isfinite(sx2) & np.isfinite(sy2)
        return bool(ok.all() and sx2.min() >= m and sx2.max() <= wi - 1 - m and sy2.min() >= m and sy2.max() <= hi - 1 - m)

    if fits(1.0):
        s = 1.0
    else:
        lo, hi_s = 0.05, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi_s)
            if fits(mid):
                lo = mid
            else:
                hi_s = mid
        s = lo

    jj = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) / d_out
    ii = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) / d_out
    px, py = np.meshgrid(jj, ii)
    sx, sy = _src_coords(px, py, s, tab, d_src, cx, cy)
    rad = np.hypot(px, py).astype(np.float32)
    ca_maps = None
    if ca != 0.0:
        r2 = (px**2 + py**2)
        fr, fb = 1.0 + ca * r2, 1.0 - ca * r2
        ca_maps = (
            (cx + (sx - cx) * fr).astype(np.float32), (cy + (sy - cy) * fr).astype(np.float32),
            (cx + (sx - cx) * fb).astype(np.float32), (cy + (sy - cy) * fb).astype(np.float32),
        )
    return Geometry(sx.astype(np.float32), sy.astype(np.float32), rad, float(s), scale0, ca_maps)


# ----------------------------------------------------------------------------
# photometry helpers
# ----------------------------------------------------------------------------
def _make_luts():
    x = np.arange(256, dtype=np.float64) / 255.0
    s2l = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)
    n = 8192
    l = np.arange(n, dtype=np.float64) / (n - 1)
    l2s = np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055).astype(np.float32)
    return s2l, l2s


_SRGB2LIN, _LIN2SRGB = _make_luts()


def _lin_to_srgb(lin: np.ndarray) -> np.ndarray:
    idx = np.clip(lin * (len(_LIN2SRGB) - 1) + 0.5, 0, len(_LIN2SRGB) - 1).astype(np.int32)
    return _LIN2SRGB[idx]


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(a, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def to_rgb_u8(img) -> np.ndarray:
    """Accept uint8/uint16/float RGB(A) arrays and return (H, W, 3) uint8 RGB."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = a[..., :3]
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(np.rint(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def load_rgb(path) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
    elif a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGB)
    else:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    return to_rgb_u8(a)


def save_rgb(path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        raise OSError(f"cannot write {path}")


# ----------------------------------------------------------------------------
# effect stages
# ----------------------------------------------------------------------------
def _geometry_stage(img: np.ndarray, p: CctvParams, geo: Geometry) -> np.ndarray:
    dx, dy = p.f_shift
    # shake: output shifted by (dx, dy) px -> source sampled at (x - dx*scale)
    sh = (-dx * geo.scale, -dy * geo.scale)

    def remap(ch, mx, my):
        return cv2.remap(ch, mx + np.float32(sh[0]), my + np.float32(sh[1]), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if geo.ca_maps is None:
        return np.stack([remap(img[..., c], geo.map_x, geo.map_y) for c in range(3)], axis=2)
    rx, ry, bx, by = geo.ca_maps
    return np.stack([remap(img[..., 0], rx, ry), remap(img[..., 1], geo.map_x, geo.map_y), remap(img[..., 2], bx, by)], axis=2)


def effective_map(p: CctvParams, geo: Geometry):
    """The exact map used for the (green channel of the) image, including shake. GT uses this."""
    dx, dy = p.f_shift
    return geo.map_x - np.float32(dx * geo.scale), geo.map_y - np.float32(dy * geo.scale)


def _sensor_stage(rgb_u8: np.ndarray, p: CctvParams, geo: Geometry) -> np.ndarray:
    """Vignette, softness, exposure/WB, noise, tone, chroma noise, sharpening -> float sRGB 0..1."""
    rng = np.random.default_rng([p.f_noise_seed & 0xFFFFFFFF, 0xC7C7])
    h, w = rgb_u8.shape[:2]
    lin = _SRGB2LIN[rgb_u8]  # float32 (H, W, 3)
    lin *= (1.0 - p.vignette * geo.rad**2)[..., None]
    if p.soften_sigma > 0:
        lin = _blur(lin, p.soften_sigma)
    wb = np.array([p.wb[0] * p.f_wb[0], 1.0, p.wb[1] * p.f_wb[1]], dtype=np.float32)
    lin *= (p.exposure * p.f_exposure) * wb
    np.maximum(lin, 0.0, out=lin)

    # Poisson-Gaussian noise in linear light, gain dependent
    a = p.noise_a * p.gain
    shot = rng.poisson(lin.astype(np.float64) / a).astype(np.float32) * np.float32(a) - lin
    read = rng.standard_normal(lin.shape, dtype=np.float32) * np.float32(p.read_noise * p.gain)
    noise = shot + read
    if p.noise_corr > 0:
        comp = 2.0 * math.sqrt(math.pi) * p.noise_corr  # restores the variance lost by the blur
        noise = _blur(noise, p.noise_corr) * np.float32(comp)
    srgb = _lin_to_srgb(lin + noise)

    # colour noise: blotchy chroma noise, stronger in the shadows
    if p.chroma_noise > 0:
        hh, hw = (h + 1) // 2, (w + 1) // 2
        cn = rng.standard_normal((hh, hw, 2), dtype=np.float32)
        cn = _blur(cn, 0.9) * np.float32(2.0 * math.sqrt(math.pi) * 0.9)
        cn = cv2.resize(cn, (w, h), interpolation=cv2.INTER_LINEAR)
        y = _luma(srgb)
        wgt = (0.15 + 0.85 * (1.0 - np.clip(y, 0, 1)) ** 3) * p.chroma_noise * math.sqrt(p.gain / 2.5)
        cb, cr = cn[..., 0] * wgt, cn[..., 1] * wgt
        srgb[..., 0] += 1.402 * cr
        srgb[..., 1] -= 0.344 * cb + 0.714 * cr
        srgb[..., 2] += 1.772 * cb

    # CCTV sharpening (edge enhancement) on luma -> light/dark halos
    if p.sharpen_amount > 0:
        y = _luma(srgb)
        delta = (y - _blur(y, p.sharpen_sigma)) * np.float32(p.sharpen_amount)
        srgb += delta[..., None]
    return np.clip(srgb, 0.0, 1.0)


@functools.lru_cache(maxsize=32)
def _font(px: int) -> ImageFont.ImageFont:
    for cand in (r"C:\Windows\Fonts\consola.ttf", "consola.ttf", "DejaVuSansMono.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSansMono.ttf", "/Library/Fonts/Courier New.ttf"):
        try:
            return ImageFont.truetype(cand, px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=px)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def timestamp_text(p: CctvParams, t: float) -> str:
    t0 = _dt.datetime.strptime(p.t0, "%Y-%m-%d %H:%M:%S")
    return (t0 + _dt.timedelta(seconds=math.floor(t + 1e-9))).strftime("%Y-%m-%d %H:%M:%S")


def draw_overlay(img_u8: np.ndarray, p: CctvParams, t: float) -> np.ndarray:
    """Burned-in DVR text: timestamp top-left, camera bottom-right, SYNTHETIC top-right."""
    h, w = img_u8.shape[:2]
    im = Image.fromarray(img_u8)
    dr = ImageDraw.Draw(im)
    big, small = _font(max(10, round(h / 38))), _font(max(9, round(h / 54)))
    stroke_big, stroke_small = max(1, round(h / 540)), max(1, round(h / 720))
    mx, my = round(w * 0.015), round(h * 0.022)
    kw = dict(fill=(255, 255, 255), stroke_fill=(0, 0, 0))
    dr.text((mx, my), timestamp_text(p, t), font=big, anchor="la", stroke_width=stroke_big, **kw)
    dr.text((w - mx, my), "SYNTHETIC", font=small, anchor="ra", stroke_width=stroke_small, **kw)
    dr.text((w - mx, h - my), f"{display_cam(p.cam_id)}  {p.cam_label}", font=big, anchor="rd", stroke_width=stroke_big, **kw)
    return np.asarray(im)


# ----------------------------------------------------------------------------
# compression (real x264 through ffmpeg)
# ----------------------------------------------------------------------------
_COLOR_ENC = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
_VF_ENC = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
_VF_DEC = "scale=in_color_matrix=bt709:in_range=tv,format=rgb24"


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH (needed for the H.264 step)")
    return exe


def _enc_cmd(w: int, h: int, fps: float, p: CctvParams, out: str, fmt: Optional[str] = None) -> list:
    if w % 2 or h % 2:
        raise ValueError("H.264 yuv420p needs even output sizes")
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-vf", _VF_ENC, "-c:v", "libx264", "-preset", p.preset, "-crf", str(p.crf), "-g", str(p.keyint),
           "-pix_fmt", "yuv420p", *_COLOR_ENC]
    if fmt:
        cmd += ["-f", fmt]
    else:
        cmd += ["-movflags", "+faststart"]
    return cmd + [out]


def compress_still(img: np.ndarray, p: CctvParams) -> np.ndarray:
    """One-frame real x264 encode -> decode round trip (blocking, ringing, 4:2:0 chroma are genuine)."""
    h, w = img.shape[:2]
    enc = subprocess.run(_enc_cmd(w, h, 25, p, "pipe:1", fmt="h264"), input=np.ascontiguousarray(img).tobytes(),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if enc.returncode != 0:
        raise RuntimeError("ffmpeg encode failed: " + enc.stderr.decode(errors="replace"))
    dec = subprocess.run([_ffmpeg(), "-loglevel", "error", "-f", "h264", "-i", "pipe:0", "-vf", _VF_DEC, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "pipe:1"], input=enc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         check=False)
    if dec.returncode != 0 or len(dec.stdout) < w * h * 3:
        raise RuntimeError("ffmpeg decode failed: " + dec.stderr.decode(errors="replace"))
    return np.frombuffer(dec.stdout[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def decode_video(path, w: int, h: int) -> Iterable[np.ndarray]:
    """Yield RGB uint8 frames of a video file decoded with ffmpeg."""
    proc = subprocess.Popen([_ffmpeg(), "-loglevel", "error", "-i", str(path), "-vf", _VF_DEC, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
    finally:
        proc.stdout.close()
        proc.wait()


# ----------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------
def process_frame(img, params: CctvParams, t: float = 0.0, gt: Optional[GtFrame] = None, compress: bool = True):
    """Clean RGB frame (+ optional GT) -> CCTV frame (+ GT warped by the identical map).

    ``params`` should be per-frame (``frame_params(base, idx, seed)``); ``t`` is the
    elapsed time in seconds (drives the timestamp). ``compress=False`` skips the
    x264 step (sequences encode the whole clip instead). Returns ``(uint8 RGB, GtFrame|None)``.
    """
    p = params
    src = to_rgb_u8(img)
    hi, wi = src.shape[:2]
    if gt is not None:
        for arr in (gt.inst, gt.depth):
            if arr is not None and arr.shape[:2] != (hi, wi):
                raise ValueError(f"GT shape {arr.shape} != image shape {src.shape[:2]}")
    out_wh = (int(p.size[0]), int(p.size[1]))
    geo = build_geometry((wi, hi), out_wh, float(p.overscan), float(p.k1), float(p.k2), float(p.ca))
    if geo.scale > 1.6:  # strong downscale: pre-filter to avoid aliasing (same normalised map)
        f = geo.scale / 1.25
        small = cv2.resize(src, (max(2, round(wi / f)), max(2, round(hi / f))), interpolation=cv2.INTER_AREA)
        sgeo = build_geometry((small.shape[1], small.shape[0]), out_wh, float(p.overscan), float(p.k1), float(p.k2), float(p.ca))
        warped = _geometry_stage(small, p, sgeo)
    else:
        warped = _geometry_stage(src, p, geo)
    srgb = _sensor_stage(warped, p, geo)
    out = np.rint(srgb * 255.0).astype(np.uint8)
    if p.overlay:
        out = draw_overlay(out, p, t)
    if compress:
        out = compress_still(out, p)
    out_gt = None
    if gt is not None:
        mx, my = effective_map(p, geo)
        out_gt = warp_gt(gt, mx, my, p.min_box_pixels)
    return out, out_gt


def _list_frames(path) -> list:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.png"))


def process_sequence(frames, out_dir, base: CctvParams, fps: float = 25.0, seed: int = 0, gt_dir=None,
                     video: bool = False, video_name: Optional[str] = None, progress: bool = False) -> dict:
    """Process a frame sequence. ``frames``: dir of PNGs or list of paths.

    Writes ``frames/<stem>.png`` (decoded from the real H.264 stream), ``<cam>.mp4`` if
    ``video``, ``gt/`` (inst, depth, boxes per frame), ``annotations_coco.json`` and
    ``camera_params.json``. Returns a dict with the output paths and the COCO doc.
    """
    paths = _list_frames(frames) if not isinstance(frames, (list, tuple)) else [Path(f) for f in frames]
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {frames}")
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    w, h = int(base.size[0]), int(base.size[1])
    mp4 = out_dir / (video_name or f"{base.cam_id.lower()}.mp4")
    enc = subprocess.Popen(_enc_cmd(w, h, fps, base, str(mp4)), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    coco_frames = []
    stems = []
    try:
        for i, pth in enumerate(paths):
            p = frame_params(base, i, seed)
            gt = load_gt(gt_dir, pth.stem) if gt_dir else None
            img, ogt = process_frame(load_rgb(pth), p, t=i / fps, gt=gt, compress=False)
            enc.stdin.write(np.ascontiguousarray(img).tobytes())
            stems.append(pth.stem)
            if ogt is not None:
                save_gt(ogt, out_dir / "gt", pth.stem, extra={"frame": i, "t": i / fps})
                coco_frames.append({"file_name": f"frames/{pth.stem}.png", "width": w, "height": h, "frame": i, "t": i / fps,
                                    "boxes": ogt.boxes})
            if progress:
                print(f"\rframe {i + 1}/{len(paths)}", end="", file=sys.stderr, flush=True)
        enc.stdin.close()
        err = enc.stderr.read().decode(errors="replace")
        if enc.wait() != 0:
            raise RuntimeError("ffmpeg encode failed: " + err)
    except BaseException:
        enc.kill()
        raise
    n = 0
    for stem, img in zip(stems, decode_video(mp4, w, h)):
        save_rgb(out_dir / "frames" / f"{stem}.png", img)
        n += 1
    if n != len(stems):
        raise RuntimeError(f"decoded {n} frames, expected {len(stems)}")
    if not video:
        mp4.unlink()
    coco = write_coco(out_dir / "annotations_coco.json", coco_frames) if coco_frames else None
    (out_dir / "camera_params.json").write_text(json.dumps(dataclasses.asdict(base), indent=1, default=list), encoding="utf-8")
    if progress:
        print(file=sys.stderr)
    return {"frames_dir": out_dir / "frames", "video": mp4 if video else None, "n": n, "coco": coco}


def process_still(path, out_dir, base: CctvParams, seed: int = 0, frame_idx: int = 0, gt_dir=None, t: float = 0.0) -> dict:
    """One PNG -> ``<out>/<stem>.png`` (real x264 round trip) + GT files if given."""
    path, out_dir = Path(path), Path(out_dir)
    p = frame_params(base, frame_idx, seed)
    gt = load_gt(gt_dir, path.stem) if gt_dir else None
    img, ogt = process_frame(load_rgb(path), p, t=t, gt=gt, compress=True)
    save_rgb(out_dir / f"{path.stem}.png", img)
    if ogt is not None:
        save_gt(ogt, out_dir / "gt", path.stem)
    return {"image": out_dir / f"{path.stem}.png", "gt": ogt}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m render.cctv.cctv_post", description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", required=True, help="frames directory (PNG) or a single PNG")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default="CAM01")
    ap.add_argument("--cam-label", default="BODY SHOP")
    ap.add_argument("--t0", default="2026-10-17 10:00:30")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt", default=None, help="GT directory (see render/cctv/gt.py)")
    ap.add_argument("--video", action="store_true", help="also keep the H.264 mp4")
    ap.add_argument("--size", default="1920x1080", help="WxH, e.g. 1920x1080 or 1280x720")
    ap.add_argument("--overscan", type=float, default=0.0, help="input overscan fraction per side (see CctvParams)")
    ap.add_argument("--crf", type=int, default=None)
    ap.add_argument("--keyint", type=int, default=None)
    ap.add_argument("--min-pixels", type=int, default=None, help="drop boxes under this visible pixel count")
    ap.add_argument("--no-overlay", action="store_true")
    a = ap.parse_args(argv)

    m = re.fullmatch(r"(\d+)[xX](\d+)", a.size)
    if not m:
        ap.error("--size must look like 1920x1080")
    ov = {}
    if a.crf is not None:
        ov["crf"] = a.crf
    if a.keyint is not None:
        ov["keyint"] = a.keyint
    if a.min_pixels is not None:
        ov["min_box_pixels"] = a.min_pixels
    if a.no_overlay:
        ov["overlay"] = False
    base = camera_params(a.cam, a.seed, cam_label=a.cam_label, t0=a.t0, size=(int(m.group(1)), int(m.group(2))),
                         overscan=a.overscan, **ov)
    inp = Path(a.inp)
    if inp.is_file():
        r = process_still(inp, a.out, base, seed=a.seed, gt_dir=a.gt)
        print(f"wrote {r['image']}")
    else:
        r = process_sequence(inp, a.out, base, fps=a.fps, seed=a.seed, gt_dir=a.gt, video=a.video, progress=True)
        print(f"wrote {r['n']} frames to {r['frames_dir']}" + (f" and {r['video']}" if r["video"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
