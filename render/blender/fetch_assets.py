"""Download NVIDIA Isaac/SimReady asset folders from the public bucket, with their licence files.

    python fetch_assets.py Robots/Fanuc/r2000ic_210f/ SimReady/Industrial/Warehouse/Racks/ [--root 6.0] [--dest D:\\icdata\\assets\\nvidia]
    python fetch_assets.py --list SimReady/Industrial/Warehouse/        # print keys + sizes, download nothing

Stdlib only (runs in Store Python or Blender's Python). The bucket is anonymously listable
(S3 ListObjectsV2). Files land under <dest>/<root>/<prefix...>, keeping the bucket layout so relative
USD references keep resolving. Thumbnails (.thumbs/) and Omniverse sidecars (.wrapp) are skipped.
Each call appends what it saw to <dest>/manifest.json: per asset folder, the licence file found
(LICENSE / LICENSE.txt / LICENSE.md, any case) and its first line, so render/credits.json can be built from it.
Assets are NEVER committed to git.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ROOTS = {"6.0": "Assets/Isaac/6.0/Isaac/", "5.1": "Assets/Isaac/5.1/Isaac/", "dt": "Assets/DigitalTwin/Assets/"}
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
SKIP = (".thumbs/", ".wrapp", ".DS_Store")
LICENCE_NAMES = ("license", "license.txt", "license.md")
DEFAULT_DEST = Path(r"D:\icdata\assets\nvidia")


def list_keys(prefix: str) -> list[tuple[str, int]]:
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall("s3:Contents", NS):
            out.append((c.findtext("s3:Key", "", NS), int(c.findtext("s3:Size", "0", NS))))
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return out
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)


def download(key: str, size: int, dest_root: Path, strip: str) -> Path:
    dst = dest_root / key[len(strip):]
    if dst.exists() and dst.stat().st_size == size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    if tmp.stat().st_size != size:
        raise IOError(f"size mismatch for {key}: {tmp.stat().st_size} != {size}")
    tmp.replace(dst)
    return dst


def fetch(prefix: str, root: str = "6.0", dest: Path = DEFAULT_DEST, max_mb: float = 2000.0) -> dict:
    """Download every file under ROOTS[root] + prefix. Returns {folder: licence_info} for the manifest."""
    base = ROOTS[root]
    keys = [(k, s) for k, s in list_keys(base + prefix) if not any(x in k for x in SKIP)]
    total = sum(s for _, s in keys) / 1e6
    if total > max_mb:
        raise SystemExit(f"{prefix}: {total:.0f} MB > --max-mb {max_mb:.0f}; narrow the prefix")
    dest_root = dest / root
    print(f"{prefix}: {len(keys)} files, {total:.1f} MB -> {dest_root}", flush=True)
    seen: dict[str, dict] = {}
    for k, s in keys:
        p = download(k, s, dest_root, base)
        if p.name.lower() in LICENCE_NAMES:
            first = next((ln.strip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()), "")
            seen[str(p.parent.relative_to(dest_root)).replace("\\", "/")] = {"licence_file": p.name, "first_line": first[:200]}
    return seen


def update_manifest(dest: Path, root: str, prefix: str, licences: dict) -> None:
    mf = dest / "manifest.json"
    data = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {"source": BUCKET, "fetched": {}}
    data["fetched"][f"{root}:{prefix}"] = {"root": ROOTS[root], "licences": licences}
    mf.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download NVIDIA Isaac/SimReady asset folders with their licence files.")
    ap.add_argument("prefixes", nargs="+", help="path under the root, e.g. Robots/Fanuc/r2000ic_210f/")
    ap.add_argument("--root", default="6.0", choices=sorted(ROOTS))
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--max-mb", type=float, default=2000.0, help="refuse a prefix larger than this")
    ap.add_argument("--list", action="store_true", help="list keys and sizes only")
    a = ap.parse_args(argv)
    for pre in a.prefixes:
        if a.list:
            for k, s in list_keys(ROOTS[a.root] + pre):
                if not any(x in k for x in SKIP):
                    print(f"{s / 1e6:9.2f} MB  {k[len(ROOTS[a.root]):]}")
            continue
        lic = fetch(pre, a.root, a.dest, a.max_mb)
        update_manifest(a.dest, a.root, pre, lic)
        for folder, info in lic.items():
            print(f"  licence {folder}: {info['first_line']}")
        if not lic:
            print(f"  WARNING no licence file under {pre}: do not use until checked", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
