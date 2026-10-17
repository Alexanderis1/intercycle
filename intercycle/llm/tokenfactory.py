"""Token Factory account helpers: `/v1/models` and `/v1/usage` (numeric fields only), `.env` loading.

The key is read from the environment / `.env` and is never printed, logged or written anywhere. `usage()` keeps only
numeric fields (and the timestamp-like string fields listed in `_KEEP_STR`), so a usage record can be logged safely.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .client import REPO_ROOT, LLMConfig

USAGE_LOG = REPO_ROOT / "out" / "gemma" / "usage_log.jsonl"
_KEEP_STR = ("reset", "date", "period")


def load_env(path: Path = REPO_ROOT / ".env") -> None:
    """Read KEY=VALUE lines into os.environ (existing variables win). Values are never printed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _get(cfg: LLMConfig, path: str, timeout: float = 20.0) -> Any:
    url = cfg.base_url.rstrip("/") + path
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {cfg.api_key}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:      # noqa: S310 - fixed https base URL from .env
        return json.loads(r.read().decode("utf-8"))


def _numeric(o: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_numeric(v, f"{prefix}{k}."))
    elif isinstance(o, list):
        for i, v in enumerate(o[:20]):
            out.update(_numeric(v, f"{prefix}{i}."))
    elif isinstance(o, bool):
        out[prefix[:-1]] = o
    elif isinstance(o, (int, float)):
        out[prefix[:-1]] = o
    elif isinstance(o, str) and any(s in prefix.lower() for s in _KEEP_STR) and len(o) < 40:
        out[prefix[:-1]] = o
    return out


def models(cfg: LLMConfig) -> list[str]:
    d = _get(cfg, "/models")
    return sorted(m.get("id", "") for m in (d.get("data") or []) if isinstance(m, dict))


def usage(cfg: LLMConfig, tag: str = "", log: bool = True) -> dict[str, Any]:
    """GET /v1/usage, numeric fields only; appended to out/gemma/usage_log.jsonl with `tag`."""
    try:
        rec = _numeric(_get(cfg, "/usage"))
        ok = True
    except (urllib.error.URLError, OSError, ValueError) as e:
        rec, ok = {"error": f"{type(e).__name__}: {str(e)[:120]}"}, False
    rec = {"ts": round(time.time(), 1), "tag": tag, "ok": ok, **rec}
    if log:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return rec
