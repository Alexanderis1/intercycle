"""OpenAI-compatible chat client for Gemma 4 on AMD Token Factory.

One small client, three modes, and a per-call usage log:

* ``live``    call the endpoint, touch no cache
* ``record``  call the endpoint and save the response in ``out/llm_cache/``
* ``replay``  cache only; a missing entry is a clear :class:`CacheMiss` (works offline)

Every call (including cache hits, flagged ``live: false``) appends one JSON line to
``out/llm_calls.jsonl`` with model, latency, tokens and cost-free status. The dashboard reads that
file as evidence of real AMD-hosted inference, so only ``live: true`` lines count as such.

Configuration comes from environment variables (see :meth:`LLMConfig.from_env`):

    INTERCYCLE_LLM_BASE_URL   OpenAI-compatible base URL (the one the Token Factory console shows)
    INTERCYCLE_LLM_API_KEY    bearer key
    INTERCYCLE_LLM_MODEL      model id (default ``google/gemma-4-31B-it``; confirm with scripts/gemma_probe.py)
    INTERCYCLE_LLM_MODE       live | record | replay   (default: record if a key is set, else replay)
    INTERCYCLE_LLM_JSON_MODE  auto | schema | prompt   (auto: try response_format, fall back to prompt-only JSON)
    INTERCYCLE_LLM_TOOLS      auto | native | react    (auto: native tool calls, fall back to a ReAct JSON loop)
    INTERCYCLE_LLM_CACHE_DIR / INTERCYCLE_LLM_LOG / INTERCYCLE_LLM_COST_FREE / INTERCYCLE_LLM_TIMEOUT_S
    INTERCYCLE_LLM_PROVIDER   label for the log (default derived from the base-URL host)

Safety boundary: nothing in this package can authorise a robot to enter a zone. Gemma output is only
ever used to lower confidence or to add warnings (see ``tasks.py``).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "google/gemma-4-31B-it"
MODES = ("live", "record", "replay")

T = TypeVar("T", bound=BaseModel)
Transport = Callable[[dict], dict]   # full request body -> OpenAI chat.completion dict

_LOG_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------- errors
class LLMError(RuntimeError):
    """Base class for everything this package raises."""


class ConfigError(LLMError):
    pass


class CacheMiss(LLMError):
    pass


class TransportError(LLMError):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class ToolsUnsupported(LLMError):
    """The endpoint/model rejected the ``tools`` parameter."""


class StructuredOutputError(LLMError):
    def __init__(self, message: str, raw: str = "", attempts: int = 0):
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


# -------------------------------------------------------------------------------------- cache key
def _norm(o: Any) -> Any:
    if isinstance(o, BaseModel):
        o = o.model_dump(mode="json")
    if isinstance(o, Mapping):
        return {str(k): _norm(v) for k, v in o.items() if v is not None}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, float) and o.is_integer() and abs(o) < 1e15:
        return int(o)          # 0.0 and 0 must hash alike
    return o


def canonical_json(obj: Any) -> str:
    return json.dumps(_norm(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_key(model: str, messages: Sequence[Mapping], params: Mapping | None = None) -> str:
    """Content address of one request: sha256 over (model, messages, params), key-order independent."""
    blob = canonical_json({"model": model, "messages": list(messages), "params": dict(params or {})})
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


def _redact(o: Any) -> Any:
    """Replace base64 image payloads by a digest so cache files stay small and readable."""
    if isinstance(o, str) and o.startswith("data:") and ";base64," in o:
        head, b64 = o.split(";base64,", 1)
        return f"{head};base64,<sha256:{hashlib.sha256(b64.encode()).hexdigest()[:16]} chars:{len(b64)}>"
    if isinstance(o, Mapping):
        return {k: _redact(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_redact(v) for v in o]
    return o


# ----------------------------------------------------------------------------------------- images
def _sniff_mime(b: bytes) -> str:
    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if b.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unsupported image bytes (expected PNG, JPEG or WEBP)")


def image_bytes(img: Any) -> tuple[bytes, str]:
    """Accept a path, raw bytes, a PIL image or an HxWxC uint8 array; return (bytes, mime)."""
    if isinstance(img, (str, Path)):
        b = Path(img).read_bytes()
        return b, _sniff_mime(b)
    if isinstance(img, (bytes, bytearray)):
        return bytes(img), _sniff_mime(bytes(img))
    import io
    from PIL import Image  # lazy: only needed for in-memory images
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def image_size(b: bytes) -> tuple[int, int]:
    """(width, height) in pixels. PNG is parsed from the header; other formats need PIL."""
    if b.startswith(b"\x89PNG\r\n\x1a\n") and len(b) >= 24:
        return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")
    import io
    from PIL import Image
    with Image.open(io.BytesIO(b)) as im:
        return im.size


def image_part(img: Any) -> dict:
    """An OpenAI ``image_url`` content part holding a base64 data URL."""
    b, mime = image_bytes(img)
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"}}


def user_message(text: str, images: Sequence[Any] = ()) -> dict:
    """User turn with optional images; images come first, then the text."""
    if not images:
        return {"role": "user", "content": text}
    return {"role": "user", "content": [image_part(i) for i in images] + [{"type": "text", "text": text}]}


def _count_images(messages: Sequence[Mapping]) -> int:
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, Mapping) and p.get("type") == "image_url")
    return n


# ------------------------------------------------------------------------------------------ JSON
def extract_json(text: str | None) -> Any:
    """Parse the first JSON value in a model reply (tolerates prose and ```json fences)."""
    if not text or not text.strip():
        raise ValueError("empty reply")
    s = text.strip()
    cands = [s]
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S | re.I)
    if m:
        cands.append(m.group(1).strip())
    for c in cands:
        try:
            return json.loads(c)
        except ValueError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                return dec.raw_decode(s[i:])[0]
            except ValueError:
                continue
    raise ValueError("no JSON value found in reply")


def inline_refs(schema: dict) -> dict:
    """Pydantic JSON schema with $defs/$ref inlined and titles dropped (friendlier to guided decoding)."""
    defs = schema.get("$defs", {})

    def walk(n: Any, seen: tuple = ()) -> Any:
        if isinstance(n, dict):
            if "$ref" in n:
                name = n["$ref"].split("/")[-1]
                if name in seen or name not in defs:
                    return {}
                base = walk(defs[name], seen + (name,))
                extra = {k: walk(v, seen) for k, v in n.items() if k != "$ref"}
                return {**base, **extra}
            return {k: walk(v, seen) for k, v in n.items()
                    if k != "$defs" and not (k == "title" and isinstance(v, str))}
        if isinstance(n, list):
            return [walk(v, seen) for v in n]
        return n

    return walk(schema)


def schema_json(model: type[BaseModel]) -> dict:
    return inline_refs(model.model_json_schema())


# ----------------------------------------------------------------------------------------- config
def _truthy(v: str | None, default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def provider_label(base_url: str) -> str:
    host = urlparse(base_url).hostname or ""
    if host.endswith(("amd.com", "amd.com.cn", "anruicloud.com")):
        return "amd-token-factory"
    return f"other:{host}" if host else "none"


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = DEFAULT_MODEL
    mode: str = "replay"
    cache_dir: Path = REPO_ROOT / "out" / "llm_cache"
    log_path: Path = REPO_ROOT / "out" / "llm_calls.jsonl"
    json_mode: str = "auto"
    tools_mode: str = "auto"
    cost_free: bool = True
    timeout_s: float = 60.0
    max_retries: int = 2
    provider: str = ""
    vlm_mode: str = "cloud"          # INTERCYCLE_VLM_MODE: off | local (on-prem OpenAI-compatible server) | cloud

    def __post_init__(self) -> None:
        if self.vlm_mode not in ("off", "local", "cloud"):
            raise ConfigError("INTERCYCLE_VLM_MODE must be off|local|cloud")
        self.cache_dir, self.log_path = Path(self.cache_dir), Path(self.log_path)
        if self.mode not in MODES:
            raise ConfigError(f"INTERCYCLE_LLM_MODE must be one of {MODES}, got {self.mode!r}")
        if self.json_mode not in ("auto", "schema", "object", "prompt"):
            raise ConfigError("INTERCYCLE_LLM_JSON_MODE must be auto|schema|object|prompt")
        if self.tools_mode not in ("auto", "native", "react"):
            raise ConfigError("INTERCYCLE_LLM_TOOLS must be auto|native|react")
        if not self.provider:
            self.provider = provider_label(self.base_url)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: Any) -> "LLMConfig":
        e = os.environ if env is None else env

        def g(k: str, d: str = "") -> str:
            return e.get(f"INTERCYCLE_LLM_{k}", d)

        base, key = g("BASE_URL"), g("API_KEY")
        kw: dict[str, Any] = dict(
            base_url=base, api_key=key, model=g("MODEL") or DEFAULT_MODEL,
            mode=g("MODE") or ("record" if (base and key) else "replay"),
            json_mode=g("JSON_MODE") or "auto", tools_mode=g("TOOLS") or "auto",
            cost_free=_truthy(g("COST_FREE"), True), timeout_s=float(g("TIMEOUT_S") or 60),
            provider=g("PROVIDER"), vlm_mode=e.get("INTERCYCLE_VLM_MODE", "cloud") or "cloud",
        )
        if g("CACHE_DIR"):
            kw["cache_dir"] = Path(g("CACHE_DIR"))
        if g("LOG"):
            kw["log_path"] = Path(g("LOG"))
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)


# ------------------------------------------------------------------------------------- transport
class OpenAITransport:
    """Thin wrapper over the ``openai`` SDK; converts SDK exceptions into :class:`TransportError`."""

    def __init__(self, cfg: LLMConfig):
        if not cfg.base_url:
            raise ConfigError("INTERCYCLE_LLM_BASE_URL is not set (use the base URL shown in the Token Factory console)")
        if not cfg.api_key:
            raise ConfigError("INTERCYCLE_LLM_API_KEY is not set")
        import openai
        self._openai = openai
        self._client = openai.OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                                     timeout=cfg.timeout_s, max_retries=cfg.max_retries)

    def list_models(self) -> list[str]:
        try:
            return [m.id for m in self._client.models.list().data]
        except self._openai.APIStatusError as e:
            raise TransportError(f"HTTP {e.status_code}: {e.message}", status=e.status_code) from e
        except self._openai.APIConnectionError as e:
            raise TransportError(f"connection error: {e}") from e

    def __call__(self, payload: dict) -> dict:
        kw = dict(payload)
        extra = kw.pop("extra_body", None)
        if extra:
            kw["extra_body"] = extra
        try:
            return self._client.chat.completions.create(**kw).model_dump()
        except self._openai.APIStatusError as e:
            raise TransportError(f"HTTP {e.status_code}: {e.message}", status=e.status_code,
                                 body=getattr(e, "body", None)) from e
        except self._openai.APIConnectionError as e:   # includes timeouts
            raise TransportError(f"connection error: {e}") from e


_RF_WORDS = ("response_format", "json_schema", "json_object", "guided", "structured output", "grammar")
_TOOL_WORDS = ("tool", "function call", "function_call")


def _unsupported(e: TransportError, words: tuple[str, ...]) -> bool:
    if e.status not in (400, 404, 405, 415, 422, 501):
        return False
    msg = str(e) + (" " + json.dumps(e.body, default=str) if e.body else "")
    return any(w in msg.lower() for w in words)


# ---------------------------------------------------------------------------------------- results
@dataclass
class ChatResult:
    content: str
    tool_calls: list[dict]
    message: dict              # assistant message, ready to append to the history
    finish_reason: str | None
    usage: dict                # prompt_tokens / completion_tokens / total_tokens (None when unreported)
    latency_s: float           # latency of the original live call (recorded value when cached)
    wall_s: float              # time this call took here (about 0 for a cache hit)
    cached: bool
    key: str
    model: str
    raw: dict = field(repr=False, default_factory=dict)


@dataclass
class StructuredResult(Generic[T]):
    value: T
    attempts: int
    latency_s: float
    wall_s: float
    cached: bool
    prompt_tokens: int | None
    completion_tokens: int | None
    used_response_format: bool
    raw: str


def _sum_opt(xs: list[int | None]) -> int | None:
    ys = [x for x in xs if x is not None]
    return sum(ys) if ys else None


# ----------------------------------------------------------------------------------------- client
class GemmaClient:
    def __init__(self, config: LLMConfig | None = None, transport: Transport | None = None):
        self.cfg = config or LLMConfig.from_env()
        self._transport = transport
        self._rf_ok: bool | None = (None if self.cfg.json_mode == "auto" else
                                    (self.cfg.json_mode in ("schema", "object")))

    @classmethod
    def from_env(cls, **overrides: Any) -> "GemmaClient":
        return cls(LLMConfig.from_env(**overrides))

    # -- cache -------------------------------------------------------------------------------
    def _cache_path(self, key: str) -> Path:
        return self.cfg.cache_dir / key[:2] / f"{key}.json"

    def _cache_read(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def _cache_write(self, key: str, entry: dict) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(entry, indent=1, ensure_ascii=True), encoding="utf-8")
        os.replace(tmp, p)

    # -- log ---------------------------------------------------------------------------------
    def _log(self, rec: dict) -> None:
        rec = {"ts": round(time.time(), 3), "model": self.cfg.model, "provider": self.cfg.provider,
               "endpoint_host": urlparse(self.cfg.base_url).hostname or None,
               "mode": self.cfg.mode, "cost_free": self.cfg.cost_free,
               "cost_usd": 0.0 if self.cfg.cost_free else None, **rec}
        try:
            self.cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_LOCK, self.cfg.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        except OSError:
            pass   # logging must never break a call

    @property
    def transport(self) -> Transport:
        if self._transport is None:
            self._transport = OpenAITransport(self.cfg)
        return self._transport

    # -- one chat call -----------------------------------------------------------------------
    def chat(self, messages: Sequence[Mapping], *, tools: Sequence[Mapping] | None = None,
             tool_choice: str | Mapping | None = None, response_format: Mapping | None = None,
             temperature: float = 0.0, max_tokens: int = 512, extra_body: Mapping | None = None,
             task: str = "chat") -> ChatResult:
        if self.cfg.vlm_mode == "off":
            raise ConfigError("INTERCYCLE_VLM_MODE=off: no model calls (the observer runs without Gemma)")
        messages = list(messages)
        params = {k: v for k, v in dict(
            temperature=temperature, max_tokens=max_tokens, tools=list(tools) if tools else None,
            tool_choice=tool_choice if tools else None, response_format=response_format,
            extra_body=dict(extra_body) if extra_body else None).items() if v is not None}
        key = cache_key(self.cfg.model, messages, params)
        t_wall = time.perf_counter()
        n_img = _count_images(messages)

        if self.cfg.mode == "replay":
            entry = self._cache_read(key)
            if entry is None:
                raise CacheMiss(
                    f"replay mode: no cached response for key {key[:16]} (task={task}, model={self.cfg.model}, "
                    f"{len(messages)} messages, {n_img} images) in {self.cfg.cache_dir}. Record it first with "
                    f"INTERCYCLE_LLM_MODE=record plus INTERCYCLE_LLM_BASE_URL/INTERCYCLE_LLM_API_KEY, or point "
                    f"INTERCYCLE_LLM_CACHE_DIR at a recorded cache (the model id must match too).")
            meta = entry.get("meta", {})
            res = self._to_result(entry["response"], key, latency_s=float(meta.get("latency_s", 0.0)),
                                  wall_s=time.perf_counter() - t_wall, cached=True)
            self._log(dict(task=task, live=False, cached=True, ok=True, latency_s=res.latency_s,
                           wall_s=round(res.wall_s, 6), n_images=n_img, finish_reason=res.finish_reason,
                           key=key[:16], **res.usage))
            return res

        payload = {"model": self.cfg.model, "messages": messages, **params}
        t0 = time.perf_counter()
        try:
            resp = self.transport(payload)
        except TransportError as e:
            self._log(dict(task=task, live=True, cached=False, ok=False, error=str(e)[:300],
                           latency_s=round(time.perf_counter() - t0, 4), n_images=n_img, key=key[:16]))
            if tools and _unsupported(e, _TOOL_WORDS):
                raise ToolsUnsupported(str(e)) from e
            raise
        latency = time.perf_counter() - t0
        res = self._to_result(resp, key, latency_s=latency, wall_s=time.perf_counter() - t_wall, cached=False)
        if self.cfg.mode == "record":
            self._cache_write(key, {
                "key": key, "request": _redact({"model": self.cfg.model, "messages": messages, "params": params}),
                "response": resp,
                "meta": {"latency_s": round(latency, 4), "usage": res.usage, "recorded_at": time.time(),
                         "model": self.cfg.model, "provider": self.cfg.provider, "task": task}})
        self._log(dict(task=task, live=True, cached=False, ok=True, latency_s=round(latency, 4),
                       wall_s=round(res.wall_s, 4), n_images=n_img, finish_reason=res.finish_reason,
                       key=key[:16], **res.usage))
        return res

    def _to_result(self, resp: dict, key: str, *, latency_s: float, wall_s: float, cached: bool) -> ChatResult:
        try:
            choice = resp["choices"][0]
            msg = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"malformed chat completion (no choices): {str(resp)[:200]}") from e
        content = msg.get("content") or ""
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments", "{}")
            calls.append({"id": tc.get("id") or f"call_{i}", "type": "function",
                          "function": {"name": fn.get("name", ""),
                                       "arguments": args if isinstance(args, str) else json.dumps(args)}})
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if calls:
            message["tool_calls"] = calls
        else:
            message["content"] = content
        u = resp.get("usage") or {}
        usage = {k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return ChatResult(content=content, tool_calls=calls, message=message,
                          finish_reason=choice.get("finish_reason"), usage=usage, latency_s=latency_s,
                          wall_s=wall_s, cached=cached, key=key, model=self.cfg.model, raw=resp)

    # -- structured output ---------------------------------------------------------------------
    def structured(self, messages: Sequence[Mapping], schema: type[T], *, name: str | None = None,
                   task: str = "structured", temperature: float = 0.0, max_tokens: int = 512,
                   retries: int = 2, extra_body: Mapping | None = None) -> StructuredResult[T]:
        """Chat call whose reply must validate against a pydantic model.

        Sends ``response_format=json_schema`` when the server supports it, otherwise (or when
        ``INTERCYCLE_LLM_JSON_MODE=prompt``) relies on the "reply only with JSON" instruction that is
        always added to the system message. Invalid replies are fed back to the model for up to
        ``retries`` further attempts.
        """
        js = schema_json(schema)
        hint = ("Reply with ONLY one JSON object (no prose, no markdown fences) that validates against this "
                "JSON Schema:\n" + json.dumps(js, separators=(",", ":")))
        base = _with_system_hint(messages, hint)
        rf = ({"type": "json_object"} if self.cfg.json_mode == "object" else
              {"type": "json_schema", "json_schema": {"name": name or schema.__name__, "schema": js}})
        kw = dict(task=task, temperature=temperature, max_tokens=max_tokens, extra_body=extra_body)
        raw, err = "", ""
        results: list[ChatResult] = []
        used_rf = False
        for attempt in range(retries + 1):
            hist = list(base)
            if attempt:
                hist += [{"role": "assistant", "content": raw or "(empty reply)"},
                         {"role": "user", "content": f"That reply was rejected: {err}. "
                                                     "Reply again with ONLY the corrected JSON object."}]
            res, used_rf = self._structured_call(hist, rf, kw)
            results.append(res)
            raw = res.content
            try:
                value = schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                err = _short_err(e)
                continue
            return StructuredResult(
                value=value, attempts=attempt + 1, latency_s=sum(r.latency_s for r in results),
                wall_s=sum(r.wall_s for r in results), cached=all(r.cached for r in results),
                prompt_tokens=_sum_opt([r.usage.get("prompt_tokens") for r in results]),
                completion_tokens=_sum_opt([r.usage.get("completion_tokens") for r in results]),
                used_response_format=used_rf, raw=raw)
        raise StructuredOutputError(f"no valid {schema.__name__} after {retries + 1} attempts: {err}",
                                    raw=raw, attempts=retries + 1)

    def _structured_call(self, hist: list[dict], rf: dict, kw: dict) -> tuple[ChatResult, bool]:
        if self._rf_ok is False:
            return self.chat(hist, **kw), False
        if self.cfg.mode == "replay" and self._rf_ok is None:
            # we do not know what the recording server supported: try both wire variants
            last: CacheMiss | None = None
            for variant in (True, False):
                try:
                    res = self.chat(hist, response_format=rf if variant else None, **kw)
                except CacheMiss as e:
                    last = e
                    continue
                self._rf_ok = variant
                return res, variant
            assert last is not None
            raise last
        try:
            return self.chat(hist, response_format=rf, **kw), True
        except TransportError as e:
            if self.cfg.json_mode == "auto" and _unsupported(e, _RF_WORDS):
                self._rf_ok = False     # remembered: later calls go straight to prompt-only JSON
                self._log(dict(task=kw["task"], event="response_format_unsupported", error=str(e)[:200]))
                return self.chat(hist, **kw), False
            raise


def _with_system_hint(messages: Sequence[Mapping], hint: str) -> list[dict]:
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system" and isinstance(msgs[0].get("content"), str):
        msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n" + hint
    else:
        msgs.insert(0, {"role": "system", "content": hint})
    return msgs


def _short_err(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in x['loc'])}: {x['msg']}" for x in e.errors()[:4])
    return str(e)[:200]


_default: GemmaClient | None = None


def default_client() -> GemmaClient:
    """Process-wide client configured from the environment."""
    global _default
    if _default is None:
        _default = GemmaClient.from_env()
    return _default


def set_default_client(client: GemmaClient | None) -> None:
    global _default
    _default = client
