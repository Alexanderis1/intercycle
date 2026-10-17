"""First-contact probe for the AMD Token Factory endpoint: run this the moment the key arrives.

    set INTERCYCLE_LLM_BASE_URL=<base URL from the Token Factory console>
    set INTERCYCLE_LLM_API_KEY=<key>
    python scripts/gemma_probe.py [--model <id>]

It (1) lists GET /v1/models and picks the Gemma 4 id, (2) checks plain text, image input (a synthetic
sketch), response_format=json_schema and native tool calling, and (3) prints the exact env lines to use.
Calls are real live calls and are logged to out/llm_calls.jsonl like any other.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.llm import GemmaClient, LLMConfig, LLMError, verify_state  # noqa: E402
from intercycle.llm.client import OpenAITransport  # noqa: E402
from intercycle.llm.fixtures import render  # noqa: E402

PREFER = ("31b", "26b", "12b", "e4b", "e2b")   # rough order: biggest first; change with --model


def pick_model(ids: list[str], wanted: str | None) -> str | None:
    """The requested id if the server lists it, else the best instruction-tuned Gemma 4 id."""
    if wanted and wanted in ids:
        return wanted
    gem = [i for i in ids if any(t in i.lower() for t in ("gemma-4", "gemma4", "gemma_4"))]
    for tag in PREFER:
        for i in gem:
            low = i.lower()
            if tag in low and ("-it" in low or "instruct" in low):
                return i
    return gem[0] if gem else None


def step(name: str, fn) -> tuple[bool, str]:
    try:
        ok, detail = fn()
    except LLMError as e:
        ok, detail = False, f"{type(e).__name__}: {str(e)[:300]}"
    print(f"[{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    return ok, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    a = ap.parse_args()
    cfg = LLMConfig.from_env(mode="live", model=a.model)
    try:
        t = OpenAITransport(cfg)
        ids = t.list_models()
    except LLMError as e:
        print(f"cannot list models: {e}")
        return 2
    print(f"{len(ids)} models at {cfg.base_url}:")
    for i in ids:
        print("   ", i)
    model = pick_model(ids, a.model or (cfg.model if cfg.model in ids else None))
    if not model:
        print("no Gemma 4 id found in /v1/models; set INTERCYCLE_LLM_MODEL by hand")
        return 3
    print(f"using model: {model}")
    cfg = LLMConfig.from_env(mode="live", model=model)
    c = GemmaClient(cfg, transport=t)
    env = {"INTERCYCLE_LLM_MODEL": model}

    step("text", lambda: (lambda r: ("ready" in r.content.lower(), f"{r.content.strip()[:60]!r} in {r.latency_s:.2f}s"))(
        c.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=16, task="probe_text")))

    def vision():
        r = verify_state([render(True, (118, 148), "R1")], "R1", {"moving": True}, client=c, raise_on_error=True)
        v = r.verdict
        return (v.moving and v.person_in_zone), f"moving={v.moving} person_in_zone={v.person_in_zone} conf={v.confidence:.2f} in {r.latency_s:.2f}s (expected True/True)"
    ok_v, _ = step("vision + structured verify_state", vision)

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    def rf():
        r = c.chat([{"role": "user", "content": 'Reply with {"ok": true}'}], max_tokens=32, task="probe_rf",
                   response_format={"type": "json_schema", "json_schema": {"name": "probe", "schema": schema}})
        return json.loads(r.content).get("ok") is True, f"reply {r.content.strip()[:60]!r}"
    try:
        ok_rf, _ = step("response_format json_schema", rf)
    except (ValueError, KeyError) as e:
        ok_rf = False
        print(f"[FAIL] response_format json_schema: reply was not JSON ({e})")
    if not ok_rf:
        env["INTERCYCLE_LLM_JSON_MODE"] = "prompt"

    tool = [{"type": "function", "function": {"name": "get_time", "description": "Current plant time",
                                              "parameters": {"type": "object", "properties": {}}}}]
    def tools():
        r = c.chat([{"role": "user", "content": "What time is it? Use the tool."}], tools=tool, tool_choice="auto",
                   max_tokens=64, task="probe_tools")
        names = [t_["function"]["name"] for t_ in r.tool_calls]
        return names == ["get_time"], f"tool_calls={names}"
    ok_t, _ = step("native tool calling", tools)
    if not ok_t:
        env["INTERCYCLE_LLM_TOOLS"] = "react"

    print("\nUse these environment variables (plus the base URL and key you already set):")
    for k, v in env.items():
        print(f"  set {k}={v}")
    print("  set INTERCYCLE_LLM_MODE=record      (live + save to out/llm_cache; later replay offline)")
    if not ok_v:
        print("WARNING: vision check failed; do not rely on verify_state/commission with this model id")
    return 0


if __name__ == "__main__":
    sys.exit(main())
