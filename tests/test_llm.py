"""Gemma integration tests. Network-free: fake transports and a localhost HTTP server only."""
from __future__ import annotations

import copy
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from intercycle.llm import (  # noqa: E402
    CacheMiss, GemmaClient, LLMConfig, StructuredOutputError, TransportError, cache_key, commission,
    extract_json, image_part, ops_agent, verify_state,
)
from intercycle.llm.client import OpenAITransport, ToolsUnsupported, canonical_json  # noqa: E402
from intercycle.llm.fixtures import MOCK_MODEL, FixtureHeuristicTransport, make_fixture, render  # noqa: E402
from intercycle.llm.tasks import StateVerdict, parse_time  # noqa: E402

VERDICT = {"moving": False, "person_in_zone": False, "confidence": 0.8, "notes": "arm at home"}


def completion(content=None, tool_calls=None, usage=(11, 7)):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}


def tool_call(name, args, id="call_1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Scripted:
    """Fake transport: returns/raises scripted items in order, records every payload."""

    def __init__(self, *items):
        self.items = list(items)
        self.payloads: list[dict] = []

    def __call__(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        it = self.items.pop(0)
        if isinstance(it, Exception):
            raise it
        return it(payload) if callable(it) else it


def make_client(tmp_path, transport, mode="record", **kw):
    cfg = LLMConfig(base_url="https://api.tokenfactory.amd.com/v1", api_key="k", model="gemma-test", mode=mode,
                    cache_dir=tmp_path / "cache", log_path=tmp_path / "calls.jsonl", **kw)
    return GemmaClient(cfg, transport=transport)


def log_lines(tmp_path):
    p = tmp_path / "calls.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ------------------------------------------------------------------------------------ cache key
def test_cache_key_stable_and_sensitive():
    msgs = [{"role": "user", "content": "hi"}]
    k = cache_key("m", msgs, {"temperature": 0.0, "max_tokens": 64})
    assert len(k) == 64 and int(k, 16) >= 0
    # key order, 0 vs 0.0, None-valued params do not matter
    assert k == cache_key("m", [{"content": "hi", "role": "user"}], {"max_tokens": 64, "temperature": 0, "tools": None})
    assert canonical_json({"a": 1.0}) == canonical_json({"a": 1})
    # anything real does
    assert k != cache_key("m2", msgs, {"temperature": 0.0, "max_tokens": 64})
    assert k != cache_key("m", [{"role": "user", "content": "hi!"}], {"temperature": 0.0, "max_tokens": 64})
    assert k != cache_key("m", msgs, {"temperature": 0.7, "max_tokens": 64})
    assert k != cache_key("m", msgs, {"temperature": 0.0, "max_tokens": 64, "response_format": {"type": "json_object"}})


def test_images_in_key_and_redacted_in_cache(tmp_path):
    img = render(False, None, "R1")
    part = image_part(img)
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_part(img) == part                       # deterministic encoding
    msgs = [{"role": "user", "content": [part, {"type": "text", "text": "?"}]}]
    c = make_client(tmp_path, Scripted(completion("ok")))
    r = c.chat(msgs, task="t")
    stored = next((tmp_path / "cache").rglob("*.json")).read_text()
    b64 = part["image_url"]["url"].split(",", 1)[1]
    assert b64 not in stored and "<sha256:" in stored    # cache files stay small
    assert r.key == cache_key("gemma-test", msgs, {"temperature": 0.0, "max_tokens": 512})
    assert log_lines(tmp_path)[0]["n_images"] == 1


# ------------------------------------------------------------------------------- modes and log
def test_record_then_replay(tmp_path):
    t = Scripted(completion("hello"))
    rec = make_client(tmp_path, t, "record")
    msgs = [{"role": "user", "content": "say hello"}]
    r1 = rec.chat(msgs, task="demo")
    assert r1.content == "hello" and not r1.cached and len(t.payloads) == 1
    assert t.payloads[0]["model"] == "gemma-test" and t.payloads[0]["temperature"] == 0

    def boom(_):
        raise AssertionError("replay must not touch the network")
    rep = make_client(tmp_path, boom, "replay")
    r2 = rep.chat(msgs, task="demo")
    assert r2.content == "hello" and r2.cached and r2.key == r1.key
    assert r2.latency_s == pytest.approx(r1.latency_s, abs=1e-3)      # recorded latency is kept
    lines = log_lines(tmp_path)
    assert [x["live"] for x in lines] == [True, False] and [x["cached"] for x in lines] == [False, True]


def test_replay_missing_entry_is_a_clear_error(tmp_path):
    rep = make_client(tmp_path, None, "replay")
    with pytest.raises(CacheMiss) as ei:
        rep.chat([{"role": "user", "content": "never recorded"}], task="demo")
    msg = str(ei.value)
    assert "replay" in msg and "INTERCYCLE_LLM_MODE=record" in msg and "gemma-test" in msg


def test_live_mode_never_reads_or_writes_cache(tmp_path):
    t = Scripted(completion("a"), completion("b"))
    c = make_client(tmp_path, t, "live")
    msgs = [{"role": "user", "content": "x"}]
    assert c.chat(msgs).content == "a" and c.chat(msgs).content == "b"
    assert not (tmp_path / "cache").exists() and len(t.payloads) == 2


def test_call_log_carries_proof_fields(tmp_path):
    c = make_client(tmp_path, Scripted(completion("hi", usage=(123, 45))), "live")
    c.chat([{"role": "user", "content": "x"}], task="probe")
    (rec,) = log_lines(tmp_path)
    assert rec["model"] == "gemma-test" and rec["task"] == "probe" and rec["live"] is True
    assert rec["prompt_tokens"] == 123 and rec["completion_tokens"] == 45 and rec["total_tokens"] == 168
    assert rec["latency_s"] >= 0 and rec["cost_free"] is True and rec["cost_usd"] == 0.0
    assert rec["provider"] == "amd-token-factory" and rec["endpoint_host"] == "api.tokenfactory.amd.com"


def test_config_from_env(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("INTERCYCLE_LLM_"):
            monkeypatch.delenv(k)
    assert LLMConfig.from_env().mode == "replay" and LLMConfig.from_env().model == "google/gemma-4-31B-it"
    monkeypatch.setenv("INTERCYCLE_LLM_BASE_URL", "https://x.amd.com/v1")
    monkeypatch.setenv("INTERCYCLE_LLM_API_KEY", "sk-1")
    monkeypatch.setenv("INTERCYCLE_LLM_MODEL", "gemma-4-26b")
    cfg = LLMConfig.from_env()
    assert (cfg.mode, cfg.model, cfg.provider) == ("record", "gemma-4-26b", "amd-token-factory")
    monkeypatch.setenv("INTERCYCLE_LLM_MODE", "live")
    assert LLMConfig.from_env().mode == "live"


# ------------------------------------------------------------------- structured JSON / fallback
def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nbye') == {"a": [1, 2]}
    assert extract_json('The answer is {"a": {"b": 2}} as requested') == {"a": {"b": 2}}
    for bad in ("", "no json here", None):
        with pytest.raises(ValueError):
            extract_json(bad)


def test_response_format_unsupported_falls_back_to_prompt_json(tmp_path):
    t = Scripted(TransportError("HTTP 400: response_format json_schema is not supported", status=400),
                 completion("Sure!\n```json\n" + json.dumps(VERDICT) + "\n```"),
                 completion(json.dumps(VERDICT)))
    c = make_client(tmp_path, t, "live")
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "check"}]
    r = c.structured(msgs, StateVerdict, name="v", task="t")
    assert r.value.moving is False and r.value.confidence == 0.8 and not r.used_response_format
    assert "response_format" in t.payloads[0] and "response_format" not in t.payloads[1]
    sysmsg = t.payloads[1]["messages"][0]["content"]
    assert sysmsg.startswith("be brief") and "ONLY one JSON object" in sysmsg and '"person_in_zone"' in sysmsg
    # remembered: next structured call goes straight to prompt-only JSON
    c.structured(msgs, StateVerdict)
    assert "response_format" not in t.payloads[2]
    assert any(x.get("event") == "response_format_unsupported" for x in log_lines(tmp_path))


def test_response_format_used_when_supported(tmp_path):
    t = Scripted(completion(json.dumps(VERDICT)))
    r = make_client(tmp_path, t, "live").structured([{"role": "user", "content": "x"}], StateVerdict)
    rf = t.payloads[0]["response_format"]
    assert r.used_response_format and rf["type"] == "json_schema"
    assert "$defs" not in json.dumps(rf) and rf["json_schema"]["schema"]["required"] == ["moving", "person_in_zone", "confidence"]


def test_json_mode_prompt_never_sends_response_format(tmp_path):
    t = Scripted(completion(json.dumps(VERDICT)))
    make_client(tmp_path, t, "live", json_mode="prompt").structured([{"role": "user", "content": "x"}], StateVerdict)
    assert "response_format" not in t.payloads[0]


def test_invalid_replies_are_fed_back_and_retried(tmp_path):
    bad_schema = {**VERDICT, "confidence": 1.7}
    t = Scripted(completion("I think the arm is at home."), completion(json.dumps(bad_schema)),
                 completion(json.dumps(VERDICT)))
    r = make_client(tmp_path, t, "live", json_mode="prompt").structured(
        [{"role": "user", "content": "x"}], StateVerdict, retries=2)
    assert r.attempts == 3 and r.value.confidence == 0.8
    assert r.prompt_tokens == 33 and r.completion_tokens == 21           # summed over attempts
    feedback = t.payloads[2]["messages"][-1]["content"]
    assert "rejected" in feedback and "confidence" in feedback           # validation error is shown to the model
    assert t.payloads[2]["messages"][-2] == {"role": "assistant", "content": json.dumps(bad_schema)}


def test_structured_gives_up_after_retries(tmp_path):
    t = Scripted(*[completion("nope")] * 3)
    with pytest.raises(StructuredOutputError) as ei:
        make_client(tmp_path, t, "live", json_mode="prompt").structured(
            [{"role": "user", "content": "x"}], StateVerdict, retries=2)
    assert ei.value.attempts == 3 and ei.value.raw == "nope"


def test_unrelated_400_is_not_swallowed_as_missing_response_format(tmp_path):
    t = Scripted(TransportError("HTTP 400: this model does not accept image input", status=400))
    with pytest.raises(TransportError):
        make_client(tmp_path, t, "live").structured([{"role": "user", "content": "x"}], StateVerdict)


def test_replay_finds_prompt_only_recording_without_knowing_server_features(tmp_path):
    t = Scripted(TransportError("HTTP 400: response_format not supported", status=400), completion(json.dumps(VERDICT)))
    msgs = [{"role": "user", "content": "check"}]
    make_client(tmp_path, t, "record").structured(msgs, StateVerdict)
    rep = make_client(tmp_path, None, "replay")
    r = rep.structured(msgs, StateVerdict)
    assert r.cached and r.value.notes == "arm at home" and not r.used_response_format


# ----------------------------------------------------------------------------- verify_state
def verdict_transport(**kw):
    return Scripted(completion(json.dumps({**VERDICT, **kw})))


def test_verify_state_person_suppresses_and_disagreement_lowers_confidence(tmp_path):
    t = verdict_transport(moving=True, person_in_zone=True, confidence=0.9, notes="person at left edge")
    c = make_client(tmp_path, t, "live")
    r = verify_state([render(True, (118, 148), "R1"), render(True, (118, 148), "R1")], "WELD_R1",
                     {"moving": False, "confidence": 0.95, "zone_hint": "robot reach envelope"}, client=c, frame_dt_s=0.5)
    assert r.ok and r.suppress_window and r.verdict.person_in_zone
    assert r.adjusted_confidence == pytest.approx(0.95 * 0.1) and r.adjusted_confidence < r.base_confidence
    assert len(r.warnings) == 2 and "person_in_zone" in r.warnings[0]
    # two image parts, in order, then the text; the expectation is NOT shown to the model
    content = t.payloads[0]["messages"][1]["content"]
    assert [p["type"] for p in content] == ["image_url", "image_url", "text"]
    assert "WELD_R1" in content[-1]["text"] and "0.5 s apart" in content[-1]["text"] and "0.95" not in content[-1]["text"]
    assert "never decide" in t.payloads[0]["messages"][0]["content"].lower()


def test_verify_state_agreement_never_raises_confidence(tmp_path):
    c = make_client(tmp_path, verdict_transport(moving=False, confidence=0.99), "live")
    r = verify_state([render(False, None, "R1")], "R1", {"moving": False, "confidence": 0.6}, client=c)
    assert r.adjusted_confidence == 0.6 and not r.suppress_window and r.warnings == []


def test_verify_state_can_only_add_caution():
    """Property: whatever the model says, confidence never goes up and 'person' always suppresses."""
    rng = random.Random(1)
    for i in range(60):
        v = {"moving": rng.random() < .5, "person_in_zone": rng.random() < .5, "confidence": rng.random(), "notes": ""}
        base, exp = rng.random(), rng.random() < .5
        c = GemmaClient(LLMConfig(mode="live", log_path=Path(__import__("tempfile").gettempdir()) / "gemma_prop.jsonl"),
                        transport=Scripted(completion(json.dumps(v))))
        r = verify_state([render(False, None, "R1")], "R1", {"moving": exp, "confidence": base}, client=c)
        assert r.adjusted_confidence <= base + 1e-12
        assert r.suppress_window == v["person_in_zone"]
        assert not hasattr(r, "entry_allowed")


def test_verify_state_fails_open_to_baseline_when_gemma_is_down(tmp_path):
    c = make_client(tmp_path, Scripted(TransportError("HTTP 503: overloaded", status=503)), "live")
    r = verify_state([render(False, None, "R1")], "R1", {"moving": False, "confidence": 0.9}, client=c)
    assert not r.ok and r.adjusted_confidence == 0.9 and not r.suppress_window
    assert r.warnings and "gemma_unavailable" in r.warnings[0]
    c2 = make_client(tmp_path, Scripted(TransportError("HTTP 503", status=503)), "live")
    with pytest.raises(TransportError):
        verify_state([render(False, None, "R1")], "R1", client=c2, raise_on_error=True)


# ------------------------------------------------------------------------------- commission
def test_commission_converts_to_pixels_and_drops_bad_polygons(tmp_path):
    from PIL import Image
    zones = {"zones": [
        {"label": "Weld cell hazard zone", "kind": "hazard_zone", "machine_id": "WELD_R1", "machine_type": "weld_robot",
         "polygon": [[100, 200], [500, 200], [500, 700], [100, 700]], "confidence": 0.8, "notes": "n"},
        {"label": "two points", "kind": "walkway", "polygon": [[0, 0], [10, 10]], "confidence": 0.5},
        {"label": "wrong scale", "kind": "walkway", "polygon": [[0, 0], [4000, 0], [4000, 3000]], "confidence": 0.5},
        {"label": "sliver", "kind": "other", "polygon": [[0, 0], [1000, 0], [1000, 0.1]], "confidence": 0.5},
        {"label": "off-edge ok", "kind": "inspection_target", "polygon": [[-20, 0], [100, 0], [100, 100]], "confidence": 0.4}],
        "warnings": ["glare on the right"]}
    t = Scripted(completion(json.dumps(zones)))
    img = Image.new("RGB", (400, 200), (128, 128, 128))
    p = commission(img, "cam-3", {"machines": ["WELD_R1"]}, client=make_client(tmp_path, t, "live"))
    assert p.image_size == (400, 200) and p.warnings == ["glare on the right"]
    z = p.zones[0]
    assert z.status == "proposed" and z.machine_id == "WELD_R1"
    assert z.polygon_px == [(40.0, 40.0), (200.0, 40.0), (200.0, 140.0), (40.0, 140.0)]
    assert [x.label for x in p.zones] == ["Weld cell hazard zone", "off-edge ok"]
    assert p.zones[1].polygon_px[0][0] == 0.0                              # clamped
    assert len(p.dropped) == 3
    text = t.payloads[0]["messages"][1]["content"][-1]["text"]
    assert "cam-3" in text and "400x200" in text and "WELD_R1" in text
    assert "proposal" in t.payloads[0]["messages"][0]["content"].lower()


# --------------------------------------------------------------------------------- ops agent
def test_parse_time():
    assert parse_time("07:30") == 27000 and parse_time("07:30:15") == 27015 and parse_time(120) == 120.0
    assert parse_time("day1 00:01") == 86460 and parse_time("27000") == 27000.0
    with pytest.raises(ValueError):
        parse_time("soonish")


def make_tools():
    calls = []

    def next_safe_window(machine, before_time):
        calls.append(("nsw", machine, before_time))
        return {"machine": machine, "start": 27120.0, "end": 27150.0, "confidence": 0.91}

    def plan_inspection(targets, deadline):
        calls.append(("plan", targets, deadline))
        return {"order": targets}

    def shift_report(time_range):
        calls.append(("report", time_range))
        return {"inspections": 3}
    return calls, {"next_safe_window": next_safe_window, "plan_inspection": plan_inspection, "shift_report": shift_report}


ANSWER = "Predicted window for WELD_R1: 07:32:00 to 07:32:30 (confidence 0.91). The robot's own check decides entry."


def test_tool_loop_native_calls_tool_then_answers(tmp_path):
    calls, tools = make_tools()
    t = Scripted(completion(tool_calls=[tool_call("next_safe_window", {"machine": "WELD_R1", "before_time": "07:30"})]),
                 completion(ANSWER))
    c = make_client(tmp_path, t, "live")
    r = ops_agent("When can I inspect WELD_R1 before 07:30?", tools, client=c, context="now=07:25:00")
    assert r.mode == "native" and r.answer == ANSWER and r.steps == 2
    assert calls == [("nsw", "WELD_R1", 27000.0)]                       # '07:30' normalised to seconds
    assert r.tool_calls[0]["name"] == "next_safe_window" and r.tool_calls[0]["result"]["confidence"] == 0.91
    p0, p1 = t.payloads
    assert [x["function"]["name"] for x in p0["tools"]] == ["next_safe_window", "plan_inspection", "shift_report"]
    assert p0["tool_choice"] == "auto" and "now=07:25:00" in p0["messages"][0]["content"]
    assert "never decide" in p0["messages"][0]["content"].lower()
    msgs = p1["messages"]
    assert msgs[-2]["role"] == "assistant" and msgs[-2]["tool_calls"][0]["id"] == "call_1"
    assert msgs[-1]["role"] == "tool" and msgs[-1]["tool_call_id"] == "call_1"
    assert json.loads(msgs[-1]["content"])["start"] == 27120.0          # tool result went back to the model


def test_tool_errors_and_unknown_tools_become_observations(tmp_path):
    def broken(machine, before_time):
        raise KeyError("no such machine")
    t = Scripted(completion(tool_calls=[tool_call("next_safe_window", {"machine": "X", "before_time": "08:00"}, "a"),
                                        tool_call("made_up", {}, "b")]),
                 completion("I could not look that up."))
    r = ops_agent("q", {"next_safe_window": broken}, client=make_client(tmp_path, t, "live"))
    obs = {m["tool_call_id"]: json.loads(m["content"]) for m in t.payloads[1]["messages"] if m["role"] == "tool"}
    assert "KeyError" in obs["a"]["error"] and "unknown tool" in obs["b"]["error"]
    assert r.answer == "I could not look that up." and len(r.tool_calls) == 2


def test_tool_budget_forces_final_answer(tmp_path):
    _, tools = make_tools()
    loop = completion(tool_calls=[tool_call("shift_report", {"time_range": ["07:00", "08:00"]})])
    t = Scripted(loop, loop, completion("Summary from the data I have."))
    r = ops_agent("report", tools, client=make_client(tmp_path, t, "live"), max_steps=2)
    assert r.answer == "Summary from the data I have." and any("budget" in w for w in r.warnings)
    assert t.payloads[2]["tool_choice"] == "none"


def react(thought, action, **kw):
    return completion(json.dumps({"thought": thought, "action": action, **kw}))


def test_react_fallback_when_model_lacks_tool_support(tmp_path):
    calls, tools = make_tools()
    t = Scripted(TransportError("HTTP 400: this model does not support tools / function calling", status=400),
                 react("need the window", "call", tool="next_safe_window", args={"machine": "WELD_R1", "before_time": "07:30"}),
                 react("have it", "final", answer=ANSWER))
    r = ops_agent("When?", tools, client=make_client(tmp_path, t, "live", json_mode="prompt"))
    assert r.mode == "react" and r.answer == ANSWER and calls == [("nsw", "WELD_R1", 27000.0)]
    assert "tools" in t.payloads[0] and "tools" not in t.payloads[1]
    sys0 = t.payloads[1]["messages"][0]["content"]
    assert "next_safe_window" in sys0 and '"action": "call"' in sys0
    assert t.payloads[2]["messages"][-1]["content"].startswith("Observation: ")
    assert '"confidence": 0.91' in t.payloads[2]["messages"][-1]["content"]


def test_react_pinned_by_config_never_sends_tools(tmp_path):
    _, tools = make_tools()
    t = Scripted(react("done", "final", answer="nothing needed"))
    r = ops_agent("hi", tools, client=make_client(tmp_path, t, "live", tools_mode="react", json_mode="prompt"))
    assert r.mode == "react" and "tools" not in t.payloads[0]


def test_text_tool_call_is_detected_as_missing_tool_support(tmp_path):
    _, tools = make_tools()
    t = Scripted(completion('{"tool": "next_safe_window", "args": {"machine": "WELD_R1"}}'),
                 react("x", "final", answer="ok"))
    r = ops_agent("hi", tools, client=make_client(tmp_path, t, "live", json_mode="prompt"))
    assert r.mode == "react" and r.answer == "ok"
    with pytest.raises(ToolsUnsupported):       # pinned native: surface the problem instead of guessing
        ops_agent("hi", tools, client=make_client(
            tmp_path, Scripted(TransportError("HTTP 400: tools not supported", status=400)), "live", tools_mode="native"))


def test_ops_agent_replays_a_recorded_conversation(tmp_path):
    calls, tools = make_tools()
    t = Scripted(completion(tool_calls=[tool_call("next_safe_window", {"machine": "WELD_R1", "before_time": "07:30"})]),
                 completion(ANSWER))
    r1 = ops_agent("When?", tools, client=make_client(tmp_path, t, "record"))
    calls.clear()
    r2 = ops_agent("When?", tools, client=make_client(tmp_path, None, "replay"))
    assert r2.answer == r1.answer == ANSWER and r2.cached and calls == [("nsw", "WELD_R1", 27000.0)]


# ----------------------------------------------------------------- harness + fixture + real SDK
def test_fixture_and_eval_harness_in_replay_mode(tmp_path):
    import gemma_eval_verify as ev
    labels = make_fixture(tmp_path / "fx")
    assert len(labels) == 6 and sum(v["person_in_zone"] for v in labels.values()) == 2
    assert sum(v["moving"] for v in labels.values()) == 3
    assert len(list((tmp_path / "fx" / "frames").glob("*.png"))) == 6

    def client(mode, transport=None):
        cfg = LLMConfig(model=MOCK_MODEL, mode=mode, provider="mock", cache_dir=tmp_path / "cache",
                        log_path=tmp_path / "calls.jsonl")
        return GemmaClient(cfg, transport=transport)
    rec = ev.evaluate(tmp_path / "fx", client("record", FixtureHeuristicTransport()))
    assert rec["metrics"]["accuracy_both"] == 1.0 and rec["metrics"]["latency_source"] == "live"
    rep = ev.evaluate(tmp_path / "fx", client("replay"), concurrency=3)      # no transport at all
    m = rep["metrics"]
    assert m["n"] == m["n_ok"] == 6 and m["errors"] == 0 and m["accuracy_both"] == 1.0
    assert m["person_in_zone"]["precision"] == 1.0 and m["person_in_zone"]["recall"] == 1.0
    assert m["latency_source"] == "recorded (replay)" and m["latency_s"]["p95"] >= m["latency_s"]["p50"] > 0
    assert m["throughput_rps_sequential"] > 0 and m["throughput_rps_wall"] is None
    assert "MOCK STAND-IN" in ev.format_report(rep)
    # replay with an empty cache reports every frame as an error instead of crashing
    empty = ev.evaluate(tmp_path / "fx", GemmaClient(LLMConfig(model=MOCK_MODEL, mode="replay",
                        cache_dir=tmp_path / "none", log_path=tmp_path / "x.jsonl")))
    assert empty["metrics"]["errors"] == 6 and "CacheMiss" in empty["rows"][0]["error"]


def test_eval_metrics_math():
    import gemma_eval_verify as ev
    def row(tp_, pp_, tm=True, pm=True, lat=1.0):
        return {"ok": True, "truth_person": tp_, "pred_person": pp_, "truth_moving": tm, "pred_moving": pm,
                "latency_s": lat, "cached": True, "confidence": .5, "attempts": 1}
    rows = [row(True, True), row(True, True), row(False, True), row(True, False), row(False, False, lat=5.0)]
    rows.append({"ok": False, "error": "x"})
    m = ev.compute_metrics(rows)
    p = m["person_in_zone"]
    assert (p["tp"], p["fp"], p["fn"], p["tn"]) == (2, 1, 1, 1)
    assert p["precision"] == pytest.approx(2 / 3) and p["recall"] == pytest.approx(2 / 3)
    assert m["accuracy_person_in_zone"] == pytest.approx(3 / 5) and m["n"] == 6 and m["errors"] == 1
    assert m["latency_s"]["p50"] == 1.0 and m["latency_s"]["p95"] == pytest.approx(4.2)


class _Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send({"object": "list", "data": [{"id": "other-model", "object": "model"},
                                               {"id": "google/gemma-4-31B-it", "object": "model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        parts = [p for m in body["messages"] if isinstance(m["content"], list) for p in m["content"]]
        if body.get("model") == "reject":
            return self._send({"error": {"message": "nope", "type": "invalid_request_error"}}, 400)
        if "tools" in body:
            out = completion(tool_calls=[tool_call("get_time", {})])
        elif any(p["type"] == "image_url" for p in parts):
            out = completion(json.dumps({"moving": True, "person_in_zone": True, "confidence": 0.9, "notes": "t"}))
        elif "response_format" in body:
            out = completion('{"ok": true}')
        else:
            out = completion("ready")
        self._send(out)


@pytest.fixture
def server(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    srv.seen = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_real_openai_sdk_transport_against_local_server(server, tmp_path):
    srv, base = server
    cfg = LLMConfig(base_url=base, api_key="secret", model="google/gemma-4-31B-it", mode="live",
                    cache_dir=tmp_path / "c", log_path=tmp_path / "calls.jsonl", max_retries=0)
    c = GemmaClient(cfg)
    r = verify_state([render(True, (118, 148), "R1")], "R1", {"moving": True}, client=c, raise_on_error=True)
    assert r.verdict.person_in_zone and r.suppress_window
    seen = srv.seen[0]
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer secret"
    url = next(p for p in seen["body"]["messages"][1]["content"] if p["type"] == "image_url")["image_url"]["url"]
    assert url.startswith("data:image/png;base64,") and seen["body"]["response_format"]["type"] == "json_schema"
    assert OpenAITransport(cfg).list_models() == ["other-model", "google/gemma-4-31B-it"]
    bad = GemmaClient(LLMConfig(base_url=base, api_key="k", model="reject", mode="live", log_path=tmp_path / "l2.jsonl",
                                max_retries=0))
    with pytest.raises(TransportError) as ei:
        bad.chat([{"role": "user", "content": "x"}])
    assert ei.value.status == 400 and "nope" in str(ei.value)
    assert json.loads((tmp_path / "l2.jsonl").read_text().splitlines()[0])["ok"] is False


def test_probe_script_end_to_end_on_local_server(server, monkeypatch, tmp_path, capsys):
    import gemma_probe
    _, base = server
    monkeypatch.setenv("INTERCYCLE_LLM_BASE_URL", base)
    monkeypatch.setenv("INTERCYCLE_LLM_API_KEY", "secret")
    monkeypatch.setenv("INTERCYCLE_LLM_LOG", str(tmp_path / "probe.jsonl"))
    monkeypatch.delenv("INTERCYCLE_LLM_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["gemma_probe.py"])
    assert gemma_probe.main() == 0
    out = capsys.readouterr().out
    assert "using model: google/gemma-4-31B-it" in out and out.count("[OK ]") == 4
    assert "INTERCYCLE_LLM_MODEL=google/gemma-4-31B-it" in out
    assert gemma_probe.pick_model(["a", "gemma-4-e4b-it", "gemma-4-26b-a4b-it"], None) == "gemma-4-26b-a4b-it"
    assert gemma_probe.pick_model(["x"], None) is None
