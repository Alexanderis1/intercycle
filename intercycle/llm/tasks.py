"""The three Gemma 4 tasks: state verification, commissioning proposals, ops agent.

SAFETY BOUNDARY (also stated in every prompt and in docs/GEMMA.md):
Gemma never decides robot entry. It has no output that grants permission, shortens a window, or
raises a confidence. It can only (a) lower a confidence, (b) add warnings, (c) flag "person in zone"
which suppresses a published safe window. Entry decisions stay with the SIPP planner, the robot's own
local check and the plant's light curtains / area scanners. If Gemma is unreachable or returns junk the
system behaves exactly as it would without Gemma (the extra caution simply is not applied).
"""
from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, field_validator, model_validator

from .client import (
    CacheMiss, GemmaClient, LLMError, ToolsUnsupported, default_client, extract_json, image_bytes, image_size,
    user_message,
)

BOUNDARY = (
    "You never decide whether a robot, person or vehicle may enter a zone. A certified safety system "
    "(light curtains, area scanners) and the robot's own checks make every entry decision. "
    "Your output can only add caution: lower a confidence or add a warning, never grant permission."
)


# =========================================================================== (a) verify_state
class StateVerdict(BaseModel):
    moving: bool
    person_in_zone: bool
    confidence: float = Field(ge=0.0, le=1.0, description="0..1 confidence in this whole verdict")
    notes: str = ""

    @field_validator("notes", mode="before")
    @classmethod
    def _trim(cls, v: Any) -> Any:
        return v[:300] if isinstance(v, str) else v


VERIFY_SYSTEM = f"""You are a vision checker inside an industrial safety-support system in an automotive body shop.
You see crops around the hazard zone of ONE automatic machine (robot reach envelope, lifter drop zone, hemming or sealer cell), taken by a fixed ceiling camera. Frames are in chronological order.
{BOUNDARY}

Report only what is visible:
- moving: true if the machine's moving parts are in motion between frames, show motion blur, or (single frame) are clearly away from their rest/home pose. false only if the parts look still and at rest.
- person_in_zone: true if any part of a human body (including a partly occluded person, a hi-vis vest or a hand or leg) is inside or overlaps the hazard zone. People outside the zone do not count.
- confidence: your confidence in the whole verdict, 0 to 1. A single frame cannot prove stillness: keep confidence at or below 0.7 then.
When unsure, take the cautious answer (moving=true, person_in_zone=true) and set confidence below 0.5.
notes: one short factual sentence about what you see. No instructions, no advice to robots or people."""


@dataclass
class VerifyResult:
    """Verdict plus how it may influence the observer. There is deliberately no 'entry_allowed' field."""
    machine_id: str
    ok: bool                         # False: Gemma unavailable/invalid, nothing was changed
    verdict: StateVerdict | None
    base_confidence: float           # the observer's own confidence before Gemma
    adjusted_confidence: float       # always <= base_confidence
    suppress_window: bool            # True when a person is in the zone: do not publish a safe window
    warnings: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cached: bool = False
    attempts: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def verify_state(images: Sequence[Any], machine_id: str, expected: Mapping[str, Any] | None = None, *,
                 client: GemmaClient | None = None, frame_dt_s: float | None = None,
                 raise_on_error: bool = False) -> VerifyResult:
    """Ask Gemma what a crop around ``machine_id``'s hazard zone shows.

    ``images``  one or more crops, oldest first (paths, bytes, PIL images or arrays).
    ``expected`` the observer's belief, e.g. ``{"moving": False, "confidence": 0.93}``; optional keys
                ``machine_type`` and ``zone_hint`` are shown to the model (neutral facts). The belief itself
                is NOT shown to the model (to avoid anchoring) and is only compared in code afterwards.

    Policy applied to the verdict (can only make the observer more cautious):
      * person_in_zone -> suppress_window=True (even at low model confidence) and a warning;
      * model disagrees with the expected moving state -> adjusted_confidence = base * min(1 - c, 0.5);
      * agreement never raises confidence above the observer's own value.
    """
    exp = dict(expected or {})
    base = float(exp.get("confidence", 1.0))
    client = client or default_client()
    facts = f"Machine: {machine_id}"
    if exp.get("machine_type"):
        facts += f" ({exp['machine_type']})"
    if exp.get("zone_hint"):
        facts += f". Hazard zone: {exp['zone_hint']}"
    n = len(images)
    facts += f". {n} frame{'s' if n != 1 else ''}, oldest first"
    if frame_dt_s and n > 1:
        facts += f", {frame_dt_s:g} s apart"
    facts += ". Return the JSON verdict."
    msgs = [{"role": "system", "content": VERIFY_SYSTEM}, user_message(facts, images)]
    try:
        r = client.structured(msgs, StateVerdict, name="state_verdict", task="verify_state", max_tokens=256)
    except LLMError as e:
        if raise_on_error:
            raise
        return VerifyResult(machine_id, False, None, base, base, False,
                            [f"gemma_unavailable: {type(e).__name__}: {str(e)[:160]}"], error=str(e)[:300])
    v = r.value
    warnings: list[str] = []
    adjusted = base
    suppress = False
    if v.person_in_zone:
        suppress = True
        warnings.append(f"person_in_zone (model confidence {v.confidence:.2f}): {v.notes}".rstrip(": "))
    exp_moving = exp.get("moving")
    if exp_moving is not None and bool(exp_moving) != v.moving:
        adjusted = min(adjusted, base * min(1.0 - v.confidence, 0.5))
        warnings.append(f"model sees moving={v.moving} but observer expected moving={bool(exp_moving)} "
                        f"(model confidence {v.confidence:.2f})")
    return VerifyResult(machine_id, True, v, base, min(adjusted, base), suppress, warnings,
                        latency_s=r.latency_s, cached=r.cached, attempts=r.attempts)


# ============================================================================ (b) commission
ZONE_KINDS = ("hazard_zone", "machine", "walkway", "vehicle_lane", "inspection_target", "exclusion", "other")
MACHINE_TYPES = ("weld_robot", "lifter", "hemming_cell", "sealer_robot", "conveyor", "agv_lane",
                 "tugger_lane", "forklift_lane", "rack", "data_hall_door", "other")


class RawZone(BaseModel):
    label: str
    kind: Literal[ZONE_KINDS]
    machine_id: str | None = None
    machine_type: Literal[MACHINE_TYPES] | None = None
    polygon: list[list[float]] = Field(description="[[x,y],...] integers 0..1000, x to the right, y down, "
                                                   "relative to image width/height; at least 3 vertices")
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class RawCommission(BaseModel):
    zones: list[RawZone] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


COMMISSION_SYSTEM = f"""You help a commissioning engineer set up a ceiling camera in an automotive body shop (BIW) with an on-site data hall.
Propose regions of interest in the image: machine hazard zones (robot reach envelope, lifter drop zone, hemming or sealer cell), the machines themselves, walkways, AGV/tugger/forklift lanes, data-hall doors, and inspection targets (parts of a machine an inspection robot should look at).
{BOUNDARY}
Everything you output is a PROPOSAL that a human reviews and edits. You do not approve anything and you do not mark any area safe.

Rules:
- Coordinates: each polygon is a list of [x, y] vertices with integers 0..1000, x to the right, y down, relative to the image width and height (so [0,0] is the top-left corner and [1000,1000] the bottom-right). At least 3 vertices, in order around the shape, no repeated first vertex.
- Hazard zones must cover the full area a machine can reach, with the machine inside, not just the machine body. When unsure, draw the zone larger, never smaller.
- If a floor-plan hint lists machine ids, use only those ids in machine_id; otherwise use null. Never invent ids.
- label: short and specific (for example "Weld cell hazard zone"). kind and machine_type must come from the allowed values.
- confidence 0..1 per item. Skip what you cannot see. At most 12 items.
- warnings: list anything that limits reliability (occlusion, glare, fisheye distortion, partly visible machines)."""


@dataclass
class Zone:
    label: str
    kind: str
    machine_id: str | None
    machine_type: str | None
    polygon_px: list[tuple[float, float]]    # image pixel coordinates
    polygon_norm: list[tuple[float, float]]  # 0..1
    confidence: float
    notes: str
    status: str = "proposed"                 # always; a human must confirm


@dataclass
class CommissionProposal:
    camera_id: str
    image_size: tuple[int, int]
    zones: list[Zone]
    warnings: list[str]
    dropped: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cached: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def _poly_area(pts: Sequence[Sequence[float]]) -> float:
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def commission(image: Any, camera_id: str, floor_plan_hint: str | Mapping | Sequence | None = None, *,
               client: GemmaClient | None = None, min_area_frac: float = 0.0005) -> CommissionProposal:
    """Propose zones/ROIs/machine types/inspection targets as polygons in image pixel coordinates."""
    client = client or default_client()
    b, _ = image_bytes(image)
    w, h = image_size(b)
    hint = ""
    if floor_plan_hint:
        hint = floor_plan_hint if isinstance(floor_plan_hint, str) else json.dumps(floor_plan_hint, default=str)
    text = f"Camera: {camera_id}. Image size {w}x{h} px (you answer in 0..1000 relative units)."
    text += f"\nFloor-plan hint: {hint}" if hint else "\nNo floor-plan hint is available."
    text += "\nReturn the JSON proposal."
    msgs = [{"role": "system", "content": COMMISSION_SYSTEM}, user_message(text, [b])]
    r = client.structured(msgs, RawCommission, name="commission_proposal", task="commission", max_tokens=2000)
    zones: list[Zone] = []
    dropped: list[str] = []
    warnings = list(r.value.warnings)
    for z in r.value.zones:
        pts = [(float(p[0]), float(p[1])) for p in z.polygon if len(p) == 2]
        if len(pts) != len(z.polygon) or len(pts) < 3:
            dropped.append(f"{z.label}: needs at least 3 [x,y] vertices")
            continue
        if any(not (-50 <= c <= 1050) for p in pts for c in p):
            dropped.append(f"{z.label}: coordinates outside 0..1000 (wrong scale?)")
            continue
        pts = [(min(max(x, 0.0), 1000.0), min(max(y, 0.0), 1000.0)) for x, y in pts]
        if _poly_area(pts) / 1e6 < min_area_frac:
            dropped.append(f"{z.label}: polygon too small or degenerate")
            continue
        norm = [(x / 1000.0, y / 1000.0) for x, y in pts]
        zones.append(Zone(z.label, z.kind, z.machine_id, z.machine_type,
                          [(round(x * w, 1), round(y * h, 1)) for x, y in norm], norm,
                          z.confidence, z.notes))
    return CommissionProposal(camera_id, (w, h), zones, warnings, dropped, r.latency_s, r.cached)


# =============================================================================== (c) ops agent
def parse_time(v: Any) -> float:
    """Seconds since 00:00 of sim day 0 from a number, numeric string, or 'HH:MM[:SS]' / 'day1 07:30'."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"(?:day\s*(\d+)[\s,]*)?(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?", s, re.I)
    if not m:
        raise ValueError(f"cannot parse time {v!r}: use seconds since 00:00 or HH:MM[:SS]")
    return int(m.group(1) or 0) * 86400 + int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4) or 0)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Any]
    normalize: Callable[[dict], dict] | None = None

    def openai(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


def _norm_next_safe_window(a: dict) -> dict:
    return {"machine": str(a["machine"]), "before_time": parse_time(a["before_time"])}


def _norm_plan_inspection(a: dict) -> dict:
    t = a["targets"]
    if isinstance(t, str):
        t = [x.strip() for x in t.split(",") if x.strip()]
    return {"targets": [str(x) for x in t], "deadline": parse_time(a["deadline"])}


def _norm_shift_report(a: dict) -> dict:
    r = a["time_range"]
    if isinstance(r, str):
        r = re.split(r"\s*(?:-|,|to)\s*", r.strip(), maxsplit=1)
    if isinstance(r, Mapping):
        r = [r.get("start"), r.get("end")]
    if len(r) != 2:
        raise ValueError("time_range must be [start, end]")
    return {"time_range": (parse_time(r[0]), parse_time(r[1]))}


_TIME = "seconds since 00:00 of the sim day, or 'HH:MM[:SS]'"
DEFAULT_TOOL_SPECS: dict[str, dict] = {
    "next_safe_window": dict(
        description="Predicted next safe window (machine idle, hazard zone free) for one machine that starts "
                    "before a given time. Returns start, end, confidence and prediction age. A prediction only.",
        parameters={"type": "object", "properties": {
            "machine": {"type": "string", "description": "machine id, e.g. 'WELD_R1'"},
            "before_time": {"type": "string", "description": f"latest acceptable window start: {_TIME}"}},
            "required": ["machine", "before_time"]},
        normalize=_norm_next_safe_window),
    "plan_inspection": dict(
        description="Plan an inspection round over several targets so that each is visited inside a predicted "
                    "safe window before the deadline. Returns the visit order, planned entry times and any "
                    "targets that do not fit.",
        parameters={"type": "object", "properties": {
            "targets": {"type": "array", "items": {"type": "string"}, "description": "target / machine ids"},
            "deadline": {"type": "string", "description": f"all visits must finish by: {_TIME}"}},
            "required": ["targets", "deadline"]},
        normalize=_norm_plan_inspection),
    "shift_report": dict(
        description="Summary of what happened in a time range: inspections done, holds, trips, line time lost, "
                    "per-machine busy fraction and anomalies.",
        parameters={"type": "object", "properties": {
            "time_range": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2,
                           "description": f"[start, end], each {_TIME}"}},
            "required": ["time_range"]},
        normalize=_norm_shift_report),
}

_PY2JSON = {float: "number", int: "integer", str: "string", bool: "boolean", list: "array", dict: "object"}


def build_tools(funcs: Mapping[str, Callable[..., Any] | ToolSpec]) -> dict[str, ToolSpec]:
    """Name -> ToolSpec. The three known tools get hand-written schemas; any other callable gets one
    derived from its signature and docstring."""
    out: dict[str, ToolSpec] = {}
    for name, f in funcs.items():
        if isinstance(f, ToolSpec):
            out[name] = f
        elif name in DEFAULT_TOOL_SPECS:
            out[name] = ToolSpec(name=name, fn=f, **DEFAULT_TOOL_SPECS[name])
        else:
            sig = inspect.signature(f)
            props = {p: {"type": _PY2JSON.get(a.annotation, "string")} for p, a in sig.parameters.items()}
            req = [p for p, a in sig.parameters.items() if a.default is inspect.Parameter.empty]
            out[name] = ToolSpec(name, (inspect.getdoc(f) or name).split("\n")[0],
                                 {"type": "object", "properties": props, "required": req}, f)
    return out


OPS_SYSTEM = f"""You are the operations assistant of InterCycle, a system that learns when automatic machines in an automotive body shop are idle so that visiting inspection robots can work between machine cycles instead of during shutdowns. You answer questions from maintenance and reliability staff.
Rules:
1. Use the tools for every fact about machines, safe windows, inspection plans or shift history. Never invent machine ids, times, counts or durations. If a tool returns nothing or an error, say so.
2. The tools are read-only. {BOUNDARY} A safe window is a prediction: report it as one, with the start, end and confidence the tool returned, and say that the robot's own camera check and the plant safety system make the final entry decision. Never write "the robot may enter now" or any equivalent.
3. If the data is uncertain, old or missing, say so and recommend waiting for confirmation rather than guessing.
4. Times are seconds since 00:00 of the simulated day. Write them as HH:MM:SS in your answer.
5. Be concise: at most six sentences or a short list."""

MAX_TOOL_RESULT_CHARS = 8000


@dataclass
class OpsAnswer:
    answer: str
    mode: str                                   # 'native' tool calls or 'react' JSON loop
    tool_calls: list[dict] = field(default_factory=list)   # {name, args, result|error}
    steps: int = 0
    warnings: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cached: bool = False


def _jsonable(o: Any) -> Any:
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def _dump(result: Any) -> str:
    s = json.dumps(result, default=_jsonable, ensure_ascii=True)
    return s if len(s) <= MAX_TOOL_RESULT_CHARS else s[:MAX_TOOL_RESULT_CHARS] + '..."(truncated)'


def _run_tool(spec: ToolSpec | None, name: str, args: Any) -> tuple[Any, str | None]:
    if spec is None:
        return None, f"unknown tool {name!r}"
    try:
        if isinstance(args, str):
            args = extract_json(args) if args.strip() else {}
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
        if spec.normalize:
            args = spec.normalize(args)
        return spec.fn(**args), None
    except Exception as e:  # noqa: BLE001 - tool errors go back to the model as an observation
        return None, f"{type(e).__name__}: {e}"


def _looks_like_text_tool_call(content: str, names: Sequence[str]) -> bool:
    """The server ignored `tools` and the model wrote a call as plain text."""
    if not any(n in content for n in names):
        return False
    try:
        obj = extract_json(content)
    except ValueError:
        return bool(re.search(r"(tool_call|call:|<\|tool)", content))
    return isinstance(obj, dict) and any(k in obj for k in ("tool", "name", "tool_name", "action"))


def _system(context: str | None) -> str:
    return OPS_SYSTEM + (f"\n\nContext:\n{context}" if context else "")


def _ops_native(client: GemmaClient, question: str, specs: dict[str, ToolSpec], max_steps: int,
                context: str | None) -> OpsAnswer:
    tools = [s.openai() for s in specs.values()]
    msgs: list[dict] = [{"role": "system", "content": _system(context)}, {"role": "user", "content": question}]
    out = OpsAnswer("", "native")
    cached: list[bool] = []
    for _ in range(max_steps):
        try:
            res = client.chat(msgs, tools=tools, tool_choice="auto", temperature=0.0, max_tokens=700, task="ops_agent")
        except CacheMiss:
            if out.steps == 0 and client.cfg.mode == "replay" and client.cfg.tools_mode == "auto":
                raise ToolsUnsupported("no native-tool recording in the cache; trying the ReAct recording") from None
            raise
        out.latency_s += res.latency_s
        cached.append(res.cached)
        out.steps += 1
        if not res.tool_calls:
            if out.steps == 1 and client.cfg.tools_mode == "auto" and _looks_like_text_tool_call(res.content, list(specs)):
                raise ToolsUnsupported("model wrote a tool call as plain text; tools are not honoured")
            out.answer, out.cached = res.content.strip(), all(cached)
            return out
        msgs.append(res.message)
        for tc in res.tool_calls:
            name = tc["function"]["name"]
            result, err = _run_tool(specs.get(name), name, tc["function"]["arguments"])
            out.tool_calls.append({"name": name, "args": tc["function"]["arguments"],
                                   **({"error": err} if err else {"result": result})})
            msgs.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": _dump({"error": err} if err else result)})
    # budget exhausted: ask for an answer with tool use switched off
    msgs.append({"role": "user", "content": "Tool budget used up. Answer now using only the tool results above."})
    res = client.chat(msgs, tools=tools, tool_choice="none", temperature=0.0, max_tokens=700, task="ops_agent")
    out.latency_s += res.latency_s
    cached.append(res.cached)
    out.answer, out.cached = res.content.strip(), all(cached)
    out.warnings.append(f"tool budget of {max_steps} steps exhausted")
    return out


class ReActStep(BaseModel):
    thought: str = ""
    action: Literal["call", "final"]
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    answer: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "ReActStep":
        if self.action == "call" and not self.tool:
            raise ValueError("action 'call' needs a 'tool'")
        if self.action == "final" and not (self.answer or "").strip():
            raise ValueError("action 'final' needs a non-empty 'answer'")
        return self


def _react_protocol(specs: dict[str, ToolSpec]) -> str:
    lines = ["You have these read-only tools (name: description; args = JSON Schema of the arguments):"]
    for s in specs.values():
        lines.append(f"- {s.name}: {s.description}\n  args: {json.dumps(s.parameters, separators=(',', ':'))}")
    lines.append(
        "On every turn reply with ONE JSON object and nothing else:\n"
        '  {"thought": "...", "action": "call", "tool": "<tool name>", "args": {...}}   to use a tool; '
        'the result comes back as the next user message starting with "Observation:"\n'
        '  {"thought": "...", "action": "final", "answer": "<answer for the user>"}   when you can answer.\n'
        "Call at most one tool per turn.")
    return "\n".join(lines)


def _ops_react(client: GemmaClient, question: str, specs: dict[str, ToolSpec], max_steps: int,
               context: str | None) -> OpsAnswer:
    msgs: list[dict] = [{"role": "system", "content": _system(context) + "\n\n" + _react_protocol(specs)},
                        {"role": "user", "content": question}]
    out = OpsAnswer("", "react")
    cached: list[bool] = []
    for i in range(max_steps + 1):
        if i == max_steps:
            msgs.append({"role": "user", "content": "Tool budget used up. Reply now with action 'final'."})
            out.warnings.append(f"tool budget of {max_steps} steps exhausted")
        r = client.structured(msgs, ReActStep, name="react_step", task="ops_agent_react", max_tokens=700)
        out.latency_s += r.latency_s
        cached.append(r.cached)
        out.steps += 1
        step = r.value
        if step.action == "final":
            out.answer, out.cached = (step.answer or "").strip(), all(cached)
            return out
        if i == max_steps:
            break
        name = step.tool or ""
        result, err = _run_tool(specs.get(name), name, step.args)
        out.tool_calls.append({"name": name, "args": json.dumps(step.args), **({"error": err} if err else {"result": result})})
        msgs.append({"role": "assistant", "content": json.dumps(step.model_dump(exclude_none=True))})
        msgs.append({"role": "user", "content": "Observation: " + _dump({"error": err} if err else result)})
    out.answer = "No conclusion within the tool budget; please ask again or narrow the question."
    out.cached = all(cached)
    return out


def ops_agent(question: str, tools: Mapping[str, Callable[..., Any] | ToolSpec], *,
              client: GemmaClient | None = None, max_steps: int = 6, context: str | None = None) -> OpsAnswer:
    """Answer an operations question by calling the supplied (read-only) Python functions.

    Uses OpenAI-style native tool calls; if the endpoint/model rejects ``tools`` (or writes calls as
    plain text) it falls back to a ReAct JSON loop. ``INTERCYCLE_LLM_TOOLS=native|react`` pins the mode.
    ``context`` is free text with facts to ground the answer (current time, machine ids).
    """
    client = client or default_client()
    specs = build_tools(tools)
    mode = client.cfg.tools_mode
    if mode in ("auto", "native"):
        try:
            return _ops_native(client, question, specs, max_steps, context)
        except ToolsUnsupported:
            if mode == "native":
                raise
    return _ops_react(client, question, specs, max_steps, context)
