"""Gemma 4 as the observer's semantic layer (DIRECTION §6): naming discovered phases / line states (G2) and
triaging change-detection blobs (G3). Commissioning (G1) is `tasks.commission`.

SAFETY BOUNDARY (same as tasks.py): Gemma output is advisory text for the operator and the dashboard.
* A phase name is a caption. It never changes a bin, a period, a window or a broadcast box.
* A triage label never frees space. Cells flagged by change detection stay UNKNOWN (treated as occupied) until the
  observer itself has learned the new cycle, whatever Gemma says; `apply_triage` enforces it and a test checks it over
  random verdicts. The only effects a triage can have are warnings (e.g. "re-calibrate camera 6").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, field_validator

from .client import GemmaClient, LLMError, default_client, user_message
from .tasks import BOUNDARY

# ================================================================================= G2: naming discovered phases
PHASE_VOCAB: dict[str, dict[str, str]] = {
    "weld": {
        "weld": "the robot arm's tool is over the car body on the conveyor (spot welding)",
        "tip_dress": "the arm has swung away from the car body to the small tip-dresser stand beside the robot base",
        "transfer": "the arm is moving back from the dresser / car towards its home pose (return / transfer move)",
        "idle": "the arm rests at its home pose and nothing moves (waiting, held or stopped)",
    },
    "lift": {
        "on_floor": "the lifter carriage is down near the floor (lowering, set down or raising a car body)",
        "clear": "the carriage is up high under the overhead frame, the floor area below is clear",
    },
    "cell": {
        "approach": "the arm moves from home towards the part on the fixture table, before the tool path starts",
        "path": "the arm's tool follows the part's edge on the fixture table (hemming / sealing path)",
        "return": "the arm moves back from the part to its home pose",
        "idle": "the arm rests at its home pose and nothing moves",
    },
}
# truth names (scripts/phase_discovery.py, coarse level) -> the vocabulary above
TRUTH_TO_VOCAB = {"weld": {"weld": "weld", "tip_dress": "tip_dress", "return": "transfer", "idle": "idle"},
                  "lift": {"on_floor": "on_floor", "clear": "clear"},
                  "cell": {"approach": "approach", "path": "path", "return": "return", "idle": "idle"}}


class PhaseLabel(BaseModel):
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""

    @field_validator("name", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> Any:
        return v.strip().lower().replace("-", "_").replace(" ", "_") if isinstance(v, str) else v

    @field_validator("notes", mode="before")
    @classmethod
    def _trim(cls, v: Any) -> Any:
        return v[:300] if isinstance(v, str) else v


PHASE_SYSTEM = f"""You label the phases of automatic machines in an automotive body shop, seen by a fixed ceiling camera.
An unsupervised observer has split one machine's repeating cycle into anonymous segments. You see keyframes taken at different times while the machine was inside ONE segment. The machine whose hazard zone is the hatched rectangle in the centre of the image is the one to label; ignore neighbouring machines.
{BOUNDARY}
Your label is only a caption for people; it changes nothing in the system.
Answer with the single best name from the allowed list (exact spelling), a confidence 0..1, and one short factual sentence in notes describing the pose you see. Motion blur on a part means it was moving during the exposure."""


@dataclass
class PhaseNaming:
    machine: str
    bin: int
    name: str | None            # None: Gemma unavailable or answered outside the vocabulary
    confidence: float
    notes: str = ""
    ok: bool = True
    latency_s: float = 0.0
    cached: bool = False
    attempts: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def name_phase(images: Sequence[Any], machine_id: str, family: str, machine_type: str | None = None, *,
               client: GemmaClient | None = None, context: str | None = None) -> PhaseNaming:
    """Name one discovered sub-phase of ``machine_id`` from 1..N keyframes taken inside it."""
    client = client or default_client()
    vocab = PHASE_VOCAB[family]
    opts = "\n".join(f"- {k}: {v}" for k, v in vocab.items())
    text = (f"Machine {machine_id}" + (f" ({machine_type})" if machine_type else "") +
            f". {len(images)} keyframe{'s' if len(images) != 1 else ''} from the same segment."
            + (f" {context}" if context else "") +
            f"\nAllowed names:\n{opts}\nReturn JSON: {{\"name\": <one allowed name>, \"confidence\": <0..1>, \"notes\": <sentence>}}.")
    msgs = [{"role": "system", "content": PHASE_SYSTEM}, user_message(text, images)]
    try:
        r = client.structured(msgs, PhaseLabel, name="phase_label", task="name_phase", max_tokens=200)
    except LLMError as e:
        return PhaseNaming(machine_id, -1, None, 0.0, ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
    v = r.value
    name = v.name if v.name in vocab else None
    return PhaseNaming(machine_id, -1, name, v.confidence if name else 0.0, v.notes, ok=name is not None,
                       latency_s=r.latency_s, cached=r.cached, attempts=r.attempts,
                       error=None if name else f"name outside vocabulary: {v.name!r}")


# ------------------------------------------------------------------------------------------- line states
LINE_VOCAB = {
    "running": "normal production: stations cycle, the line index is on time, normal people flow",
    "andon": "an unplanned line stop: the line index is overdue / the line is blocked while people are present at work",
    "break": "a planned break: machines and line idle, many people leave the stations and walk (to rest areas)",
    "shift_change": "a shift change: line mostly idle, a crowd of people at once (both shifts on the floor)",
}
LINE_FEAT_DOC = {
    "index_overdue": "how overdue the conveyor index is, in takt units (0 = on time, >1 = late)",
    "line_working": "fraction of line stations seen cycling (0..1)",
    "line_blocked": "fraction of line stations stopped mid-cycle (0..1)",
    "cells_working": "fraction of off-line cells seen cycling (0..1)",
    "people_excess": "people on the floor above the usual level for running production (z-score like)",
    "people_moving": "fraction of tracked people walking (0..1)",
    "crowd_excess": "excess of people concentrated in aisles / exits (z-score like)",
}

LINE_SYSTEM = f"""You label production states of an automotive body-shop line. An unsupervised observer clustered the plant's camera-derived features into a few recurring states; you get one cluster's centre (feature averages) plus how long its episodes last, and you name it.
{BOUNDARY}
Your label is only a caption for people; it changes nothing in the system.
Answer with the single best name from the allowed list (exact spelling), a confidence 0..1, and one short sentence in notes explaining which features decided it."""


def name_line_state(centre: Mapping[str, float], *, median_run_s: float | None = None, share: float | None = None,
                    client: GemmaClient | None = None) -> PhaseNaming:
    client = client or default_client()
    feats = "\n".join(f"- {k} = {v:.2f}  ({LINE_FEAT_DOC.get(k, '')})" for k, v in centre.items())
    opts = "\n".join(f"- {k}: {v}" for k, v in LINE_VOCAB.items())
    text = f"Cluster centre:\n{feats}\n"
    if median_run_s is not None:
        text += f"Median episode length: {median_run_s:.0f} s.\n"
    if share is not None:
        text += f"Share of observed time: {100 * share:.1f} %.\n"
    text += f"Allowed names:\n{opts}\nReturn JSON: {{\"name\": <one allowed name>, \"confidence\": <0..1>, \"notes\": <sentence>}}."
    msgs = [{"role": "system", "content": LINE_SYSTEM}, {"role": "user", "content": text}]
    try:
        r = client.structured(msgs, PhaseLabel, name="line_state_label", task="name_line_state", max_tokens=200)
    except LLMError as e:
        return PhaseNaming("LINE", -1, None, 0.0, ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
    v = r.value
    name = v.name if v.name in LINE_VOCAB else None
    return PhaseNaming("LINE", -1, name, v.confidence if name else 0.0, v.notes, ok=name is not None,
                       latency_s=r.latency_s, cached=r.cached, attempts=r.attempts,
                       error=None if name else f"name outside vocabulary: {v.name!r}")


# ===================================================================================== G3: new-machine triage
TRIAGE_CLASSES = ("new_moving_machine", "parked_object", "camera_bump", "no_visible_change")


class TriageVerdict(BaseModel):
    category: Literal[TRIAGE_CLASSES]
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""

    @field_validator("category", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> Any:
        return v.strip().lower().replace("-", "_").replace(" ", "_") if isinstance(v, str) else v

    @field_validator("notes", mode="before")
    @classmethod
    def _trim(cls, v: Any) -> Any:
        return v[:300] if isinstance(v, str) else v


TRIAGE_SYSTEM = f"""You help an observer that maps where machines move in an automotive body shop. Its change detector found unexplained, persistent change in floor space that was static before. You see a REFERENCE frame from before the change and then frames AFTER it, all from the same fixed ceiling camera, oldest first. Decide what caused the change:
- new_moving_machine: a machine that was not there before and whose parts move between the after-frames (a newly installed robot / arm).
- parked_object: a vehicle, cart, pallet or other object standing still where the floor was empty; nothing else changed.
- camera_bump: the camera itself moved; the whole scene (floor markings, zones, posts, machines) is shifted or rotated compared with the reference, although nothing new appears.
- no_visible_change: the after-frames show nothing new and no camera shift compared with the reference (a transient such as a reflection, flicker or a passing shadow triggered the detector).
{BOUNDARY}
Your answer does NOT free or block any space: the flagged area stays unknown (treated as occupied) until the observer has learned it. It only tells the operator what to check.
Answer with one category, a confidence 0..1, and one short factual sentence in notes."""


# cell states used by the observer map; UNKNOWN is treated as occupied
FREE, OCCUPIED, UNKNOWN = "FREE", "OCCUPIED", "UNKNOWN"


@dataclass
class TriageResult:
    ok: bool
    verdict: TriageVerdict | None
    cells_state: str = UNKNOWN                 # always UNKNOWN: Gemma never frees space
    warnings: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    cached: bool = False
    attempts: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.model_dump() if self.verdict else None
        return d


ADVICE = {"new_moving_machine": "possible new machine: keep the area UNKNOWN until the observer has learned its cycle; "
                                "notify the plant engineer to register it",
          "parked_object": "possible parked object: area stays UNKNOWN; ask the area supervisor to confirm or remove it",
          "camera_bump": "possible camera bump: check / re-calibrate this camera; its area stays UNKNOWN until verified",
          "no_visible_change": "nothing visible: possibly a transient (reflection / flicker); area stays UNKNOWN until "
                               "the change detector clears it"}


def triage_change(reference: Any, after: Sequence[Any], *, camera_id: str | None = None,
                  client: GemmaClient | None = None, after_dt_s: float | None = None) -> TriageResult:
    """Classify a change-detection blob from a reference frame and 1..N after-frames. Never frees space."""
    client = client or default_client()
    imgs = [reference, *after]
    text = (f"Camera {camera_id}. " if camera_id else "") + \
           f"Image 1 is the REFERENCE (before). Images 2..{len(imgs)} are AFTER the change" + \
           (f", {after_dt_s:g} s apart" if after_dt_s else "") + \
           f".\nReturn JSON: {{\"category\": one of {list(TRIAGE_CLASSES)}, \"confidence\": <0..1>, \"notes\": <sentence>}}."
    msgs = [{"role": "system", "content": TRIAGE_SYSTEM}, user_message(text, imgs)]
    try:
        r = client.structured(msgs, TriageVerdict, name="triage_verdict", task="triage_change", max_tokens=200)
    except LLMError as e:
        return TriageResult(False, None, UNKNOWN, ["gemma_unavailable: area stays UNKNOWN"],
                            error=f"{type(e).__name__}: {str(e)[:200]}")
    v = r.value
    res = TriageResult(True, v, UNKNOWN, [f"{ADVICE[v.category]} (model confidence {v.confidence:.2f})"],
                       latency_s=r.latency_s, cached=r.cached, attempts=r.attempts)
    assert res.cells_state == UNKNOWN, "Gemma never frees space"
    return res


def apply_triage(cell_states: Mapping[Any, str], result: TriageResult | None) -> dict[Any, str]:
    """Cell states after a triage: unchanged. UNKNOWN cells stay UNKNOWN until the observer itself has learned the
    cycle (observer/change.py `machine_converged`), whatever Gemma said; no cell becomes FREE because of Gemma.
    The triage only contributes warnings (``result.warnings``)."""
    out = dict(cell_states)
    for c, s in cell_states.items():
        assert out[c] == s and not (s != FREE and out[c] == FREE), "Gemma never frees space"
    if result is not None:
        assert result.cells_state == UNKNOWN, "Gemma never frees space"
    return out
