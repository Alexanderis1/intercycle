"""Source-clip provenance (PLAN_TWIN §5 "CAL uses only train frames. compile.py asserts that every source id in
site.yaml and dynamics.json is a train id").

Every value derived from clips names them: a site/tasks value whose source is cal_v0, cal_v1, traced_v0, traced_v1
or measured, a dynamics parameter whose source is fitted or fitted_censored, an OBSERVED guard's evidence and a
resampled day-script segment all carry non-empty clip ids (the schemas refuse them otherwise). `source_clips`
collects those ids per contract file, and `assert_train_only` checks them against the ACTIVE split
(`prereg/ACTIVE_SPLIT`, else `prereg/split.csv`): an id that is dev, test or not in the split fails. compile.py calls
`assert_train_only(source_clips(site=..., dynamics=..., safety=..., tasks=...))` before it builds anything.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from twin.schemas.dayscript import DayScript
from twin.schemas.dynamics import Dynamics
from twin.schemas.prereg import SplitRow, load_active_split
from twin.schemas.safety import Safety
from twin.schemas.site import Site
from twin.schemas.tasks import Tasks


class ProvenanceError(ValueError):
    """A source clip id is not a train id of the active split."""


def source_clips(*, site: Site | None = None, dynamics: Dynamics | None = None, safety: Safety | None = None,
                 tasks: Tasks | None = None, day_scripts: Iterable[DayScript] = ()) -> dict[str, list[str]]:
    """{contract file: sorted clip ids it was derived from}."""
    out: dict[str, set[str]] = {}
    if site is not None:
        out["site"] = set(site.cal_sources) | set(site.tagged_clips())
    if dynamics is not None:
        out["dynamics"] = set(dynamics.train_clips())
    if safety is not None:
        out["safety"] = {c for p in safety.presses for c in p.guard.evidence_clips}
    if tasks is not None:
        out["tasks"] = {c for cp in tasks.checkpoints for c in cp.clips}
    for ds in day_scripts:
        out.setdefault("generated_days", set()).update(s.clip_id for s in ds.segments if s.clip_id is not None)
    return {k: sorted(v) for k, v in out.items()}


def assert_train_only(refs: Mapping[str, Iterable[str]], split: Mapping[str, SplitRow] | None = None) -> int:
    """Raise ProvenanceError unless every id is a TRAIN id of the active split. Returns the number of ids checked."""
    rows = load_active_split() if split is None else split
    bad: dict[str, list[str]] = {}
    n = 0
    for where, ids in refs.items():
        for cid in ids:
            n += 1
            r = rows.get(cid)
            if r is None or r.split != "train":
                bad.setdefault(where, []).append(f"{cid} ({'not in split' if r is None else r.split})")
    if bad:
        raise ProvenanceError(f"source clips that are not train ids of the active split: {bad}")
    return n
