"""Parquet contracts: one pydantic row model per table; column order and Arrow types live on the fields
(`x-arrow`, exported with the JSON Schema), so the row model is the single source of truth.

* `$INTERCYCLE_DATA/inventory.parquet` (S3a): InventoryRow. camera is taken from the class prefix
  (0,1,2,4,5,6 -> 9; 3,7 -> 14).
* `$INTERCYCLE_DATA/tracks/<clip>.parquet` (S3a): TrackRow. t from container PTS (s); (u, v) is the foot point in
  pixels; (x, y) its floor position (NaN when the framing has no fitted H), re-projected when H changes.
* `$INTERCYCLE_DATA/machines/<press>/<clip>.parquet` (S3a): MachineRow. Vertical flow on the ram ROI and stroke
  events; `missing` (flow NaN, no stroke) while a person box overlaps the ROI.
* `runs/<id>/log.parquet` (S5a): LogRow, the plant's event log.

pyarrow is imported lazily (it is not in the Isaac environment, tests/test_isaac_deps.py).
"""
from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from twin.schemas.common import Camera, Split, Strict

CAMERA_OF_CLASS = {0: 9, 1: 9, 2: 9, 4: 9, 5: 9, 6: 9, 3: 14, 7: 14}
ARROW_OK = {"string": ("string", "large_string", "string_view"), "int64": ("int64",), "float64": ("double",),
            "bool": ("bool",)}


def col(arrow: str, **kw: Any) -> Any:
    return Field(json_schema_extra={"x-arrow": arrow}, **kw)


def _nan_ok(v: float | None) -> bool:
    return v is None or math.isnan(v)


class InventoryRow(Strict):
    clip_id: str = col("string", min_length=1)
    camera: Camera = col("int64")
    framing_id: str = col("string", min_length=1)
    group_id: str = col("string", min_length=1)
    cls: int = col("int64", alias="class", ge=0, le=7)
    duration: float = col("float64", gt=0)
    sha256: str = col("string", pattern=r"^[0-9a-f]{64}$")
    split: Split = col("string")

    @model_validator(mode="after")
    def _camera(self):
        if CAMERA_OF_CLASS[self.cls] != self.camera:
            raise ValueError(f"{self.clip_id}: class {self.cls} implies camera {CAMERA_OF_CLASS[self.cls]}, not {self.camera}")
        return self


class TrackRow(Strict):
    split: Split = col("string")
    t: float = col("float64", ge=0)
    track_id: int = col("int64", ge=0)
    cls: Literal["person", "forklift", "vehicle"] = col("string", alias="class")
    u: float = col("float64")
    v: float = col("float64")
    x: float | None = col("float64")
    y: float | None = col("float64")
    conf: float = col("float64", ge=0, le=1)
    framing: str = col("string", min_length=1)


class MachineRow(Strict):
    split: Split = col("string")
    clip_id: str = col("string", min_length=1)
    press: str = col("string", min_length=1)
    framing: str = col("string", min_length=1)
    t: float = col("float64", ge=0)
    flow_v: float | None = col("float64")          # mean vertical flow on the ram ROI (px/s, + = down); NaN missing
    roi_energy: float | None = col("float64")      # motion energy on the ROI; NaN missing
    stroke: bool = col("bool")                     # a stroke event starts at this frame
    missing: bool = col("bool")                    # a person box overlaps the ROI

    @model_validator(mode="after")
    def _missing(self):
        if self.missing and (not _nan_ok(self.flow_v) or not _nan_ok(self.roi_energy) or self.stroke):
            raise ValueError("a missing frame has NaN flow_v / roi_energy and no stroke")
        return self


LOG_KINDS = ("mission_start", "mission_end", "checkpoint_attempt", "checkpoint_pass", "checkpoint_fail",
             "checkpoint_decline", "press_trip", "press_hold", "press_restart", "incident", "contact", "error")


class LogRow(Strict):
    run_id: str = col("string", min_length=1)
    t: float = col("float64", ge=0)                # sim seconds since the mission start
    kind: Literal["mission_start", "mission_end", "checkpoint_attempt", "checkpoint_pass", "checkpoint_fail",
                  "checkpoint_decline", "press_trip", "press_hold", "press_restart", "incident", "contact",
                  "error"] = col("string")
    entity: str = col("string")                    # robot / press / mover id the event is about
    checkpoint: str | None = col("string")
    zone: str | None = col("string")
    value: float | None = col("float64")           # e.g. lost seconds for press_trip / press_hold, gap (m) for contact
    detail: str | None = col("string")             # JSON text


TABLES: dict[str, type[BaseModel]] = {"inventory": InventoryRow, "tracks": TrackRow, "machines": MachineRow,
                                      "log": LogRow}


def columns(model: type[BaseModel]) -> list[tuple[str, str]]:
    """[(column name, arrow type)] in contract order."""
    out = []
    for name, f in model.model_fields.items():
        extra = f.json_schema_extra
        assert isinstance(extra, dict)
        out.append((f.alias or name, str(extra["x-arrow"])))
    return out


def validate_table(table: Any, model: type[BaseModel], *, allow_test: bool = True, max_rows: int | None = None) -> int:
    """Check a pyarrow Table (or a parquet path) against a row model: exact column names and order, Arrow types,
    then every row (or the first `max_rows`). Raises ValueError; returns the number of rows checked."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not isinstance(table, pa.Table):
        table = pq.read_table(table)
    want = columns(model)
    got = [(f.name, str(f.type)) for f in table.schema]
    if [n for n, _ in got] != [n for n, _ in want]:
        raise ValueError(f"columns {[n for n, _ in got]} != contract {[n for n, _ in want]}")
    for (name, typ), (_, want_t) in zip(got, want):
        if typ not in ARROW_OK[want_t]:
            raise ValueError(f"column {name}: arrow type {typ}, contract {want_t}")
    rows = table.slice(0, max_rows).to_pylist() if max_rows is not None else table.to_pylist()
    for i, r in enumerate(rows):
        row = model.model_validate(r)
        if not allow_test and getattr(row, "split", None) == "test":
            raise ValueError(f"row {i}: test split rows are not allowed here")
    return len(rows)
