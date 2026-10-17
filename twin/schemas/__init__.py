"""Twin contracts frozen at G0 (PLAN_TWIN §2). One convention everywhere: pydantic v2 models, with their JSON Schema
export committed in `twin/schemas/json/` (regenerate with `python -m twin.schemas`; the contract test fails when the
export and the models differ).

| file                                   | model                                   | owner |
|----------------------------------------|-----------------------------------------|-------|
| $INTERCYCLE_DATA/inventory.parquet     | tables.InventoryRow                     | S3a   |
| $INTERCYCLE_DATA/tracks/<clip>.parquet | tables.TrackRow                         | S3a   |
| $INTERCYCLE_DATA/machines/<press>/<clip>.parquet | tables.MachineRow             | S3a   |
| twin/site.yaml                         | site.Site                               | S2    |
| twin/tasks.yaml                        | tasks.Tasks                             | S2    |
| twin/safety.yaml                       | safety.Safety                           | S5a   |
| twin/dynamics.json                     | dynamics.Dynamics                       | S3a   |
| day event script (json)                | dayscript.DayScript                     | S3a   |
| runs/<id>/realised_state.jsonl         | runs.RealisedHeader + runs.RealisedTick | S5a   |
| runs/<id>/log.parquet                  | tables.LogRow                           | S5a   |
| runs/<id>/summary.json                 | runs.Summary                            | S5a   |
| primary-metric json                    | runs.PrimaryMetric                      | S1b   |

Observation v2 is owned by lane S4 (`intercycle/observer/mapping.py`) and is not defined here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from twin.schemas.dayscript import DayScript
from twin.schemas.dynamics import Dynamics
from twin.schemas.runs import PrimaryMetric, RealisedHeader, RealisedTick, Summary, parse_realised
from twin.schemas.safety import Safety
from twin.schemas.site import Site
from twin.schemas.tables import InventoryRow, LogRow, MachineRow, TrackRow, validate_table
from twin.schemas.tasks import Tasks

JSON_DIR = Path(__file__).resolve().parent / "json"

MODELS: dict[str, type[BaseModel]] = {
    "site": Site, "tasks": Tasks, "safety": Safety, "dynamics": Dynamics, "day_script": DayScript,
    "realised_state_header": RealisedHeader, "realised_state_tick": RealisedTick, "summary": Summary,
    "primary_metric": PrimaryMetric, "inventory_row": InventoryRow, "tracks_row": TrackRow,
    "machines_row": MachineRow, "log_row": LogRow,
}

__all__ = ["MODELS", "DayScript", "Dynamics", "InventoryRow", "LogRow", "MachineRow", "PrimaryMetric",
           "RealisedHeader", "RealisedTick", "Safety", "Site", "Summary", "Tasks", "TrackRow", "export_json",
           "json_schema", "load", "parse_realised", "validate_table"]


def json_schema(name: str) -> dict[str, Any]:
    return MODELS[name].model_json_schema(by_alias=True)


def export_json(out_dir: Path = JSON_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in MODELS:
        p = out_dir / f"{name}.schema.json"
        p.write_text(json.dumps(json_schema(name), indent=2) + "\n", encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def load(model: type[BaseModel], path: str | Path) -> Any:
    """Read a .yaml / .json contract file and validate it."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if p.suffix in (".yaml", ".yml") else json.loads(text)
    return model.model_validate(data)
