"""InterCycle digital twin of a real press shop (docs/PLAN_TWIN.md, Option B).

Contracts every lane codes against live in `twin.schemas` (pydantic models; the JSON Schema export is committed in
`twin/schemas/json/`). v0 stubs frozen at G0: `site_v0.yaml`, `dynamics_v0.json`, `daygen_stub.py` and the
MEASURED_SHIM camera adapter `camera_v0.py`. Truth (realised state, plant state, day scripts, per-world dynamics
draws, world seeds, defect state) may only be opened by the modules listed in `twin.truth_guard`.

Only numpy, astropy, pydantic, nacl, yaml and the standard library may be imported at module level by the twin
modules the Isaac side imports (tests/test_isaac_deps.py).
"""
