# prereg/

Pre-registration for the press-shop twin (docs/PLAN_TWIN.md §2, §5). Owner: lane S1a. Stub only: nothing here is
registered yet.

Files that will live here:

| File | When | Content |
|---|---|---|
| `split.csv` | before Pass 1 ends (§3) | clip_id, group_id, split, sha256 (no video) |
| `PREREG_TWIN.md` | PREREG v1, before the first dev-world run | primary formula, NI margin δ = 30 s/h, gate order and tests, r0, unsafe-free threshold, press inspection-window definition, exclusion and re-run rules, S7b's α and δ, the frozen baseline parameter space |
| `AMENDMENTS.md` | after PREREG v1 | each amendment with a timestamp, the dev run that prompted it and its evidence; implementation details only; none after FREEZE |
| `FREEZE` | FREEZE (08:30) | the commit sha of the frozen code |
| `TEST_SEED_COMMIT` | after FREEZE | sha256(salt ‖ seeds), written by the custodian |
