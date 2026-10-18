# LEARN20 amendment: the learned arm's observer learns 20 min of the same day

Status: in force on branch `lane/learn20b` (from `lane/integration`), before the robot FREEZE (Sun 10:15 CEST).
It amends PLAN_TWIN §5 "Design" (the observer phase of the CPU-double batch, `scripts/run_double.py`).

## User decision

> "Try days 1-8, else 20 min" (USER DECISION, Sun ~08:35 CEST)

## Why the fallback applies (measured)

- The days 1-8 observer phase (PLAN_TWIN §5: the robot-free stream of days 1-8, then the scored day up to each
  mission start, through ONE observer per world) runs at **90.3 s wall per 1,200 s simulated** (measured,
  robot-redundancy timeline).
- Days 1-8 = 8 x 86,400 s = 691,200 s simulated = 576 windows of 1,200 s, so **~14.4 h wall per world**
  (all worlds run in parallel, one container each, so the phase alone is ~14.4 h).
- That cannot finish before 12:30, so the "days 1-8" branch of the decision fails and the **20 min fallback** is in
  force.

## What changes

The learned condition's observer now learns exactly as `scripts/slice.py` does (`LEARN_S = 1200`):

- For each learned mission (world, day, zone) with start `t0`, a **fresh** `twin/observer_live.py` observer ingests the
  **robot-free plant stream of the SAME world day** whose captures lie in **[t0 - 1200 s, t0)**
  (`run_double.learn20_phase`, window from `run_double.learn_window`). The stream is the plant's own per-tick path:
  twin/engine.py -> the plant's camera model (built fresh, as a plant started at the window start builds it) -> the
  plant's ObsAdapter, with no robot mover.
- Then, as before: the observer state is pickled (the snapshot) and the learned arm's broadcaster
  (`twin/broadcast_live.py --snapshot-t0`) starts from it and tails the mission's live `obs_v2.jsonl`.
- The delivery rule is unchanged: the snapshot holds only the observations out by `t0` (the one-latency gap at the
  mission start, the same for all three conditions, as before).
- Only the mission days (9 and 10) are generated; days 1-8 are no longer generated for the batch.
- Edge case (disclosed): `twin_batch.SLOT_PAD_S = 600 s`, so a mission can start less than 1,200 s into its day. Then
  the window is **clipped at the day start** (it never reaches into the day before), and the run record says so
  (`observer_phase[].windows[...].clipped = true`, `learn_s` < 1200). Probability per slot is about
  600 / 84,480 = 0.7 %, so about 0.45 of the 64 learned missions are expected to be affected.

Unchanged: the static and reactive conditions; the 192 sealed run ids (`w<W>_d<D>_<zone>_<condition>`); the slots
`t0`; every guard (the sealed refusal outside `sealed_run.py`, the 16-seed check, the Modal-only sealed backend, the
stub-input refusal, the snapshot-predates-t0 check, the RUNLOG records written by `sealed_run.py`).

## Where it is in the code

- Default for the sealed and the dev paths: `run_double.DEFAULT_OBSERVER_MODE = "learn20"`; `run_sealed` always
  passes `observer_mode="learn20"`; the Modal job carries `observer_mode` and the container dispatches on it
  (`run_observer_phase`).
- The days 1-8 code (`observer_phase`) is **kept**, only behind the explicit, non-default, dev-only CLI flag
  `--observer-days18` (mode `days18`). `check_observer_mode` refuses `days18` in a sealed run
  (`INTERCYCLE_SEALED_RUN=1`).
- Batch record: `double_batch.json` gains `observer_mode` and `learn_window_s`.
- Tests: `tests/test_learn20.py`.

## Consequences

- **No learning across days.** The day 10 observer knows nothing of day 9, and nothing of days 1-8. Each learned
  mission starts from 20 min of its own day only; the two zones of one day learn independently.
- **Weaker learning.** 20 min (2,400 ticks at 2 Hz) instead of ~8 days: rarely seen press cycles, slow drifts and
  day-to-day regularities cannot be learned. The learned arm's result is therefore a test of what the observer
  learns in 20 min plus the live stream, not of what it would learn in 8 days. Any H1-H3 claim must say so.
- This is the same learning budget as the slice (`scripts/slice.py`), so the batch and the slice now test the same
  observer setting.

## Wall time of the 192-cell batch in this mode (estimate from the measured rates)

- Observer phase: 4 learned windows per world (2 days x 2 zones), run one after another in the world's container,
  all 16 worlds in parallel: 4 x 90.3 s = **~6 min**.
- Missions: 4 reactive rounds (48 missions each, wave size 50), ~6.6-7 min per round: **~26-28 min**.
- Total: **~32-34 min**, plus container start, pulls and classification overhead (versus ~14.4 h + 28 min with
  days 1-8).
