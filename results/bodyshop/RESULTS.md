# Body-shop event simulation results

Seeds 0-19 (n = 20 per strategy), 6 h shift window from 07:00, code commit `80fb1a8c12603ff46460eba74c6aa559e4886d5e`.
Command: `python -m intercycle.eval.bodyshop --seeds 0-19 --hours 6` (286 s wall, 6 processes).

Means per run with 95 % confidence half-widths (1.96 x standard error over seeds).

| strategy | insp/h | trips/run | holds/run | line time lost s/run | insp/h vs C |
|---|---|---|---|---|---|
| A0 shutdown-only | 7.7 ± 0.1 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 | 0.164 |
| A reactive | 48.1 ± 0.6 | 10.95 ± 2.25 | 44.55 ± 5.22 | 1135.3 ± 216.4 | 1.028 |
| B InterCycle (learned observer) | 41.5 ± 0.4 | 3.35 ± 1.39 | 0.90 ± 0.35 | 304.6 ± 124.7 | 0.886 |
| C PLC oracle | 46.8 ± 0.2 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0 ± 0.0 | 1.000 |

- B/C throughput (InterCycle inspections per production hour / PLC oracle): **0.886**
- Line-time gain retained by B, (A - B) / (A - C): 0.732
- insp/h = machine inspections per production hour; trips = robot crossed into an active zone (protective stop of the line); holds = inspections where the line waited for the robot; line time lost = seconds of production stopped by the robot.
- Simulation only (event-based model of a body-shop cell), not measured on a real line.
