# Body-shop event simulation at commit 1671f9e (earlier sim version)

`results_dev_0_19.json` is the unedited output of

    python -m intercycle.eval.run_seeds --seeds 0 19

run on 18 Oct 2026 in a clean checkout of commit `1671f9ea76d51a1cfb1d9e87336839604e177744`
("Lead: learned observer replaces ShimObserver in the event sim"), seeds 0-19, 6 h window (25200-46800 s).

This is the earlier version of the simulation: each robot's own camera was ideal (no occlusion, no missed or false
motion, no latency) and A0 / A did not share B's walkway rule. Commit 4660155 ("event-sim audit fixes") made the
robot camera noisy, added speed jitter per seed and gave every strategy the same walkway rule. The current numbers
are in `../RESULTS.md` and `../summary.json` (command `python -m intercycle.eval.bodyshop --seeds 0-19 --hours 6`).

Means from this file: B 39.9 insp/h, 0 trips, 0.5 holds, 1.5 s line time lost; A 50.4 insp/h, 5.0 trips, 568 s;
C 46.7 insp/h; A0 7.6 insp/h, 0.25 trips; B/C 0.854; line-time gain retained 0.997.
