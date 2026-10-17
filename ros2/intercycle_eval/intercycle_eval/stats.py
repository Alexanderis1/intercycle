"""Paired statistics over seeds (DIRECTION 5.6) + the ROBOT_RESULTS tables.

    .venv/Scripts/python.exe ros2/intercycle_eval/intercycle_eval/stats.py out/robot/<batch>   (after metrics.py)

  continuous metrics  Wilcoxon signed-rank on the per-seed paired differences (scipy, zero_method "wilcox", exact when
                      small), mean difference with a paired bootstrap 95 % CI (10 000 resamples over seeds)
  binary metrics      exact McNemar test (binomial on the discordant pairs)
  PRE-REGISTERED      primary: line_time_lost_s, learned vs none, exact sign-flip permutation (SciPy, 2^10 flips);
                      safety: any_collision, Clopper-Pearson + exact Poisson per hour; secondary: Holm-corrected family
  rates               Clopper-Pearson 95 % interval; with 0 events in N the upper bound is about 3/N ("rule of three")
Only valid episodes enter a comparison; a seed missing in either condition drops out of that pair (reported).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

CONT = ["n_inspected", "inspections_ok", "inspections_ok_identify", "inspection_success_rate", "capture_pass_rate", "time_per_target_s",
        "line_time_lost_s", "cell_time_lost_s", "machine_time_lost_s", "trips", "hold_events", "windows_hit",
        "windows_missed", "collisions", "collisions_robot_moving", "min_dist_person_m", "min_ttc_person_s", "min_ttc_vehicle_s",
        "personal_space_events", "personal_space_s", "stops_navigating", "stalled_s", "path_length_m", "spl",
        "irregularity_rad_per_m", "rtf"]
BIN = ["safe_success", "any_collision", "any_trip"]
PAIRS = [("learned", "none"), ("learned", "static"), ("oracle", "learned")]
# PRE-REGISTERED (docs/ROBOT_RESULTS.md section 0, committed before the held-out batch; PRODUCTION_BRIEF A3):
#   primary endpoint  line_time_lost_s, learned vs none (no map): exact paired sign-flip permutation test
#                     (SciPy permutation_test, permutation_type="samples", all 2^10 = 1024 flips), two-sided, alpha 0.05
#   safety endpoint   any_collision per condition: Clopper-Pearson 95 % on episodes, exact Poisson 95 % bound per hour
#   secondary         every other endpoint x pair below: the same exact test, Holm-corrected as one family
PRIMARY = ("learned", "none", "line_time_lost_s")
SECONDARY_PAIRS = [("learned", "none"), ("learned", "static"), ("oracle", "learned")]
SECONDARY = ["inspections_ok", "inspections_ok_identify", "line_time_lost_s", "machine_time_lost_s", "trips", "windows_hit", "any_collision",
             "collisions_robot_moving", "min_dist_person_m", "stalled_s", "spl", "time_per_target_s"]
EPISODE_H = 600.0 / 3600.0


def perm_p(da, db) -> float | None:
    """Exact paired sign-flip permutation p (SciPy permutation_test, permutation_type='samples', all 2^n flips)."""
    from scipy.stats import permutation_test

    da, db = np.asarray(da, float), np.asarray(db, float)
    ok = np.isfinite(da) & np.isfinite(db)
    da, db = da[ok], db[ok]
    if len(da) < 2 or np.all(da == db):
        return None
    r = permutation_test((da, db), lambda x, y, axis: np.mean(x - y, axis=axis), permutation_type="samples",
                         vectorized=True, n_resamples=np.inf, alternative="two-sided")
    return float(r.pvalue)


def poisson_upper_per_h(k: int, hours: float, conf: float = 0.95) -> float:
    """Exact two-sided Poisson upper bound on an event rate (k events in `hours`): chi2(1 - a/2, 2k + 2) / 2 / hours."""
    from scipy.stats import chi2

    return float(chi2.ppf(1 - (1 - conf) / 2, 2 * k + 2) / 2 / hours) if hours > 0 else math.nan


def signflip_p(d) -> float | None:
    """Exact two-sided sign-flip permutation p-value for mean(d) = 0 (paired differences; n <= 20 enumerated)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0 or np.all(d == 0):
        return None
    if n > 20:
        rng = np.random.default_rng(0)
        signs = rng.choice((-1.0, 1.0), size=(200000, n))
    else:
        signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
    obs = abs(d.mean())
    means = np.abs((signs * d[None, :]).mean(axis=1))
    return float((means >= obs - 1e-12).mean())


def holm(ps: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (None stays None)."""
    items = sorted(((p, k) for k, p in ps.items() if p is not None))
    m = len(items)
    out, run = {k: None for k in ps}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    if n == 0:
        return (math.nan, math.nan)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(d: np.ndarray) -> float | None:
    from scipy.stats import wilcoxon

    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.all(d == 0):
        return None
    try:
        return float(wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return None


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    from scipy.stats import binomtest

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = None if n01 + n10 == 0 else float(binomtest(n10, n01 + n10, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p": p}


def by_cond(rows: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for r in rows:
        if r.get("valid"):
            out.setdefault(r["condition"], {})[int(r["seed"])] = r
    return out


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def summarize(rows: list[dict]) -> dict:
    bc = by_cond(rows)
    res = {"n_rows": len(rows), "n_valid": sum(1 for r in rows if r.get("valid")),
           "invalid": [{"tag": r["tag"], "reasons": r.get("invalid_reasons"), "error": r.get("error")} for r in rows
                       if not r.get("valid")],
           "timeouts": [r["tag"] for r in rows if r.get("timeout")],
           "per_condition": {}, "pairs": {}}
    for c, d in bc.items():
        pc = {"n": len(d)}
        for k in CONT:
            x = np.array([_num(r.get(k)) for r in d.values()])
            x = x[np.isfinite(x)]
            pc[k] = {"mean": float(x.mean()) if len(x) else None, "median": float(np.median(x)) if len(x) else None,
                     "ci95": bootstrap_ci(x) if len(x) else None, "n": int(len(x))}
        for k in BIN:
            kk = sum(1 for r in d.values() if r.get(k))
            pc[k] = {"k": kk, "n": len(d), "rate": kk / len(d) if d else None, "cp95": clopper_pearson(kk, len(d))}
        for cat in ("person", "vehicle", "machine", "static"):
            kk = sum(1 for r in d.values() if (r.get(f"collision_{cat}") or 0) > 0)
            pc[f"collision_ep_{cat}"] = {"k": kk, "n": len(d), "cp95": clopper_pearson(kk, len(d))}
        res["per_condition"][c] = pc
    for a, b in PAIRS:
        if a not in bc or b not in bc:
            continue
        seeds = sorted(set(bc[a]) & set(bc[b]))
        pr = {"seeds": seeds, "dropped": sorted(set(bc[a]) ^ set(bc[b]))}
        for k in CONT:
            da = np.array([_num(bc[a][s].get(k)) for s in seeds])
            db = np.array([_num(bc[b][s].get(k)) for s in seeds])
            ok = np.isfinite(da) & np.isfinite(db)
            d = da[ok] - db[ok]
            pr[k] = {"n": int(ok.sum()), "mean_diff": float(d.mean()) if len(d) else None,
                     "ci95": bootstrap_ci(d) if len(d) else None, "wilcoxon_p": wilcoxon_p(d)}
        for k in BIN:
            pr[k] = mcnemar_exact(np.array([bool(bc[a][s].get(k)) for s in seeds]),
                                  np.array([bool(bc[b][s].get(k)) for s in seeds]))
        res["pairs"][f"{a}_vs_{b}"] = pr
    # pre-registered primary
    a, b, k = PRIMARY
    if a in bc and b in bc:
        seeds = sorted(set(bc[a]) & set(bc[b]))
        da = np.array([_num(bc[a][q].get(k)) for q in seeds])
        db = np.array([_num(bc[b][q].get(k)) for q in seeds])
        res["primary"] = {"comparison": f"{a}_vs_{b}", "endpoint": k, "seeds": seeds, "diffs": (da - db).tolist(),
                          "mean_diff": float(np.nanmean(da - db)) if len(seeds) else None,
                          "median_diff": float(np.nanmedian(da - db)) if len(seeds) else None,
                          "perm_p": perm_p(da, db), "signflip_p_check": signflip_p(da - db)}
    # safety endpoint
    res["safety"] = {}
    for c, d in bc.items():
        kk = sum(1 for r in d.values() if r.get("any_collision"))
        km = sum(1 for r in d.values() if (r.get("collisions_robot_moving") or 0) > 0)
        ev = int(sum(r.get("collisions") or 0 for r in d.values()))
        hrs = len(d) * EPISODE_H
        res["safety"][c] = {"episodes": len(d), "with_collision": kk, "cp95": clopper_pearson(kk, len(d)),
                            "hours": round(hrs, 3), "events": ev, "events_per_h_upper95": poisson_upper_per_h(ev, hrs),
                            "zero_event_rate_upper95_per_h": poisson_upper_per_h(0, hrs),
                            "episodes_robot_moving_collision": km}
    # secondary family, Holm-corrected
    raw = {}
    res["secondary"] = {}
    for a, b in SECONDARY_PAIRS:
        if a not in bc or b not in bc:
            continue
        seeds = sorted(set(bc[a]) & set(bc[b]))
        for k in SECONDARY:
            if (a, b, k) == PRIMARY:
                continue
            da = np.array([_num(bc[a][q].get(k)) for q in seeds])
            db = np.array([_num(bc[b][q].get(k)) for q in seeds])
            key = f"{a}_vs_{b}:{k}"
            raw[key] = perm_p(da, db)
            ok = np.isfinite(da) & np.isfinite(db)
            res["secondary"][key] = {"n": int(ok.sum()), "diffs": (da - db).tolist(),
                                     "mean_diff": float((da[ok] - db[ok]).mean()) if ok.any() else None,
                                     "perm_p": raw[key]}
    for key, p in holm(raw).items():
        res["secondary"][key]["holm_p"] = p
    return res


def _f(v, nd=2):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{nd}f}"


def table_md(st: dict) -> str:
    conds = [c for c in ("none", "static", "learned", "oracle") if c in st["per_condition"]]
    lines = ["| Metric | " + " | ".join(f"{c} (n={st['per_condition'][c]['n']})" for c in conds) + " |",
             "|---|" + "---|" * len(conds)]
    for k in BIN:
        cells = []
        for c in conds:
            v = st["per_condition"][c][k]
            cells.append(f"{v['k']}/{v['n']} [{_f(v['cp95'][0])}, {_f(v['cp95'][1])}]")
        lines.append(f"| {k} (episodes, CP 95 %) | " + " | ".join(cells) + " |")
    for cat in ("person", "vehicle", "machine", "static"):
        cells = []
        for c in conds:
            v = st["per_condition"][c][f"collision_ep_{cat}"]
            cells.append(f"{v['k']}/{v['n']} (<= {_f(v['cp95'][1])})")
        lines.append(f"| episodes with a {cat} collision (CP upper 95 %) | " + " | ".join(cells) + " |")
    for k in CONT:
        cells = []
        for c in conds:
            v = st["per_condition"][c][k]
            cells.append("-" if v["mean"] is None else f"{_f(v['mean'])} [{_f(v['ci95'][0])}, {_f(v['ci95'][1])}]")
        lines.append(f"| {k} (mean [boot 95 %]) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pairs_md(st: dict, keys=None) -> str:
    keys = keys or ["inspections_ok", "line_time_lost_s", "machine_time_lost_s", "trips", "windows_hit", "collisions",
                    "collisions_robot_moving", "min_dist_person_m", "stalled_s", "spl", "time_per_target_s"]
    lines = ["| Pair | Metric | n | mean diff [boot 95 %] | Wilcoxon p |", "|---|---|---|---|---|"]
    for name, pr in st["pairs"].items():
        for k in keys:
            v = pr[k]
            ci = v["ci95"]
            lines.append(f"| {name} | {k} | {v['n']} | {_f(v['mean_diff'])} "
                         f"[{_f(ci[0]) if ci else '-'}, {_f(ci[1]) if ci else '-'}] | {_f(v['wilcoxon_p'], 3)} |")
        for k in BIN:
            v = pr[k]
            lines.append(f"| {name} | {k} (McNemar: {v['a_only']} vs {v['b_only']} discordant) | {len(pr['seeds'])} | - "
                         f"| {_f(v['p'], 3)} |")
    return "\n".join(lines)


def primary_md(st: dict) -> str:
    pr = st.get("primary")
    out = []
    if pr:
        out.append(f"**Primary (pre-registered): {pr['endpoint']}, {pr['comparison']}**, n = {len(pr['seeds'])} paired seeds: "
                   f"mean difference {_f(pr['mean_diff'])} s, median {_f(pr['median_diff'])} s, exact sign-flip "
                   f"permutation p = {_f(pr['perm_p'], 4)}.")
        out.append("")
        out.append("| seed | " + " | ".join(str(q) for q in pr["seeds"]) + " |")
        out.append("|---|" + "---|" * len(pr["seeds"]))
        out.append("| learned - none (s) | " + " | ".join(_f(v, 1) for v in pr["diffs"]) + " |")
        out.append("")
    out.append("**Safety endpoint: any collision** (Clopper-Pearson 95 % on episodes; exact Poisson 95 % upper bound per "
               "hour of episode time)")
    out.append("")
    out.append("| condition | episodes with a collision | CP 95 % | collision events | events/h upper 95 % | "
               "episodes with a collision while the robot moved |")
    out.append("|---|---|---|---|---|---|")
    for c in ("none", "static", "learned", "oracle"):
        v = (st.get("safety") or {}).get(c)
        if v:
            out.append(f"| {c} | {v['with_collision']}/{v['episodes']} | [{_f(v['cp95'][0])}, {_f(v['cp95'][1])}] | "
                       f"{v['events']} | {_f(v['events_per_h_upper95'], 1)} | {v['episodes_robot_moving_collision']} |")
    out.append("")
    out.append("**Secondary** (exact sign-flip permutation, Holm-corrected over the family)")
    out.append("")
    out.append("| comparison : endpoint | n | mean diff | perm p | Holm p |")
    out.append("|---|---|---|---|---|")
    for k, v in (st.get("secondary") or {}).items():
        out.append(f"| {k} | {v['n']} | {_f(v['mean_diff'])} | {_f(v['perm_p'], 3)} | {_f(v.get('holm_p'), 3)} |")
    return "\n".join(out)


def plot_paired(rows: list[dict], path: Path) -> None:
    """All paired per-seed differences (no bootstrap at n = 10): line time lost and inspections passed, vs none."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bc = by_cond(rows)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, k, lab in ((axs[0], "line_time_lost_s", "line time lost (s): condition - none"),
                       (axs[1], "inspections_ok", "inspections passed: condition - none")):
        for j, c in enumerate(("static", "learned", "oracle")):
            if c not in bc or "none" not in bc:
                continue
            seeds = sorted(set(bc[c]) & set(bc["none"]))
            d = [(_num(bc[c][q].get(k)) - _num(bc["none"][q].get(k))) for q in seeds]
            xs = j + (np.arange(len(d)) - (len(d) - 1) / 2) * 0.05
            ax.scatter(xs, d, s=18)
            if d:
                ax.hlines(np.nanmedian(d), j - 0.3, j + 0.3, colors="k", linewidth=1)
        ax.axhline(0, color="0.6", linewidth=0.8)
        ax.set_xticks(range(3), ["static", "learned", "timing oracle"])
        ax.set_title(lab, fontsize=9)
    fig.suptitle("SYNTHETIC: paired per-seed differences (one dot per seed; bar = median)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    d = Path(sys.argv[1])
    rows = json.loads((d / "metrics.json").read_text())
    st = summarize(rows)
    (d / "stats.json").write_text(json.dumps(st, indent=1, default=str))
    try:
        plot_paired(rows, d / "paired_differences.png")
    except Exception as e:  # noqa: BLE001
        print("plot failed:", e)
    md = ("## Pre-registered endpoints\n\n" + primary_md(st) + "\n\n## Per condition\n\n"
          + table_md(st) + "\n\n## Paired comparisons (secondary)\n\n" + pairs_md(st))
    (d / "tables.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
