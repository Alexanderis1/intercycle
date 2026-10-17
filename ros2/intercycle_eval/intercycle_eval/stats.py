"""Paired statistics over seeds (DIRECTION 5.6) + the ROBOT_RESULTS tables.

    .venv/Scripts/python.exe ros2/intercycle_eval/intercycle_eval/stats.py out/robot/<batch>   (after metrics.py)

  continuous metrics  Wilcoxon signed-rank on the per-seed paired differences (scipy, zero_method "wilcox", exact when
                      small), mean difference with a paired bootstrap 95 % CI (10 000 resamples over seeds)
  binary metrics      exact McNemar test (binomial on the discordant pairs)
  PRIMARY endpoints   line_time_lost_s, inspections_ok, any_collision; learned vs none and learned vs static; exact
                      sign-flip permutation test, Holm-corrected over the 6 tests (declared before the held-out batch)
  rates               Clopper-Pearson 95 % interval; with 0 events in N the upper bound is about 3/N ("rule of three")
Only valid episodes enter a comparison; a seed missing in either condition drops out of that pair (reported).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

CONT = ["n_inspected", "inspections_ok", "inspection_success_rate", "capture_pass_rate", "time_per_target_s",
        "line_time_lost_s", "cell_time_lost_s", "machine_time_lost_s", "trips", "hold_events", "windows_hit",
        "windows_missed", "collisions", "collisions_robot_moving", "min_dist_person_m", "min_ttc_person_s", "min_ttc_vehicle_s",
        "personal_space_events", "personal_space_s", "stops_navigating", "stalled_s", "path_length_m", "spl",
        "irregularity_rad_per_m", "rtf"]
BIN = ["safe_success", "any_collision", "any_trip"]
PAIRS = [("learned", "none"), ("learned", "static"), ("oracle", "learned")]
# Pre-registered before the held-out batch (AUDIT Tier 2): primary endpoints and primary comparisons. Each is tested with
# an exact two-sided sign-flip permutation test on the per-seed paired differences (all 2^n sign assignments), and the
# family of PRIMARY x PRIMARY_PAIRS p-values is Holm-corrected. Everything else is secondary / descriptive.
PRIMARY = ["line_time_lost_s", "inspections_ok", "any_collision"]
PRIMARY_PAIRS = [("learned", "none"), ("learned", "static")]


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
    raw = {}
    for a, b in PRIMARY_PAIRS:
        if a not in bc or b not in bc:
            continue
        seeds = sorted(set(bc[a]) & set(bc[b]))
        for k in PRIMARY:
            da = np.array([_num(bc[a][s].get(k)) for s in seeds])
            db = np.array([_num(bc[b][s].get(k)) for s in seeds])
            ok = np.isfinite(da) & np.isfinite(db)
            raw[f"{a}_vs_{b}:{k}"] = signflip_p(da[ok] - db[ok])
            res.setdefault("primary", {})[f"{a}_vs_{b}:{k}"] = {
                "n": int(ok.sum()), "mean_diff": float((da[ok] - db[ok]).mean()) if ok.any() else None,
                "ci95": bootstrap_ci(da[ok] - db[ok]) if ok.any() else None, "signflip_p": raw[f"{a}_vs_{b}:{k}"]}
    for k, p in holm(raw).items():
        res["primary"][k]["holm_p"] = p
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
    lines = ["| Primary comparison : endpoint | n | mean diff [boot 95 %] | sign-flip p (exact) | Holm p |",
             "|---|---|---|---|---|"]
    for k, v in (st.get("primary") or {}).items():
        ci = v["ci95"]
        lines.append(f"| {k} | {v['n']} | {_f(v['mean_diff'])} [{_f(ci[0]) if ci else '-'}, {_f(ci[1]) if ci else '-'}] "
                     f"| {_f(v['signflip_p'], 3)} | {_f(v.get('holm_p'), 3)} |")
    return "\n".join(lines)


def main() -> None:
    d = Path(sys.argv[1])
    rows = json.loads((d / "metrics.json").read_text())
    st = summarize(rows)
    (d / "stats.json").write_text(json.dumps(st, indent=1, default=str))
    md = ("## Primary endpoints (pre-registered; Holm-corrected)\n\n" + primary_md(st) + "\n\n## Per condition\n\n"
          + table_md(st) + "\n\n## Paired comparisons (secondary)\n\n" + pairs_md(st))
    (d / "tables.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
