"""Run and score the Gemma 4 observer tasks G1-G3 on the AMD Token Factory (record mode: every reply is cached).

    python scripts/gemma_keyframes.py g1|g2|g3       # render the SYNTHETIC keyframes + truth first
    python scripts/gemma_tasks.py run g1             # G1 commissioning: 10 machine cameras (+ --stills: workstream X renders)
    python scripts/gemma_tasks.py run g2             # G2 naming P's discovered phase bins (+ 4 line-state clusters)
    python scripts/gemma_tasks.py run g3             # G3 triage of change-detection crops
    python scripts/gemma_tasks.py score              # recompute out/gemma/results_g.json from the checkpoints
    INTERCYCLE_LLM_MODE=replay python scripts/gemma_tasks.py run g1   # offline, from out/llm_cache

Rate limit: <= 15 request starts per 60 s, <= 2 in flight, 429 -> Retry-After / back-off (RateLimitedTransport from
scripts/gemma_run_eval.py). /v1/usage is read before and after each batch (numeric fields only). The key comes from
.env and is never printed. Checkpoints out/gemma/rows_g{1,2,3}.jsonl: a re-run skips finished items.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.setdefault("INTERCYCLE_LLM_JSON_MODE", "object")     # response_format {"type": "json_object"} (§6)

from intercycle.llm import GemmaClient, LLMConfig, commission  # noqa: E402
from intercycle.llm import observer_tasks as OT  # noqa: E402
from intercycle.llm.tokenfactory import load_env, models, usage  # noqa: E402

OUT = ROOT / "out" / "gemma"
TYPE_OF_KIND = {"weld_station": "weld_robot", "respot": "weld_robot", "framing": "weld_robot",
                "unload_lifter": "lifter", "roller_hemming": "hemming_cell", "sealer": "sealer_robot"}
USAGE_KEYS = ("today.requests", "today.errors", "today.prompt_tokens", "today.completion_tokens", "today.cost",
              "daily_cost_remaining_usd", "rpm_limit")


# ----------------------------------------------------------------------------------------------- client
def jpeg(path, person_boxes=()) -> bytes:
    """Privacy-filtered image for the VLM: people boxes blacked out (+15 %), downscaled, metadata-free JPEG."""
    from intercycle.llm.privacy import filtered_jpeg
    return filtered_jpeg(path, person_boxes)


def make_client(max_requests: int):
    import gemma_run_eval as RE
    load_env()
    os.environ.setdefault("INTERCYCLE_LLM_MODE", "record")
    os.environ.setdefault("INTERCYCLE_LLM_TIMEOUT_S", "360")
    cfg = LLMConfig.from_env(mode=os.environ["INTERCYCLE_LLM_MODE"])
    if cfg.mode == "replay":
        return GemmaClient(cfg), None, cfg
    ids = models(cfg)
    if cfg.model not in ids:
        raise SystemExit(f"model {cfg.model} not in /v1/models ({len(ids)} ids); stop")
    tr = RE.RateLimitedTransport(cfg, max_per_min=15, max_conc=2, min_gap_s=4.2, max_requests=max_requests)
    return GemmaClient(cfg, transport=tr), tr, cfg


def usage_short(cfg, tag):
    u = usage(cfg, tag)
    return {k: u.get(k) for k in USAGE_KEYS} | {"ok": u.get("ok")}


def load_rows(path: Path) -> dict:
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return rows


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def batch(task: str, items: list[tuple[str, callable]], retry_errors: bool, max_requests: int, conc: int = 2) -> dict:
    """items: (id, fn(client) -> row dict). Runs the ones not yet checkpointed, sequentially (the transport paces)."""
    path = OUT / f"rows_{task}.jsonl"
    done = load_rows(path)
    todo = [(i, f) for i, f in items if i not in done or (retry_errors and not done[i].get("ok"))]
    print(f"[{task}] {len(items)} items, {len(todo)} to run", flush=True)
    if not todo:
        return {}
    client, tr, cfg = make_client(max_requests)
    print(f"model={cfg.model} mode={cfg.mode} provider={cfg.provider} json_mode={cfg.json_mode}", flush=True)
    info = {"task": task, "mode": cfg.mode}
    if tr is not None:
        info["usage_before"] = usage_short(cfg, f"{task}_before")
        print("usage before:", info["usage_before"], flush=True)
    t0 = time.time()
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock, cnt, stop = threading.Lock(), [0], [False]

    def work(item):
        i, fn = item
        if stop[0]:
            return
        if tr is not None:
            tr.begin()
        ts = time.time()
        try:
            row = fn(client)
        except RuntimeError as e:          # request budget exhausted -> stop, keep what we have
            print(f"stopped: {e}", flush=True)
            stop[0] = True
            return
        row.update(id=i, wall_s=round(time.time() - ts, 2), t_run=round(ts - t0, 1))
        if tr is not None:
            calls, wait = tr.end()
            row["n_http"] = len(calls)
            row["service_latency_s"] = round(sum(c[0] for c in calls), 3)
            row["prompt_tokens"] = int(sum((c[1].get("prompt_tokens") or 0) for c in calls))
            row["completion_tokens"] = int(sum((c[1].get("completion_tokens") or 0) for c in calls))
            row["queue_wait_s"] = round(wait, 2)
        with lock:
            append(path, row)
            cnt[0] += 1
            n = cnt[0]
        if n % 5 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)} in {time.time() - t0:.0f} s" +
                  (f" (attempts {tr.n_attempts}, 429s {tr.n_429}, errors {tr.n_errors})" if tr else ""), flush=True)

    with ThreadPoolExecutor(conc) as ex:
        list(ex.map(work, todo))
    info["wall_s"] = round(time.time() - t0, 1)
    if tr is not None:
        info.update(attempts=tr.n_attempts, http_429=tr.n_429, errors=tr.n_errors)
        info["usage_after"] = usage_short(cfg, f"{task}_after")
        print("usage after:", info["usage_after"], flush=True)
    with (OUT / "batches_g.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": round(time.time(), 1), **info}) + "\n")
    return info


# ==================================================================================================== G1
G1_VARIANTS = {"A": "headline: no floor-plan hint, no painted hazard hatch, generic camera id",
               "B": "secondary: same frames + floor-plan hint listing the machine ids/types in view (from the floor plan)"}


def g1_items(variants: str = "AB"):
    d = OUT / "g1"
    labs = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    items = []
    for v in variants:
        for name, lab in labs.items():
            hint = ({"machines_in_view": [{"id": z["machine_id"], "type": z["machine_type"]} for z in lab["zones"]]}
                    if v == "B" else None)

            def fn(client, name=name, lab=lab, hint=hint, v=v):
                try:
                    p = commission(jpeg(d / name, lab.get("person_boxes", [])), lab["camera_id"], hint, client=client)
                except Exception as e:  # noqa: BLE001
                    return {"file": name, "variant": v, "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                return {"file": name, "variant": v, "ok": True, "latency_s": p.latency_s, "cached": p.cached,
                        "zones": [{"label": z.label, "kind": z.kind, "machine_id": z.machine_id,
                                   "machine_type": z.machine_type, "polygon_px": z.polygon_px,
                                   "confidence": z.confidence} for z in p.zones],
                        "dropped": p.dropped, "warnings": p.warnings}
            items.append((f"{v}:{name}", fn))
    return items


def _mask(poly, w, h):
    im = Image.new("L", (w, h), 0)
    if len(poly) >= 3:
        ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=1)
    return np.asarray(im, bool)


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def score_g1(variant: str):
    d = OUT / "g1"
    labs = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    rows = load_rows(OUT / "rows_g1.jsonl")
    per, prim, allz, tp_pred, n_pred, mid_ok, type_ok, n_match = [], [], [], 0, 0, 0, 0, 0
    lat, kept, dropped = [], 0, 0
    for name, lab in labs.items():
        r = rows.get(f"{variant}:{name}")
        if not r or not r.get("ok"):
            continue
        w, h = lab["size"]
        gts = [(z, _mask(z["polygon_px"], w, h)) for z in lab["zones"]]
        preds = [(p, _mask(p["polygon_px"], w, h)) for p in r["zones"] if p["kind"] == "hazard_zone"]
        kept += len(r["zones"])
        dropped += len(r["dropped"])
        lat.append(r.get("service_latency_s") or r.get("latency_s"))
        M = np.array([[_iou(g, p) for _, p in preds] for _, g in gts]) if preds else np.zeros((len(gts), 0))
        best = M.max(1) if M.shape[1] else np.zeros(len(gts))
        # one-to-one greedy matching for precision / id accuracy
        used, matches = set(), []
        for gi, pj in sorted(((gi, pj) for gi in range(M.shape[0]) for pj in range(M.shape[1])), key=lambda x: -M[x]):
            if M[gi, pj] < 0.5 or gi in {m[0] for m in matches} or pj in used:
                continue
            used.add(pj)
            matches.append((gi, pj))
        for gi, pj in matches:
            g, p = gts[gi][0], preds[pj][0]
            n_match += 1
            mid_ok += int(p.get("machine_id") == g["machine_id"])
            type_ok += int(p.get("machine_type") == TYPE_OF_KIND.get(g["kind"]))
        n_pred += len(preds)
        tp_pred += len(matches)
        pi = [i for i, (z, _) in enumerate(gts) if z["primary"]]
        pr = float(best[pi[0]]) if pi else None
        prim.append(pr)
        allz += best.tolist()
        per.append({"camera": lab["camera_id"], "n_gt": len(gts), "n_pred_hazard": len(preds), "primary_iou": pr,
                    "best_iou": [round(float(b), 3) for b in best], "matched@0.5": len(matches)})
    if not per:
        return {}
    allz = np.array(allz)
    return {"n_cameras": len(per), "primary_zone_iou_mean": float(np.mean(prim)),
            "primary_zone_iou_median": float(np.median(prim)), "primary_zone_hit@0.5": float(np.mean(np.array(prim) >= 0.5)),
            "all_visible_zones": {"n": int(len(allz)), "best_iou_mean": float(allz.mean()),
                                  "recall@0.5": float((allz >= 0.5).mean())},
            "pred_hazard_zones": n_pred, "precision@0.5": tp_pred / n_pred if n_pred else None,
            "matched_machine_id_acc": mid_ok / n_match if n_match else None,
            "matched_machine_type_acc": type_ok / n_match if n_match else None,
            "zones_kept": kept, "zones_dropped": dropped,
            "latency_s": _lat(lat), "calls": len(per), "variant": G1_VARIANTS[variant], "per_camera": per}


def _lat(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    a = np.array(xs, float)
    return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2), "n": len(a)}


# ==================================================================================================== G2
def g2_items():
    d = OUT / "g2"
    labs = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    items = []
    for key, b in labs["bins"].items():
        def fn(client, b=b):
            r = OT.name_phase([jpeg(d / k["file"]) for k in b["keyframes"]], b["machine"], b["family"], b["machine_type"],
                              client=client)
            return {"machine": b["machine"], "bin": b["bin"], "ok": r.ok, "name": r.name, "confidence": r.confidence,
                    "notes": r.notes, "latency_s": r.latency_s, "cached": r.cached, "error": r.error}
        items.append((key, fn))
    tot = sum(c["n_ticks"] for c in labs["line"].values())
    for c, L in labs["line"].items():
        def fn(client, L=L):
            r = OT.name_line_state(L["centre"], median_run_s=L["median_run_s"], share=L["n_ticks"] / tot, client=client)
            return {"cluster": L["cluster"], "ok": r.ok, "name": r.name, "confidence": r.confidence, "notes": r.notes,
                    "latency_s": r.latency_s, "cached": r.cached, "error": r.error}
        items.append((f"line:{c}", fn))
    return items


def score_g2():
    d = OUT / "g2"
    labs = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    rows = load_rows(OUT / "rows_g2.jsonl")
    res_b, lat = [], []
    for key, b in labs["bins"].items():
        r = rows.get(key)
        if not r:
            continue
        truth = OT.TRUTH_TO_VOCAB[b["family"]].get(b["truth_majority"], b["truth_majority"])
        kf_truth = [OT.TRUTH_TO_VOCAB[b["family"]].get(k["truth_coarse"], k["truth_coarse"]) for k in b["keyframes"]]
        res_b.append({"id": key, "machine": b["machine"], "family": b["family"], "bin": b["bin"], "bin_name": b["bin_name"],
                      "n_ticks": b["n_ticks"], "truth": truth, "truth_purity": b["truth_purity"],
                      "keyframe_truth": kf_truth, "gemma": r.get("name"), "confidence": r.get("confidence"),
                      "ok": bool(r.get("ok")), "correct": r.get("name") == truth})
        lat.append(r.get("service_latency_s") or r.get("latency_s"))
    if not res_b:
        return {}
    ok = [x for x in res_b if x["ok"]]
    w = np.array([x["n_ticks"] for x in ok], float)
    c = np.array([x["correct"] for x in ok], float)
    fam = {}
    for f in ("weld", "lift", "cell"):
        g = [x for x in ok if x["family"] == f]
        if g:
            fam[f] = {"n_bins": len(g), "acc": float(np.mean([x["correct"] for x in g])),
                      "acc_time_weighted": float(np.average([x["correct"] for x in g], weights=[x["n_ticks"] for x in g]))}
    by_truth = {}
    for t in sorted({x["truth"] for x in ok}):
        g = [x for x in ok if x["truth"] == t]
        by_truth[t] = {"n": len(g), "acc": float(np.mean([x["correct"] for x in g])),
                       "gemma_said": {k: sum(1 for x in g if x["gemma"] == k) for k in sorted({str(x["gemma"]) for x in g})}}
    # baselines: uniform chance over the family's vocabulary; always answering the family's most frequent truth name
    chance = float(np.mean([1.0 / len(OT.PHASE_VOCAB[x["family"]]) for x in ok]))
    maj_name = {}
    for f_ in {x["family"] for x in ok}:
        g = [x["truth"] for x in ok if x["family"] == f_]
        maj_name[f_] = max(set(g), key=g.count)
    maj = np.array([x["truth"] == maj_name[x["family"]] for x in ok], float)
    baselines = {"chance": chance, "majority_per_family": float(maj.mean()),
                 "majority_per_family_time_weighted": float(np.average(maj, weights=w)), "majority_names": maj_name}
    # bins whose keyframes all share the bin's majority truth (clean bins) vs mixed
    clean = [x for x in ok if all(k == x["truth"] for k in x["keyframe_truth"])]
    line = []
    for cid, L in labs["line"].items():
        r = rows.get(f"line:{cid}")
        if r:
            line.append({"cluster": int(cid), "truth": L["truth_majority"], "truth_purity": L["truth_purity"],
                         "observer_rule_name": L["observer_rule_name"], "gemma": r.get("name"),
                         "confidence": r.get("confidence"), "correct": r.get("name") == L["truth_majority"],
                         "rule_correct": L["observer_rule_name"] == L["truth_majority"], "notes": r.get("notes")})
            lat.append(r.get("service_latency_s") or r.get("latency_s"))
    return {"n_bins": len(res_b), "n_ok": len(ok), "bin_acc": float(c.mean()) if len(c) else None,
            "bin_acc_time_weighted": float(np.average(c, weights=w)) if len(c) else None,
            "baselines": baselines,
            "clean_bins": {"n": len(clean), "acc": float(np.mean([x["correct"] for x in clean])) if clean else None},
            "by_family": fam, "by_truth": by_truth, "line_states": line,
            "line_acc": float(np.mean([x["correct"] for x in line if x["gemma"]])) if any(x["gemma"] for x in line) else None,
            "line_rule_acc": float(np.mean([x["rule_correct"] for x in line])) if line else None,
            "latency_s": _lat(lat), "calls": len(res_b) + len(line), "bins": res_b}


def write_phase_names():
    """out/gemma/phase_names.json: per machine, discovered bin -> Gemma's name (captions for the dashboard)."""
    d = OUT / "g2"
    labs = json.loads((d / "labels.json").read_text(encoding="utf-8"))
    rows = load_rows(OUT / "rows_g2.jsonl")
    out = {"label": "SYNTHETIC; names are Gemma captions (advisory, change nothing in the observer)",
           "model": os.environ.get("INTERCYCLE_LLM_MODEL"), "provider": "runs on AMD GPUs via Token Factory",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "source": "out/phases/model_final_train.pkl (workstream P)",
           "machines": {}, "line_clusters": {}}
    for key, b in labs["bins"].items():
        r = rows.get(key)
        if r and r.get("ok"):
            out["machines"].setdefault(b["machine"], {})[str(b["bin"])] = {
                "bin_name": b["bin_name"], "name": r["name"], "confidence": r["confidence"], "notes": r.get("notes", "")}
    for cid, L in labs["line"].items():
        r = rows.get(f"line:{cid}")
        if r and r.get("ok"):
            out["line_clusters"][cid] = {"observer_rule_name": L["observer_rule_name"], "name": r["name"],
                                         "confidence": r["confidence"]}
    (OUT / "phase_names.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ==================================================================================================== G3
def g3_items():
    d = OUT / "g3"
    labs = json.loads((d / "labels_events.json").read_text(encoding="utf-8"))
    labs |= json.loads((d / "labels.json").read_text(encoding="utf-8"))
    items = []
    for key, L in labs.items():
        def fn(client, L=L, key=key):
            f = [jpeg(d / x) for x in L["files"]]
            dt = L["t_after"][1] - L["t_after"][0] if len(L["t_after"]) > 1 else None
            r = OT.triage_change(f[0], f[1:], camera_id=f"CAM-{key.upper()}", client=client, after_dt_s=dt and round(dt, 1))
            st = OT.apply_triage({"blob": OT.UNKNOWN}, r)
            assert st["blob"] == OT.UNKNOWN
            v = r.verdict
            return {"truth": L["truth"], "ok": r.ok, "category": v.category if v else None,
                    "confidence": v.confidence if v else None, "notes": v.notes if v else "",
                    "cells_state": r.cells_state, "cells_after_apply": st["blob"], "warnings": r.warnings,
                    "latency_s": r.latency_s, "cached": r.cached, "error": r.error}
        items.append((key, fn))
    return items


def score_g3(prefix: str):
    rows = {k: r for k, r in load_rows(OUT / "rows_g3.jsonl").items() if k.startswith(prefix)}
    ok = [r for r in rows.values() if r.get("ok")]
    if not ok:
        return {}
    cls = OT.TRIAGE_CLASSES
    truths = [t for t in cls if any(r["truth"] == t for r in ok)]
    conf = {t: {p: sum(1 for r in ok if r["truth"] == t and r["category"] == p) for p in cls} for t in truths}
    rec = {t: conf[t][t] / sum(conf[t].values()) for t in truths}
    prec = {p: (conf[p][p] / sum(conf[t][p] for t in truths)) if p in truths and sum(conf[t][p] for t in truths) else None
            for p in cls}
    return {"n": len(rows), "n_ok": len(ok), "acc": float(np.mean([r["truth"] == r["category"] for r in ok])),
            "chance": 1 / len(cls), "confusion_truth_x_gemma": conf, "recall": rec, "precision": prec,
            "items": {k: {"truth": r["truth"], "gemma": r.get("category"), "confidence": r.get("confidence"),
                          "notes": r.get("notes")} for k, r in rows.items()} if prefix == "evt" else None,
            "cells_never_freed": all(r.get("cells_state") == OT.UNKNOWN and r.get("cells_after_apply") == OT.UNKNOWN
                                     for r in rows.values()),
            "latency_s": _lat([r.get("service_latency_s") or r.get("latency_s") for r in ok]), "calls": len(rows),
            "framing": "M's real change events (detector rect / camera)" if prefix == "evt" else
                       "ORACLE-FRAMED: crops centred on the true object / camera (synthetic variants)"}


# ================================================================================================= report
def score_all():
    load_env()
    res = {"model": os.environ.get("INTERCYCLE_LLM_MODEL"), "provider": "AMD Token Factory (AMD GPUs)",
           "g1": score_g1("A"), "g1_hinted": score_g1("B"), "g2": score_g2(), "g3": score_g3("evt"), "g3_oracle": score_g3("chg")}
    calls = [json.loads(x) for x in (ROOT / "out" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    g = [c for c in calls if c.get("task") in ("commission", "name_phase", "name_line_state", "triage_change")]
    live = [c for c in g if c.get("live")]
    res["calls_log"] = {"live_calls": len(live), "live_ok": sum(1 for c in live if c.get("ok")),
                        "replayed": sum(1 for c in g if not c.get("live")),
                        "prompt_tokens_live": sum(c.get("prompt_tokens") or 0 for c in live),
                        "completion_tokens_live": sum(c.get("completion_tokens") or 0 for c in live),
                        "by_task": {t: sum(1 for c in live if c["task"] == t) for t in sorted({c["task"] for c in live})}}
    if (OUT / "batches_g.jsonl").is_file():
        res["batches"] = [json.loads(x) for x in (OUT / "batches_g.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    (OUT / "results_g.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    if res["g2"]:
        write_phase_names()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("task", nargs="?", choices=["g1", "g2", "g3"])
    ap.add_argument("--variants", default="AB", help="G1 variants (A headline, B hinted)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--max-requests", type=int, default=900, help="cap on logged attempts in out/gemma/request_log.jsonl")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        items = {"g1": lambda: g1_items(a.variants), "g2": g2_items, "g3": g3_items}[a.task]()
        batch(a.task, items[:a.limit] if a.limit else items, a.retry_errors, a.max_requests, min(a.concurrency, 2))
    res = score_all()
    for k in ("g1", "g1_hinted", "g2", "g3", "g3_oracle"):
        r = {kk: v for kk, v in (res.get(k) or {}).items() if kk not in ("per_camera", "bins")}
        print(k, json.dumps(r, default=str)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
