"""Evaluate Gemma 4 verify_state against labelled frames (accuracy, person-in-zone precision/recall,
latency p50/p95, throughput). Works offline in replay mode.

Layout of --frames DIR:
    DIR/frames/*.png          (or the PNGs directly in DIR)
    DIR/labels.json           {"file.png": {"machine": "R1", "moving": bool, "person_in_zone": bool}, ...}
Optional label keys: "images": [other crops, oldest first, last = the key itself is added if absent],
"machine_type", "zone_hint". Keys starting with "_" are ignored.

    python scripts/gemma_make_fixture.py                         # 6 synthetic frames in out/gemma_fixture
    python scripts/gemma_eval_verify.py --mock                   # offline smoke test with the pixel-heuristic
                                                                 #   stand-in (NOT Gemma; own cache and log)
    python scripts/gemma_eval_verify.py --mock --mode replay     # same, from the recorded cache
    INTERCYCLE_LLM_MODE=replay python scripts/gemma_eval_verify.py --frames out/sim_frames
    python scripts/gemma_eval_verify.py --mode record --concurrency 4 --frames out/sim_frames   # live Gemma

Latency is the server latency of the original live call: in replay mode it is the recorded value and the
report says so. Throughput is requests per second of wall time when calls are live, plus a
sequential-equivalent figure (n / sum of latencies) that is always available.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intercycle.llm import GemmaClient, LLMConfig, LLMError, verify_state  # noqa: E402
from intercycle.llm.fixtures import MOCK_MODEL, FixtureHeuristicTransport, make_fixture  # noqa: E402

DEFAULT_FRAMES = ROOT / "out" / "gemma_fixture"


def load_labels(frames_dir: Path, labels_path: Path | None = None) -> dict[str, dict]:
    p = labels_path or frames_dir / "labels.json"
    labels = json.loads(Path(p).read_text(encoding="utf-8"))
    return {k: v for k, v in labels.items() if not k.startswith("_")}


def resolve_image(frames_dir: Path, name: str) -> Path:
    for cand in (frames_dir / "frames" / name, frames_dir / name):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"frame {name!r} not found under {frames_dir} (looked in frames/ and the folder itself)")


def _row(client: GemmaClient, frames_dir: Path, name: str, lab: dict) -> dict:
    row = {"file": name, "machine": lab["machine"], "truth_moving": bool(lab["moving"]),
           "truth_person": bool(lab["person_in_zone"]), "ok": False}
    names = list(lab.get("images") or [])
    if name not in names:
        names.append(name)
    imgs = [resolve_image(frames_dir, n) for n in names]
    exp = {k: lab[k] for k in ("machine_type", "zone_hint") if k in lab}
    try:
        r = verify_state(imgs, lab["machine"], exp, client=client, raise_on_error=True)
    except LLMError as e:
        row["error"] = f"{type(e).__name__}: {str(e)[:240]}"
        return row
    v = r.verdict
    row.update(ok=True, pred_moving=v.moving, pred_person=v.person_in_zone, confidence=v.confidence,
               latency_s=r.latency_s, cached=r.cached, attempts=r.attempts, notes=v.notes)
    return row


def _pr(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f1}


def compute_metrics(rows: list[dict], wall_s: float | None = None,
                    usage: dict | None = None) -> dict:
    ok = [r for r in rows if r["ok"]]
    n, k = len(rows), len(ok)
    m: dict = {"n": n, "n_ok": k, "errors": n - k}
    if not ok:
        return m
    acc = lambda f: sum(1 for r in ok if f(r)) / k  # noqa: E731
    m["accuracy_moving"] = acc(lambda r: r["pred_moving"] == r["truth_moving"])
    m["accuracy_person_in_zone"] = acc(lambda r: r["pred_person"] == r["truth_person"])
    m["accuracy_both"] = acc(lambda r: r["pred_moving"] == r["truth_moving"] and r["pred_person"] == r["truth_person"])
    tp = sum(1 for r in ok if r["pred_person"] and r["truth_person"])
    fp = sum(1 for r in ok if r["pred_person"] and not r["truth_person"])
    fn = sum(1 for r in ok if not r["pred_person"] and r["truth_person"])
    tn = k - tp - fp - fn
    m["person_in_zone"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **_pr(tp, fp, fn)}
    mtp = sum(1 for r in ok if r["pred_moving"] and r["truth_moving"])
    mfp = sum(1 for r in ok if r["pred_moving"] and not r["truth_moving"])
    mfn = sum(1 for r in ok if not r["pred_moving"] and r["truth_moving"])
    m["moving"] = {"tp": mtp, "fp": mfp, "fn": mfn, "tn": k - mtp - mfp - mfn, **_pr(mtp, mfp, mfn)}
    lat = np.array([r["latency_s"] for r in ok], dtype=float)
    live = not all(r["cached"] for r in ok)
    m["latency_source"] = "live" if live else "recorded (replay)"
    m["latency_s"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                      "mean": float(lat.mean())}
    m["throughput_rps_sequential"] = k / float(lat.sum()) if lat.sum() > 0 else None
    m["throughput_rps_wall"] = (k / wall_s) if (live and wall_s) else None
    m["mean_confidence"] = float(np.mean([r["confidence"] for r in ok]))
    m["mean_attempts"] = float(np.mean([r["attempts"] for r in ok]))
    return m


def evaluate(frames_dir: Path, client: GemmaClient, labels_path: Path | None = None,
             concurrency: int = 1, limit: int | None = None) -> dict:
    labels = load_labels(frames_dir, labels_path)
    items = list(labels.items())[:limit]
    t0 = time.perf_counter()
    if concurrency > 1:
        with ThreadPoolExecutor(concurrency) as ex:
            rows = list(ex.map(lambda kv: _row(client, frames_dir, *kv), items))
    else:
        rows = [_row(client, frames_dir, *kv) for kv in items]
    wall = time.perf_counter() - t0
    cfg = client.cfg
    return {"config": {"model": cfg.model, "mode": cfg.mode, "provider": cfg.provider,
                       "is_mock": cfg.model == MOCK_MODEL, "concurrency": concurrency,
                       "frames_dir": str(frames_dir)},
            "metrics": compute_metrics(rows, wall), "rows": rows}


def format_report(res: dict) -> str:
    c, m = res["config"], res["metrics"]
    f = lambda x, p=3: "n/a" if x is None else f"{x:.{p}f}"  # noqa: E731
    lines = [f"verify_state eval: model={c['model']} mode={c['mode']} provider={c['provider']}"
             + ("   [MOCK STAND-IN, NOT GEMMA]" if c["is_mock"] else ""),
             f"frames: {m['n']}  ok: {m['n_ok']}  errors: {m['errors']}"]
    if m["n_ok"]:
        p, mv, lat = m["person_in_zone"], m["moving"], m["latency_s"]
        lines += [
            f"accuracy  moving={f(m['accuracy_moving'])}  person_in_zone={f(m['accuracy_person_in_zone'])}  both={f(m['accuracy_both'])}",
            f"person_in_zone  precision={f(p['precision'])}  recall={f(p['recall'])}  f1={f(p['f1'])}  "
            f"(tp={p['tp']} fp={p['fp']} fn={p['fn']} tn={p['tn']})",
            f"moving          precision={f(mv['precision'])}  recall={f(mv['recall'])}  f1={f(mv['f1'])}  "
            f"(tp={mv['tp']} fp={mv['fp']} fn={mv['fn']} tn={mv['tn']})",
            f"latency [{m['latency_source']}]  p50={f(lat['p50'])}s  p95={f(lat['p95'])}s  mean={f(lat['mean'])}s",
            f"throughput  sequential-equivalent={f(m['throughput_rps_sequential'], 2)} req/s"
            f"  wall={f(m['throughput_rps_wall'], 2)} req/s",
            f"mean confidence={f(m['mean_confidence'], 2)}  mean attempts={f(m['mean_attempts'], 2)}"]
    errs = [r for r in res["rows"] if not r["ok"]]
    for r in errs[:3]:
        lines.append(f"  error {r['file']}: {r['error']}")
    return "\n".join(lines)


def build_client(args: argparse.Namespace) -> GemmaClient:
    if args.mock:
        cfg = LLMConfig.from_env(model=MOCK_MODEL, mode=args.mode or "record", provider="mock",
                                 cache_dir=args.cache_dir or ROOT / "out" / "llm_cache_mock",
                                 log_path=ROOT / "out" / "llm_calls_mock.jsonl", cost_free=True)
        return GemmaClient(cfg, transport=FixtureHeuristicTransport())
    return GemmaClient(LLMConfig.from_env(model=args.model, mode=args.mode, cache_dir=args.cache_dir))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--mode", choices=("live", "record", "replay"))
    ap.add_argument("--model")
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--mock", action="store_true", help="offline pixel-heuristic stand-in (not Gemma)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "gemma_eval_verify.json")
    a = ap.parse_args(argv)

    if a.frames == DEFAULT_FRAMES and not (a.frames / "labels.json").is_file():
        make_fixture(a.frames)
        print(f"generated synthetic fixture in {a.frames}")
    client = build_client(a)
    res = evaluate(a.frames, client, a.labels, a.concurrency, a.limit)
    print(format_report(res))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0 if res["metrics"]["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
