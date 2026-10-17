"""Rate-limited live evaluation of Gemma 4 `verify_state` on simulator frames (single frame vs 0.5 s pair).

    python scripts/gemma_frames.py                       # renders out/gemma/frames (labels.json, labels_pair.json)
    python scripts/gemma_run_eval.py                     # live: 160 frames x {single, pair} = 320 requests, ~25 min
    python scripts/gemma_run_eval.py --limit 6           # smoke test
    python scripts/gemma_run_eval.py --analyze-only      # recompute results / figures / examples from the checkpoints

Shared quota (20 req/min per account, 8 concurrent) -> this script keeps to <= 15 requests in any 60 s window,
<= 4 in flight, backs off on 429 using Retry-After, and never exceeds --max-requests attempts in total (counted in
out/gemma/request_log.jsonl).  Every response is cached by the client (INTERCYCLE_LLM_MODE=record, out/llm_cache) so
the demo can replay offline, and every live call is appended to out/llm_calls.jsonl.  The key is read from .env and is
never printed or logged.

Outputs: out/gemma/results.json, confusion_matrices.png, accuracy_by_state.png, examples.md (+ examples/),
rows_single.jsonl / rows_pair.jsonl (checkpoints; a re-run skips finished samples).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from intercycle.llm import GemmaClient, LLMConfig, LLMError, verify_state  # noqa: E402
from intercycle.llm.client import TransportError  # noqa: E402

OUT = ROOT / "out" / "gemma"
FRAMES = OUT / "frames"
REQ_LOG = OUT / "request_log.jsonl"


# ------------------------------------------------------------------------------------------------ env + limiter
def load_env(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines into os.environ (existing variables win). Values are never printed."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def budget_used() -> int:
    if not REQ_LOG.is_file():
        return 0
    return sum(1 for _ in REQ_LOG.open("r", encoding="utf-8"))


class RateLimitedTransport:
    """OpenAI transport with a sliding-window rate limit, a concurrency cap and 429 back-off.

    * at most `max_per_min` request attempts in any 60 s window (every attempt counts, retries included)
    * at most `max_conc` in flight
    * `min_gap_s` between request starts (set to 0 for a deliberate burst)
    * 429: wait Retry-After (or exponential back-off), pause ALL threads, retry
    * `tls.calls` collects (service latency, usage) of the calling thread's successful calls, so queueing and
      back-off time never pollutes the latency numbers.
    """

    def __init__(self, cfg: LLMConfig, max_per_min: int = 15, max_conc: int = 4, min_gap_s: float = 0.0,
                 max_requests: int = 450):
        import openai
        self._openai = openai
        self._client = openai.OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout_s, max_retries=0)
        self.max_per_min, self.min_gap_s, self.max_requests = max_per_min, min_gap_s, max_requests
        self._sem = threading.BoundedSemaphore(max_conc)
        self._lock = threading.Lock()
        self._stamps: collections.deque[float] = collections.deque()
        self._last_start = 0.0
        self._cool_until = 0.0
        self.tls = threading.local()
        self.n_attempts = 0
        self.n_429 = 0
        self.n_errors = 0

    def _slot(self) -> float:
        """Block until a request may start; returns the seconds spent waiting."""
        t_wait = time.time()
        while True:
            with self._lock:
                now = time.time()
                while self._stamps and now - self._stamps[0] >= 61.0:
                    self._stamps.popleft()
                wait = max(self._cool_until - now, self._last_start + self.min_gap_s - now, 0.0)
                if wait <= 0 and len(self._stamps) >= self.max_per_min:
                    wait = self._stamps[0] + 61.0 - now
                if wait <= 0:
                    if budget_used() >= self.max_requests:
                        raise RuntimeError(f"request budget of {self.max_requests} exhausted (see {REQ_LOG})")
                    self._stamps.append(now)
                    self._last_start = now
                    self.n_attempts += 1
                    return now - t_wait
            time.sleep(min(max(wait, 0.05), 2.0))

    def _log(self, rec: dict) -> None:
        REQ_LOG.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, REQ_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": round(time.time(), 3), **rec}) + "\n")

    def begin(self) -> None:
        self.tls.calls, self.tls.wait = [], 0.0

    def end(self) -> tuple[list, float]:
        return getattr(self.tls, "calls", []), getattr(self.tls, "wait", 0.0)

    def __call__(self, payload: dict) -> dict:
        oa = self._openai
        attempt = 0
        while True:
            with self._sem:
                wait = self._slot()
                self.tls.wait = getattr(self.tls, "wait", 0.0) + wait
                t0 = time.perf_counter()
                try:
                    resp = self._client.chat.completions.create(**payload)
                    lat = time.perf_counter() - t0
                    d = resp.model_dump()
                    self._log({"status": 200, "latency_s": round(lat, 3)})
                    if not hasattr(self.tls, "calls"):
                        self.tls.calls = []
                    self.tls.calls.append((lat, d.get("usage") or {}))
                    return d
                except oa.RateLimitError as e:
                    self.n_429 += 1
                    ra = None
                    try:
                        ra = float(e.response.headers.get("retry-after"))
                    except (TypeError, ValueError, AttributeError):
                        pass
                    pause = min(max(ra if ra is not None else 0.0, 4.0 * 2 ** attempt), 90.0)
                    with self._lock:
                        self._cool_until = max(self._cool_until, time.time() + pause)
                    self._log({"status": 429, "retry_after": ra, "pause_s": pause})
                    attempt += 1
                    if attempt > 6:
                        raise TransportError("HTTP 429: gave up after 6 back-offs", status=429) from e
                except oa.APIStatusError as e:
                    self.n_errors += 1
                    self._log({"status": e.status_code, "error": str(e.message)[:120]})
                    if e.status_code >= 500 and attempt < 2:
                        attempt += 1
                        with self._lock:
                            self._cool_until = max(self._cool_until, time.time() + 3.0 * attempt)
                        continue
                    raise TransportError(f"HTTP {e.status_code}: {e.message}", status=e.status_code,
                                         body=getattr(e, "body", None)) from e
                except oa.APIConnectionError as e:
                    self.n_errors += 1
                    self._log({"status": "connection", "error": str(e)[:120]})
                    if attempt < 2:
                        attempt += 1
                        with self._lock:
                            self._cool_until = max(self._cool_until, time.time() + 3.0 * attempt)
                        continue
                    raise TransportError(f"connection error: {e}") from e


def make_client(max_requests: int = 450, min_gap_s: float = 0.0) -> tuple[GemmaClient, RateLimitedTransport]:
    load_env()
    os.environ.setdefault("INTERCYCLE_LLM_MODE", "record")
    cfg = LLMConfig.from_env(mode=os.environ.get("INTERCYCLE_LLM_MODE", "record"))
    tr = RateLimitedTransport(cfg, max_requests=max_requests, min_gap_s=min_gap_s)
    return GemmaClient(cfg, transport=tr), tr


# --------------------------------------------------------------------------------------------------- metrics
def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)
    return {"precision": p, "recall": r, "f1": f1}


def _conf(rows, key_t, key_p):
    tp = sum(1 for r in rows if r[key_p] and r[key_t])
    fp = sum(1 for r in rows if r[key_p] and not r[key_t])
    fn = sum(1 for r in rows if not r[key_p] and r[key_t])
    tn = len(rows) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **_prf(tp, fp, fn)}


def _acc(rows, key_t, key_p):
    return (sum(1 for r in rows if r[key_p] == r[key_t]) / len(rows)) if rows else None


def mode_metrics(rows: list[dict], wall_s: float | None) -> dict:
    ok = [r for r in rows if r.get("ok")]
    m: dict = {"n": len(rows), "n_ok": len(ok), "errors": len(rows) - len(ok)}
    if not ok:
        return m
    m["accuracy_moving"] = _acc(ok, "truth_moving", "pred_moving")
    m["accuracy_person_in_zone"] = _acc(ok, "truth_person", "pred_person")
    m["accuracy_both"] = sum(1 for r in ok if r["pred_moving"] == r["truth_moving"] and r["pred_person"] == r["truth_person"]) / len(ok)
    m["person_in_zone"] = _conf(ok, "truth_person", "pred_person")
    m["moving"] = _conf(ok, "truth_moving", "pred_moving")
    lat = np.array([r["latency_s"] for r in ok])
    m["latency_s"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                      "mean": float(lat.mean()), "max": float(lat.max())}
    m["throughput_rps_single_stream"] = float(len(ok) / lat.sum())
    m["throughput_rps_wall_quota_limited"] = float(len(ok) / wall_s) if wall_s else None
    m["mean_confidence"] = float(np.mean([r["confidence"] for r in ok]))
    m["mean_calls_per_verdict"] = float(np.mean([r["n_calls"] for r in ok]))
    m["tokens"] = {"prompt_total": int(sum(r.get("prompt_tokens") or 0 for r in ok)),
                   "completion_total": int(sum(r.get("completion_tokens") or 0 for r in ok)),
                   "prompt_mean": float(np.mean([r.get("prompt_tokens") or 0 for r in ok])),
                   "completion_mean": float(np.mean([r.get("completion_tokens") or 0 for r in ok]))}
    # slices
    sl: dict = {}
    for cat in sorted({r["category"] for r in ok}):
        g = [r for r in ok if r["category"] == cat]
        sl[f"category={cat}"] = {"n": len(g), "accuracy_moving": _acc(g, "truth_moving", "pred_moving"),
                                 "accuracy_person_in_zone": _acc(g, "truth_person", "pred_person"),
                                 "person_recall": _conf(g, "truth_person", "pred_person")["recall"]}
    for name, f in (("moving_true,visible_motion", lambda r: r["truth_moving"] and r["visible_motion"]),
                    ("moving_true,near_still", lambda r: r["truth_moving"] and not r["visible_motion"]),
                    ("person_inside,hard(<1 m from border)", lambda r: r["truth_person"] and r["hard_margin"]),
                    ("person_inside,clear(>=1 m)", lambda r: r["truth_person"] and not r["hard_margin"]),
                    ("person_inside,machine_active", lambda r: r["truth_person"] and r["truth_moving"]),
                    ("person_inside,machine_not_active", lambda r: r["truth_person"] and not r["truth_moving"]),
                    ("no_person_in_view", lambda r: not r["any_person_in_view"]),
                    ("outside_person_in_view", lambda r: r["any_person_in_view"] and not r["truth_person"])):
        g = [r for r in ok if f(r)]
        if g:
            sl[name] = {"n": len(g), "accuracy_moving": _acc(g, "truth_moving", "pred_moving"),
                        "accuracy_person_in_zone": _acc(g, "truth_person", "pred_person"),
                        "pred_person_rate": float(np.mean([r["pred_person"] for r in g])),
                        "pred_moving_rate": float(np.mean([r["pred_moving"] for r in g]))}
    m["slices"] = sl
    return m


# ----------------------------------------------------------------------------------------------------- tasks
def load_tasks(frames: Path, limit: int | None, seed: int) -> list[dict]:
    single = json.loads((frames / "labels.json").read_text(encoding="utf-8"))
    pair = json.loads((frames / "labels_pair.json").read_text(encoding="utf-8"))
    pair_by_stem = {}
    for k, v in pair.items():
        pair_by_stem[v["images"][0]] = (k, v)
    tasks = []
    for k, v in single.items():
        pk, pv = pair_by_stem[k]
        tasks.append({"sample": k, "single": (k, v), "pair": (pk, pv)})
    rng = np.random.default_rng(seed)
    rng.shuffle(tasks)                                    # any prefix is a random subset
    return tasks[:limit] if limit else tasks


def label_row(mode: str, key: str, lab: dict) -> dict:
    ppl = lab.get("people", [])
    inside = [p for p in ppl if p["inside"]]
    return {"file": key, "mode": mode, "machine": lab["machine"], "category": lab["category"], "state": lab["state"],
            "truth_moving": bool(lab["moving"]), "truth_person": bool(lab["person_in_zone"]),
            "visible_motion": bool(lab.get("visible_motion")), "any_person_in_view": bool(ppl),
            "hard_margin": bool(inside and min(p["margin_m"] for p in inside) < 1.0),
            "person_source": ("synthetic" if any(p["source"] == "synthetic" for p in inside) else
                              ("sim" if inside else None)),
            "conveyor_moving": bool(lab.get("conveyor_moving"))}


def run_one(client, tr, frames: Path, mode: str, key: str, lab: dict, t_run0: float) -> dict:
    row = label_row(mode, key, lab)
    names = list(lab.get("images") or [])
    if mode == "single":
        names = [key]
    elif key not in names:
        names.append(key)
    imgs = [frames / n for n in names]
    exp = {"machine_type": lab.get("machine_type"), "zone_hint": lab.get("zone_hint")}
    tr.begin()
    t0 = time.time()
    try:
        r = verify_state(imgs, lab["machine"], exp, client=client, frame_dt_s=0.5 if len(imgs) > 1 else None,
                         raise_on_error=True)
    except (LLMError, RuntimeError) as e:
        calls, wait = tr.end()
        row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}", t_start=t0 - t_run0, t_end=time.time() - t_run0)
        if isinstance(e, RuntimeError) and not isinstance(e, LLMError):
            raise
        return row
    calls, wait = tr.end()
    v = r.verdict
    row.update(ok=True, pred_moving=bool(v.moving), pred_person=bool(v.person_in_zone), confidence=float(v.confidence),
               notes=v.notes, suppress_window=bool(r.suppress_window), latency_s=float(sum(c[0] for c in calls)),
               n_calls=len(calls) or 1, cached=bool(r.cached), queue_wait_s=round(wait, 3),
               prompt_tokens=int(sum((c[1].get("prompt_tokens") or 0) for c in calls)),
               completion_tokens=int(sum((c[1].get("completion_tokens") or 0) for c in calls)),
               t_start=t0 - t_run0, t_end=time.time() - t_run0)
    return row


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["file"]] = r
    return rows


def run_eval(frames: Path, limit, conc: int, max_requests: int, gap: float, burst: int, retry_errors: bool) -> dict:
    tasks = load_tasks(frames, limit, seed=11)
    paths = {m: OUT / f"rows_{m}.jsonl" for m in ("single", "pair")}
    done = {m: load_rows(p) for m, p in paths.items()}
    todo = []
    for t in tasks:
        for m in ("single", "pair"):
            k, lab = t[m]
            old = done[m].get(k)
            if old is None or (not old.get("ok") and retry_errors):
                todo.append((m, k, lab))
    # interleave so single and pair of one sample are close in time
    print(f"{len(tasks)} samples, {len(todo)} requests to run ({sum(len(d) for d in done.values())} already done); "
          f"budget used so far {budget_used()}/{max_requests}", flush=True)
    if not todo:
        return {"wall_s": None}
    client, tr = make_client(max_requests, min_gap_s=0.0)
    print(f"model={client.cfg.model} mode={client.cfg.mode} provider={client.cfg.provider}", flush=True)
    lock = threading.Lock()
    t_run0 = time.time()
    counter = {"n": 0}

    def work(item):
        m, k, lab = item
        row = run_one(client, tr, frames, m, k, lab, t_run0)
        with lock:
            with paths[m].open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            done[m][k] = row
            counter["n"] += 1
            n = counter["n"]
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t_run0
                print(f"  {n}/{len(todo)} done in {el:.0f} s (attempts {tr.n_attempts}, 429s {tr.n_429}, errors "
                      f"{tr.n_errors}); eta {el / n * (len(todo) - n):.0f} s", flush=True)
        return row

    with ThreadPoolExecutor(conc) as ex:
        b = todo[:burst]
        list(ex.map(work, b))                              # deliberate burst at concurrency `conc`
        tr.min_gap_s = gap                                 # then even pacing
        list(ex.map(work, todo[burst:]))
    return {"wall_s": time.time() - t_run0, "attempts": tr.n_attempts, "n_429": tr.n_429, "n_errors": tr.n_errors}


# ------------------------------------------------------------------------------------------------------ report
def analyze(frames: Path, run_info: dict | None, limit=None) -> dict:
    rows = {m: list(load_rows(OUT / f"rows_{m}.jsonl").values()) for m in ("single", "pair")}
    cfg = {"model": os.environ.get("INTERCYCLE_LLM_MODEL", "?"), "frames_dir": str(frames.relative_to(ROOT)),
           "provider": "amd-token-factory (AMD Instinct GPUs)", "mode": "record (live calls, cached)",
           "rate_limit": "<= 15 req/min, concurrency 4"}
    labs = json.loads((frames / "labels.json").read_text(encoding="utf-8"))
    cfg["n_samples"] = len(labs)
    cfg["exposure_s"] = next(iter(labs.values()))["exposure_s"]
    res = {"config": cfg}
    walls = {}
    for m in ("single", "pair"):
        rs = rows[m]
        if rs:
            t0 = min(r["t_start"] for r in rs if "t_start" in r)
            t1 = max(r["t_end"] for r in rs if "t_end" in r)
            walls[m] = t1 - t0
        res[m] = mode_metrics(rs, None)
        res[m]["wall_s_of_run"] = walls.get(m)
        res[m]["throughput_rps_wall_quota_limited"] = (len([r for r in rs if r.get("ok")]) / walls[m]) if walls.get(m) else None
    allr = [r for m in ("single", "pair") for r in rows[m] if r.get("ok")]
    burst = sorted([r for r in allr if r.get("queue_wait_s", 9) < 0.5], key=lambda r: r["t_start"])
    if len(burst) >= 4:
        b = burst[:8]
        span = max(r["t_end"] for r in b) - min(r["t_start"] for r in b)
        res["burst_concurrency4"] = {"n": len(b), "wall_s": span, "throughput_rps": len(b) / span,
                                     "latency_p50_s": float(np.percentile([r["latency_s"] for r in b], 50)),
                                     "note": "first requests of the run, issued back to back at concurrency 4 before pacing"}
    res["tokens_total"] = {"prompt": sum(r.get("prompt_tokens") or 0 for r in allr),
                           "completion": sum(r.get("completion_tokens") or 0 for r in allr)}
    # requests accounting
    req = [json.loads(line) for line in REQ_LOG.read_text(encoding="utf-8").splitlines() if line.strip()] if REQ_LOG.is_file() else []
    res["requests"] = {"attempts_logged": len(req), "http_429": sum(1 for r in req if r.get("status") == 429),
                       "other_errors": sum(1 for r in req if r.get("status") not in (200, 429))}
    # pair vs single agreement
    s = {r["file"].replace(".png", ""): r for r in rows["single"] if r.get("ok")}
    p = {r["file"].replace("_b.png", ""): r for r in rows["pair"] if r.get("ok")}
    common = sorted(set(s) & set(p))
    if common:
        res["agreement_single_vs_pair"] = {"n": len(common),
                                           "moving": float(np.mean([s[k]["pred_moving"] == p[k]["pred_moving"] for k in common])),
                                           "person": float(np.mean([s[k]["pred_person"] == p[k]["pred_person"] for k in common]))}
    res["rows_file"] = {"single": "out/gemma/rows_single.jsonl", "pair": "out/gemma/rows_pair.jsonl"}
    if run_info:
        res["run"] = run_info
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    make_figures(res)
    make_examples(frames, rows, labs)
    return res


# ----------------------------------------------------------------------------------------------------- figures
def make_figures(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, ink2, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    ramp = LinearSegmentedColormap.from_list("blue", ["#eaf1fb", "#9ec3f0", "#2a78d6", "#12437f"])
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.4), facecolor=surf)
    for i, (key, title, neg, pos) in enumerate((("person_in_zone", "Person in hazard zone", "no person", "person"),
                                                  ("moving", "Machine moving", "still", "moving"))):
        for j, mode in enumerate(("single", "pair")):
            ax = axes[i, j]
            ax.set_facecolor(surf)
            c = res.get(mode, {}).get(key)
            if not c:
                ax.axis("off")
                continue
            M = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], float)    # rows = truth (neg, pos), cols = Gemma
            rown = M / np.maximum(M.sum(1, keepdims=True), 1)
            ax.imshow(rown, cmap=ramp, vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    dark = rown[a, b] > 0.55
                    ax.text(b, a, f"{int(M[a, b])}\n{rown[a, b] * 100:.0f}%", ha="center", va="center", fontsize=12,
                            color="#ffffff" if dark else ink, fontweight="bold" if a == b else "normal")
            ax.set_xticks([0, 1], [f"Gemma: {neg}", f"Gemma: {pos}"], color=ink2, fontsize=9)
            ax.set_yticks([0, 1], [f"truth: {neg}", f"truth: {pos}"], color=ink2, fontsize=9, rotation=90, va="center")
            f1 = c.get("f1")
            ax.set_title(f"{title} - {mode}" + ("" if mode == "single" else " (2 frames, 0.5 s apart)"), fontsize=10, color=ink,
                         loc="left")
            ax.set_xlabel(f"precision {c['precision']:.2f}  recall {c['recall']:.2f}  F1 {f1:.2f}"
                          if c["precision"] is not None and c["recall"] is not None and f1 is not None else "", color=ink2, fontsize=9)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.tick_params(length=0)
    fig.suptitle(f"verify_state on simulator frames: Gemma-4-31B-IT on AMD  (n = {res['single'].get('n_ok', 0)} samples, "
                 f"counts and % of each truth row)", fontsize=11, color=ink, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "confusion_matrices.png", dpi=140, facecolor=surf)
    plt.close(fig)

    # accuracy of "moving" and of "person" by machine state, single vs pair
    cats = [k.split("=")[1] for k in res["single"].get("slices", {}) if k.startswith("category=")]
    if not cats:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), facecolor=surf, sharey=True)
    for ax, key, title in ((axes[0], "accuracy_moving", "Accuracy of 'moving' by machine state"),
                           (axes[1], "accuracy_person_in_zone", "Accuracy of 'person in zone' by machine state")):
        ax.set_facecolor(surf)
        x = np.arange(len(cats))
        for k, (mode, col) in enumerate((("single", "#2a78d6"), ("pair", "#eb6834"))):
            vals = [res[mode]["slices"].get(f"category={c}", {}).get(key) or 0 for c in cats]
            ns = [res[mode]["slices"].get(f"category={c}", {}).get("n", 0) for c in cats]
            bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.34, color=col, label=f"{mode} frame" if mode == "single" else "pair of frames")
            for bx, v in zip(x + (k - 0.5) * 0.38, vals):
                ax.text(bx, v + 0.015, f"{v * 100:.0f}", ha="center", fontsize=8, color=ink2)
        ax.set_xticks(x, [f"{c}\n(n={res['single']['slices'][f'category={c}']['n']})" for c in cats], fontsize=8, color=ink2)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10, color=ink, loc="left")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(axis="y", colors=ink2, length=0)
    axes[0].set_ylabel("accuracy", color=ink2)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_by_state.png", dpi=140, facecolor=surf)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ examples
def make_examples(frames: Path, rows: dict, labs: dict) -> None:
    from PIL import Image
    ex_dir = OUT / "examples"
    ex_dir.mkdir(exist_ok=True)
    s = {r["file"]: r for r in rows["single"] if r.get("ok")}
    p = {r["file"].replace("_b.png", ".png"): r for r in rows["pair"] if r.get("ok")}
    keys = sorted(set(s) & set(p))
    if not keys:
        return
    rng = np.random.default_rng(5)

    def pick(pred, used):
        c = [k for k in keys if k not in used and pred(k)]
        return c[int(rng.integers(len(c)))] if c else None

    used: set = set()
    plan = [("Person standing inside an idle / held / stopped zone, found by both",
             lambda k: s[k]["truth_person"] and not s[k]["truth_moving"] and s[k]["pred_person"] and p[k]["pred_person"]),
            ("Person inside a zone whose machine is ACTIVE (curtain-fault scenario)",
             lambda k: s[k]["truth_person"] and s[k]["truth_moving"]),
            ("A person missed (false negative) in single or pair mode",
             lambda k: s[k]["truth_person"] and (not s[k]["pred_person"] or not p[k]["pred_person"])),
            ("A false alarm (person reported, none inside the zone)",
             lambda k: not s[k]["truth_person"] and (s[k]["pred_person"] or p[k]["pred_person"])),
            ("Machine active with visible arm motion: single frame vs pair",
             lambda k: s[k]["truth_moving"] and s[k]["visible_motion"]),
            ("Machine stopped mid-motion (trip) or at home (hold): the single-frame ambiguity",
             lambda k: s[k]["state"] in ("trip", "hold") and not s[k]["truth_person"])]
    chosen = []
    for title, pred in plan:
        k = pick(pred, used)
        if k is None:
            k = pick(lambda k: True, used)
        if k is None:
            continue
        used.add(k)
        chosen.append((title, k))
    L = ["# Gemma 4 verify_state: qualitative sample", "",
         "Six frames picked by rule (not by looking at the results): the first line of each entry says what the rule was; "
         "when no frame satisfied a rule (for example no missed person) a random frame is shown instead and the title is "
         "then not literally true, so check the truth line. Images are SYNTHETIC renders from the InterCycle simulator. "
         "`Gemma notes` is the model's own one-sentence description; the model is shown neither the label nor the "
         "observer's belief.", ""]
    for i, (title, k) in enumerate(chosen, 1):
        lab = labs[k]
        a = Image.open(frames / k)
        b = Image.open(frames / k.replace(".png", "_b.png"))
        comp = Image.new("RGB", (a.width * 2 + 8, a.height), (20, 20, 20))
        comp.paste(a, (0, 0))
        comp.paste(b, (a.width + 8, 0))
        name = f"ex{i}_{k.replace('.png', '')}_pair.png"
        comp.save(ex_dir / name)
        ppl = ", ".join(f"{q['source']} person {'INSIDE' if q['inside'] else 'outside'} ({q['margin_m']:+.1f} m)"
                        for q in lab["people"]) or "no person in view"
        L += [f"## {i}. {title}", "",
              f"![{k}](examples/{name})", "",
              f"- **Truth**: machine `{lab['machine']}`, state `{lab['state']}` ({lab['category']}), moving = {lab['moving']}, "
              f"person_in_zone = {lab['person_in_zone']}; arm tool speed {lab['arm_speed_mps']:.2f} m/s; {ppl}.",
              ""]
        for mode, r in (("single frame (left image)", s[k]), ("pair, 0.5 s apart (both images)", p[k])):
            L.append(f"- **Gemma, {mode}**: moving = {r['pred_moving']}, person_in_zone = {r['pred_person']}, "
                     f"confidence {r['confidence']:.2f}, latency {r['latency_s']:.1f} s, {r.get('prompt_tokens')}+"
                     f"{r.get('completion_tokens')} tokens. Gemma notes: _{r.get('notes', '').strip()}_")
        L.append("")
    (OUT / "examples.md").write_text("\n".join(L), encoding="utf-8")


def format_summary(res: dict) -> str:
    f = lambda x, d=3: "n/a" if x is None else f"{x:.{d}f}"  # noqa: E731
    out = []
    for m in ("single", "pair"):
        r = res.get(m, {})
        if not r.get("n_ok"):
            continue
        p, mv, lat = r["person_in_zone"], r["moving"], r["latency_s"]
        out.append(f"[{m}] n_ok={r['n_ok']}/{r['n']}  acc moving={f(r['accuracy_moving'])} person={f(r['accuracy_person_in_zone'])} "
                   f"both={f(r['accuracy_both'])}\n   person P={f(p['precision'])} R={f(p['recall'])} F1={f(p['f1'])} "
                   f"(tp{p['tp']} fp{p['fp']} fn{p['fn']} tn{p['tn']})  moving P={f(mv['precision'])} R={f(mv['recall'])} "
                   f"F1={f(mv['f1'])} (tp{mv['tp']} fp{mv['fp']} fn{mv['fn']} tn{mv['tn']})\n   latency p50={f(lat['p50'], 2)}s "
                   f"p95={f(lat['p95'], 2)}s  single-stream {f(r['throughput_rps_single_stream'], 2)} req/s  "
                   f"tokens/verdict {r['tokens']['prompt_mean']:.0f}+{r['tokens']['completion_mean']:.0f}")
    if "burst_concurrency4" in res:
        b = res["burst_concurrency4"]
        out.append(f"burst @4: {b['n']} requests in {b['wall_s']:.1f} s = {b['throughput_rps']:.2f} req/s")
    out.append(f"requests logged {res['requests']}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--limit", type=int, help="only the first N samples of a fixed random order")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-requests", type=int, default=450)
    ap.add_argument("--gap", type=float, default=4.2, help="seconds between request starts after the burst")
    ap.add_argument("--burst", type=int, default=8)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    load_env()
    info = None
    if not a.analyze_only:
        info = run_eval(a.frames, a.limit, min(a.concurrency, 4), a.max_requests, a.gap, a.burst, a.retry_errors)
    res = analyze(a.frames, info)
    print(format_summary(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
