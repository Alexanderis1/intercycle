"""Label the episodes of a batch infrastructure-invalid or not, from the logs alone (PLAN_TWIN §5 Rules).

    python scripts/classify_invalid.py <batch_dir> [--commit]

Rule (PREREG; it cannot be amended): an episode is INFRASTRUCTURE-INVALID when

1. its logs are missing: no episode record `<batch>/<tag>.json`, or the pulled episode directory `<batch>/<tag>/`
   lacks a required log (episode.json and sim.log always; for a twin run also summary.json and log.parquet or
   log.jsonl), or the record has no sim summary;
2. Modal or RunPod returned an error code: the record is the client's error stub (`error` without `phases`, written
   by intercycle_eval/batch.py when the remote call raised), or a provider status FAILED / CANCELLED / TIMED_OUT;
3. the container failed, was pre-empted or ran out of memory before the robot's first command: no first command in
   the logs (no `t_goal_sent` in the goal record, no `LEG ` line in goal.log) and either the episode returned an
   error or a log shows a kill / OOM / pre-emption signature.

Every other episode is VALID, including one that errored after the robot's first command: EVAL scores it as a
failure (Y = 0 with full T), never drops it. The tags come from `<batch>/batch.json` (cfgs) so an episode with no
record at all is still listed. The list is written to `<batch>/invalid.json` once, before any metric: the script
refuses when a metric file already sits in the batch directory or the list exists; `--commit` commits only that file.
Nothing here opens truth; it reads episode records and logs only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

CONTRACT = "twin.invalid/1"
OUT_NAME = "invalid.json"
METRIC_GLOBS = ("primary_metric*.json", "metrics*.json", "results*.json")
ALWAYS = ("episode.json", "sim.log")
TWIN = ("summary.json",)
TWIN_ONE_OF = ("log.parquet", "log.jsonl")
PROVIDER_FAILED = {"FAILED", "CANCELLED", "TIMED_OUT"}
INFRA_SIG = re.compile(r"out of memory|OOMKill|Killed process|exit(?:ed)? (?:with )?code:? ?137|\bSIGKILL\b|"
                       r"pre-?empt|InputCancellation|ContainerTerminated|worker (?:was )?(?:lost|terminated)",
                       re.IGNORECASE)
LOGS_SCANNED = ("sim.log", "goal.log", "nav2.log", "bridge.log")


class ClassifyError(RuntimeError):
    pass


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def expected_tags(batch: Path) -> list[str]:
    b = _read_json(batch / "batch.json")
    if b is None or not isinstance(b.get("cfgs"), list) or not b["cfgs"]:
        raise ClassifyError(f"{batch / 'batch.json'} is missing or lists no cfgs: the episode set is unknown")
    tags = [str(c["tag"]) for c in b["cfgs"]]
    if len(set(tags)) != len(tags):
        raise ClassifyError("batch.json repeats a tag")
    return tags


def _cfg(batch: Path, tag: str) -> dict:
    b = _read_json(batch / "batch.json") or {}
    return next((c for c in b.get("cfgs", []) if c.get("tag") == tag), {})


def first_command(rec: dict, ep_dir: Path) -> bool:
    g = rec.get("goal")
    if isinstance(g, dict) and isinstance(g.get("t_goal_sent"), (int, float)):
        return True
    gl = ep_dir / "goal.log"
    if gl.is_file():
        return any(ln.startswith("LEG ") for ln in gl.read_text(encoding="utf-8", errors="replace").splitlines())
    return False


def infra_signature(rec: dict, ep_dir: Path) -> str | None:
    texts = [str(rec.get("error") or ""), str(rec.get("log_tail") or "")]
    texts += [(ep_dir / n).read_text(encoding="utf-8", errors="replace") for n in LOGS_SCANNED if (ep_dir / n).is_file()]
    for t in texts:
        m = INFRA_SIG.search(t)
        if m:
            return m.group(0)
    return None


def classify_one(batch: Path, tag: str) -> dict:
    rec = _read_json(batch / f"{tag}.json")
    if rec is None:
        return {"invalid": True, "reason": "logs missing: no episode record", "commanded": False}
    if rec.get("error") and "phases" not in rec:
        return {"invalid": True, "reason": f"Modal/RunPod returned an error: {str(rec['error'])[:300]}",
                "commanded": False}
    if str(rec.get("provider_status") or "").upper() in PROVIDER_FAILED:
        return {"invalid": True, "reason": f"provider status {rec['provider_status']}", "commanded": False}
    ep = batch / tag
    twin = bool((rec.get("cfg") or {}).get("twin") or _cfg(batch, tag).get("twin"))
    need = list(ALWAYS) + (list(TWIN) if twin else [])
    missing = [n for n in need if not (ep / n).is_file()]
    if twin and not any((ep / n).is_file() for n in TWIN_ONE_OF):
        missing.append("|".join(TWIN_ONE_OF))
    if rec.get("sim") is None:
        missing.append("sim summary in the record")
    cmd = first_command(rec, ep)
    if missing:
        return {"invalid": True, "reason": f"logs missing: {missing}", "commanded": cmd}
    if not cmd:
        sig = infra_signature(rec, ep)
        if rec.get("error") or sig:
            why = f"error {str(rec.get('error'))[:200]}" if rec.get("error") else f"log signature {sig!r}"
            return {"invalid": True, "reason": f"container failed before the robot's first command ({why})",
                    "commanded": False}
    out = {"invalid": False, "reason": "valid", "commanded": cmd}
    if rec.get("error"):
        out["reason"] = "valid: errored after the first command, scored as a failure (Y = 0, full T)"
    return out


def classify(batch: Path) -> dict:
    for pat in METRIC_GLOBS:
        hit = sorted(batch.glob(pat))
        if hit:
            raise ClassifyError(f"a metric file exists before classification: {hit[0].name}")
    if (batch / OUT_NAME).exists():
        raise ClassifyError(f"{batch / OUT_NAME} exists: a batch is classified once")
    tags = expected_tags(batch)
    eps = {t: classify_one(batch, t) for t in tags}
    return {"contract": CONTRACT, "batch": batch.name, "rule": "PLAN_TWIN §5 Rules (infrastructure-invalid)",
            "classified_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "n": len(tags),
            "invalid": sorted(t for t, e in eps.items() if e["invalid"]), "episodes": eps}


def write(batch: Path, out: dict) -> Path:
    p = batch / OUT_NAME
    p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return p


def commit(path: Path) -> str:
    repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path.parent, check=True,
                               capture_output=True, text=True).stdout.strip())
    rel = path.resolve().relative_to(repo.resolve()).as_posix()
    subprocess.run(["git", "add", "--", rel], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", f"classify_invalid: {rel} (before any metric)", "--", rel],
                   cwd=repo, check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="infrastructure-invalid episodes from the logs alone (PLAN_TWIN §5)")
    ap.add_argument("batch", type=Path)
    ap.add_argument("--commit", action="store_true", help="commit only <batch>/invalid.json")
    a = ap.parse_args(argv)
    try:
        out = classify(a.batch)
    except ClassifyError as e:
        print(f"classify_invalid REFUSED: {e}", file=sys.stderr)
        return 2
    p = write(a.batch, out)
    print(f"{len(out['invalid'])} of {out['n']} infrastructure-invalid -> {p}")
    if a.commit:
        print(f"committed {commit(p)[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
