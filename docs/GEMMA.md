# Gemma 4 in InterCycle

Gemma 4 runs behind one OpenAI-compatible client (`intercycle/llm/`). It does three jobs. **None of them can authorise a robot to enter a zone.**

## Safety boundary (read first)

**Gemma never decides robot entry.** Entry stays with the SIPP planner, the robot's own local camera check, and the plant's light curtains / area scanners (design assumptions in `BRIEF.md`). Gemma output can only do three things:

1. lower a confidence,
2. add a warning,
3. flag "person in zone", which *suppresses* a published safe window.

There is deliberately no field, flag or code path that raises a confidence, extends a window or says "go". `VerifyResult` has no `entry_allowed`; `ops_agent` answers are text with predictions labelled as predictions; `commission` output is `status="proposed"` until a human confirms. Every prompt states this (`BOUNDARY` in `tasks.py`). If Gemma is down or returns junk, `verify_state` returns the observer's own confidence unchanged (`ok=False`, warning `gemma_unavailable`): the system behaves exactly as it does without Gemma. A test checks the invariant over random verdicts (`test_verify_state_can_only_add_caution`).

## The three uses

| # | Task | What it does | Why it matters |
|---|---|---|---|
| a | `verify_state(images, machine_id, expected)` | Looks at 1..N crops (oldest first) around one machine's hazard zone and returns `{moving, person_in_zone, confidence, notes}`. The observer calls it at chosen timestamps. A person in the zone suppresses safe windows; disagreement with the observer's belief lowers its confidence. The observer's belief is *not* shown to the model (no anchoring); it is compared in code. | **Product:** a second, independent look at exactly the case the camera statistics handle worst (a person standing in an idle zone). **AMD:** one small-image call per check, latency and throughput are measurable (`scripts/gemma_eval_verify.py`). **Google prize:** real use of Gemma 4 vision. |
| b | `commission(image, camera_id, floor_plan_hint)` | One camera image in; proposed hazard zones, ROIs, machine types and inspection targets out, as polygons in image pixel coordinates with labels and per-item confidence. Bad polygons are dropped and reported. | **Product:** commissioning a camera is the expensive manual step of every deployment; a first draft for the engineer to correct cuts it from hours to minutes. **Google prize:** structured, multimodal output. |
| c | `ops_agent(question, tools)` | Answers maintenance questions by calling our read-only Python tools `next_safe_window(machine, before_time)`, `plan_inspection(targets, deadline)`, `shift_report(time_range)`. Native OpenAI tool calling, with an automatic ReAct-JSON fallback. | **Product:** the maintenance manager (buyer) asks in plain language and gets numbers from the real planner, not from the model. **Google prize:** Gemma 4 native function calling. |

**AMD usage proof.** Every call, live or replayed, appends one line to `out/llm_calls.jsonl`: `model`, `provider`, `endpoint_host`, `latency_s`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost_free`, `cost_usd`, `task`, `live`, `cached`, `ok`. Only lines with `"live": true` are real endpoint calls; replayed lines say `"live": false` and carry the recorded latency. The dashboard should count only `live` lines as AMD usage. Lines with an `"event"` key (for example `response_format_unsupported`) are not calls. The offline mock never writes to this file.

## Confirmed vs unknown (as of 17 Oct 2026, from public docs; no key yet)

| Item | Status |
|---|---|
| Token Factory is the hackathon's model access, console at `notebooks.amd.com/tokenfactory`, AMD AI Developer Program login (SAML), no billing / credit card | Confirmed (lablab ACT III page, console login page) |
| Using Gemma on AMD GPUs makes a project eligible for the Google prize ($5,000) | Confirmed (lablab ACT III page) |
| **Base URL and API key for the hackathon Token Factory** | **Unknown.** Behind the login. AMD's own docs say to use the base URL the console shows. |
| **Whether Gemma 4 is listed on the hackathon Token Factory, and its exact model id** | **Unknown.** `scripts/gemma_probe.py` lists `/v1/models` and picks it. |
| Gemma 4 family: E2B, E4B, 26B A4B (MoE), 31B dense, plus a 12B Unified card; Apache 2.0; text + image in (video as frames); native function calling; configurable thinking | Confirmed (Google model card, HF blog, vLLM recipe) |
| Hugging Face ids (what a vLLM server would serve): `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it` | Confirmed (vLLM recipe). The default model id here, `google/gemma-4-31B-it`, is **a guess** that the Token Factory uses the HF id (AMD's China-region Radeon Cloud catalog uses ids without an org prefix, e.g. `DeepSeek-V4-Flash`). |
| Gemma 4 day-0 support on AMD Instinct via vLLM; ROCm image `vllm/vllm-openai-rocm:gemma4` | Reported by AMD / vLLM docs; not verified by us |
| Image input over the OpenAI API as `image_url` parts; tool calling via `--tool-call-parser gemma4`; `response_format: json_schema` via guided decoding | Confirmed for vLLM-served Gemma 4. **Unknown for Token Factory**: depends on how AMD serves it. The probe tests each. |
| AMD Radeon Cloud Token Factory (China region, `https://developer.amd.com.cn/radeon/api/v1`) is OpenAI-compatible, accepts only a fixed parameter set (unlisted ones such as `seed`, `stop`, `top_k` are dropped silently), `response_format` and `tools` only on models that support them; its public catalog has no Gemma | Documented, but it is a *different region/service*; treat the hackathon one as possibly different |

The client is written to survive every "unknown": missing `response_format` falls back to a JSON-only prompt with validation and retries, missing tool support falls back to a ReAct JSON loop, and `INTERCYCLE_LLM_JSON_MODE` / `INTERCYCLE_LLM_TOOLS` can pin either behaviour.

## Switching on live mode when the key arrives

1. Get the base URL, key and Gemma 4 model id from the Token Factory console.
2. Set the environment (PowerShell shown; bash uses `export`):

   ```powershell
   $env:INTERCYCLE_LLM_BASE_URL = "<base URL from the console>"
   $env:INTERCYCLE_LLM_API_KEY  = "<key>"
   .venv\Scripts\python.exe scripts\gemma_probe.py
   ```

   The probe lists the models, picks the Gemma 4 id, checks text, image input (a synthetic sketch), `json_schema` and native tool calling, then prints the exact lines to add, for example `INTERCYCLE_LLM_MODEL=...`, and `INTERCYCLE_LLM_JSON_MODE=prompt` or `INTERCYCLE_LLM_TOOLS=react` if the endpoint lacks those features.
3. Record real responses (live call plus save to `out/llm_cache/`):

   ```powershell
   $env:INTERCYCLE_LLM_MODEL = "<id from the probe>"
   $env:INTERCYCLE_LLM_MODE  = "record"
   .venv\Scripts\python.exe scripts\gemma_eval_verify.py --frames out\gemma_fixture --concurrency 4
   ```

4. For the demo and for CI, run offline from the recorded cache: `$env:INTERCYCLE_LLM_MODE = "replay"` (the default whenever no key is set). The cache is keyed by (model, messages, params), so replay needs the same model id and the same prompts. `out/` is git-ignored: copy the recorded cache somewhere tracked (for example `demo_cache/llm/`) and set `INTERCYCLE_LLM_CACHE_DIR` if judges should be able to replay it.
5. Use `live` (no cache read or write) only for latency measurements.

If the Token Factory turns out not to serve Gemma 4, self-host it on an AMD Developer Cloud MI300X and point the same variables at it (this also counts as "Gemma on AMD GPUs"). From the vLLM Gemma 4 recipe (unverified by us):

```bash
docker run --device=/dev/kfd --device=/dev/dri -p 8000:8000 vllm/vllm-openai-rocm:gemma4 \
  vllm serve google/gemma-4-31B-it --enable-auto-tool-choice --tool-call-parser gemma4 \
  --reasoning-parser gemma4 --limit-mm-per-prompt image=4
# INTERCYCLE_LLM_BASE_URL=http://<host>:8000/v1   INTERCYCLE_LLM_API_KEY=any   INTERCYCLE_LLM_MODEL=google/gemma-4-31B-it
```

For that case set `INTERCYCLE_LLM_PROVIDER=amd-developer-cloud` so the log labels it correctly (the default provider label is `amd-token-factory` only when the host ends in `amd.com`, `amd.com.cn` or `anruicloud.com`; otherwise `other:<host>`). Never label a non-AMD endpoint as AMD.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `INTERCYCLE_LLM_BASE_URL` | none | OpenAI-compatible base URL (ending in `/v1`) |
| `INTERCYCLE_LLM_API_KEY` | none | Bearer key |
| `INTERCYCLE_LLM_MODEL` | `google/gemma-4-31B-it` | model id (a guess until the probe confirms it) |
| `INTERCYCLE_LLM_MODE` | `record` if base URL and key are set, else `replay` | `live`, `record` or `replay` |
| `INTERCYCLE_LLM_JSON_MODE` | `auto` | `auto` tries `response_format` and remembers a refusal; `schema` requires it; `prompt` never sends it |
| `INTERCYCLE_LLM_TOOLS` | `auto` | `auto` native tool calls with ReAct fallback; `native`; `react` |
| `INTERCYCLE_LLM_CACHE_DIR`, `INTERCYCLE_LLM_LOG` | `out/llm_cache`, `out/llm_calls.jsonl` | locations |
| `INTERCYCLE_LLM_COST_FREE` | `1` | Token Factory needs no billing for participants; set `0` for a paid endpoint (log then has `cost_usd: null`) |
| `INTERCYCLE_LLM_TIMEOUT_S`, `INTERCYCLE_LLM_PROVIDER` | `60`, derived from host | |

## Usage

```python
from intercycle.llm import verify_state, commission, ops_agent

r = verify_state([crop_t0, crop_t1], "WELD_R1", {"moving": False, "confidence": 0.93}, frame_dt_s=0.5)
if r.suppress_window: ...          # person in zone: do not publish a safe window
conf = r.adjusted_confidence       # never above 0.93

proposal = commission(image, "cam-3", {"machines": ["WELD_R1", "LIFT_2"]})    # polygons in pixels, status "proposed"

ans = ops_agent("When can I inspect WELD_R1 before 07:30?",
                {"next_safe_window": f1, "plan_inspection": f2, "shift_report": f3},
                context="now=07:25:00")
```

Tool contracts: `next_safe_window(machine: str, before_time: float)`, `plan_inspection(targets: list[str], deadline: float)`, `shift_report(time_range: tuple[float, float])`. Times are sim seconds since 00:00; the model may write `HH:MM`, which is converted before the call. Results must be JSON-serialisable. Other callables can be passed too (schema derived from the signature) or a `ToolSpec` with an explicit schema.

## Evaluation harness

`scripts/gemma_eval_verify.py --frames DIR` reads `DIR/frames/*.png` and `DIR/labels.json` (`{file: {machine, moving, person_in_zone}}`; optional `images`, `machine_type`, `zone_hint`) and reports accuracy (moving, person, both), person-in-zone precision / recall / F1 with the confusion counts, latency p50 / p95, throughput, and writes `out/gemma_eval_verify.json`. In replay mode latency is the *recorded* live latency and the report says so.

`scripts/gemma_make_fixture.py` writes six labelled SYNTHETIC top-down sketches (arm home or extended, person inside the zone, outside it, or absent). `--mock` runs the harness with a pixel-counting stand-in (**not Gemma**, own cache `out/llm_cache_mock`, own log) to prove the plumbing offline. Later, feed it frames rendered by the simulator, where the labels are known for free.

## Results

To be filled after the first recorded run (do not report mock numbers as Gemma numbers).

| Task | Metric | Value | n | Model id | Mode | Date |
|---|---|---|---|---|---|---|
| verify_state (synthetic fixture) | accuracy moving / person / both | TBD | 6 | TBD | record | TBD |
| verify_state (synthetic fixture) | person-in-zone precision / recall | TBD | 6 | TBD | record | TBD |
| verify_state (simulator frames) | person-in-zone precision / recall | TBD | TBD | TBD | record | TBD |
| verify_state | latency p50 / p95 (s), live | TBD | TBD | TBD | live | TBD |
| verify_state | throughput (req/s, concurrency 4) | TBD | TBD | TBD | live | TBD |
| commission | zones kept / dropped; human-edit effort | TBD | TBD | TBD | record | TBD |
| ops_agent | tool-call success; native vs ReAct | TBD | TBD | TBD | record | TBD |

## Citations

- Gemma 4 model card (variants, modalities, function calling, thinking, Apache 2.0): https://ai.google.dev/gemma/docs/core/model_card_4 (the page notes that it is being moved to gemma.ai)
- Gemma 4 launch: https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ and https://huggingface.co/blog/gemma4
- vLLM recipe for Gemma 4 (HF ids, `image_url`, tool-call and reasoning parsers, `json_schema`, ROCm image): https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html
- AMD, "Day 0 Support for Gemma 4 on AMD Processors and GPUs": https://www.amd.com/en/developer/resources/technical-articles/2026/day-0-support-for-gemma-4-on-amd-processors-and-gpus.html
- AMD Developer Hackathon ACT III (Token Factory, Google Gemma prize): https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-iii ; Token Factory console: https://notebooks.amd.com/tokenfactory
- AMD Radeon Cloud API docs (OpenAI-compatible shared model API, accepted parameters; China region): https://amd-aim.github.io/radeon-cloud-docs/api/overview/ and https://amd-aim.github.io/radeon-cloud-docs/api/chat-completions/
- OpenAI-compatible Chat Completions API (messages, `image_url` parts, `tools`, `response_format`): https://platform.openai.com/docs/api-reference/chat
