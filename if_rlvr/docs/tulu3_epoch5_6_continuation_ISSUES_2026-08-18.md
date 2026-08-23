# Issue tracker: Tulu3-8B-DPO epoch 5–6 continuation (2026-08-18)

Standalone tracking doc, split out from the narrative writeup in
`if_rlvr_tulu3_epoch5_6_continuation_review_2026-08-18.md` (that doc has full
root-cause detail, timing tables, and citations for every item below — this
one is the scannable, per-issue view for follow-up). Each issue is
independently actionable; statuses are current as of this writing.

---

## ISSUE-1: numba/numpy ABI break kills every vLLM engine at import time

- **Type**: bug (environment)
- **Status**: **Fixed**, this session
- **Severity**: blocking — every vLLM engine in a run fails identically
- **Symptom**: `ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.5.` at preflight, ~20s into launch.
- **Root cause**: `verl` conda env had `numba==0.61.2` (hard-requires `numpy<=2.2`) against actual `numpy==2.5.2`. A prior workaround ("numba 0.67.0 installed instead," documented in `llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh` comments) had not persisted on this environment/image.
- **Fix applied**: `pip install --upgrade numba==0.67.0` in the `verl` conda env. Produces a benign pip resolver warning (vllm wants `numba==0.61.2`) — safe, since vLLM's only numba usage here (speculative-decode n-gram proposer) is never exercised.
- **Follow-up needed**: pin `numba>=0.67.0` in the environment spec/Dockerfile so this doesn't need re-fixing on the next fresh instance. **Not yet done.**

---

## ISSUE-2: `HF_HOME` silently resolves to a quota-limited network mount

- **Type**: bug (environment / infra)
- **Status**: **Fixed**, this session (workaround only — see follow-up)
- **Severity**: blocking for any large model download
- **Symptom**: `OSError: ... Disk quota exceeded (os error 122)` 44 GB into a 65.2 GB weight download.
- **Root cause**: inherited `HF_HOME=/workspace/.cache/huggingface/` is a network mount (`mfs#ca-mtl-1.runpod.net:9421`) whose `df`-reported "209 TB available" is the **cluster-wide** aggregate, not this pod's actual per-tenant quota. The existing scripts' own `free_gb()` safety check trusts `df` and was fooled the same way.
- **Fix applied**: force `HF_HOME`/`HF_HUB_CACHE`/`HF_DATASETS_CACHE` onto the local root overlay (`${VERL_DIR}/.cache/huggingface`, confirmed non-network, 957 GB free) at the top of the orchestrator script, before any per-leaf-script cache logic runs. Deleted the 44 GB partial download from `/workspace` to free quota headroom.
- **Follow-up needed**: this is a workaround scoped to the continuation orchestrator only. Any *other* script on this box that relies on the inherited `HF_HOME` default will hit the same wall. Consider fixing the shell profile's default `HF_HOME`, or hardening `free_gb()` to not trust `df` on network mounts. **Not yet done.**

---

## ISSUE-3: `openai_harmony` tiktoken cache corrupted by concurrent-write race

- **Type**: bug (race condition)
- **Status**: **Fixed**, this session — now permanent (baked into the launch script, not a one-off)
- **Severity**: blocking, and misleading to diagnose (looked like all 8 replicas crashed; only 1 did)
- **Symptom**: `openai_harmony.HarmonyError: invalid tiktoken vocab file: ... could not split on ' '` on exactly one of 8 gpt-oss-120b replicas, ~4 minutes after that replica had otherwise finished loading successfully. The dead replica then broke the trainer's "sleep all judges" call, crashing the whole run — which read as "all 8 replicas failed" when only 1 did.
- **Root cause**: `load_harmony_encoding()` caches its parsed vocab in a content-addressed file (`/tmp/tiktoken-rs-cache/<hash>`); the write is not atomic, and all 8 `vllm serve` processes request the same encoding at ~the same time on first use, so the losers of the race get a corrupted file.
- **Fix applied**: pre-warm the cache with one sequential Python call (`load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)`) before spawning any of the 8 replicas. Now in `llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh`'s weight-prefetch section permanently.
- **Follow-up needed**: none for this script. Worth remembering as a general pattern if any *other* N-way-concurrent model launch in this repo lazily populates a shared first-use cache (tokenizer downloads, compiled-kernel caches).

---

## ISSUE-4: CUDA OOM in gpt-oss-120b's Triton MoE kernel, mid-training

- **Type**: bug (resource exhaustion) — **also a live throughput/config trade-off, see ISSUE-6**
- **Status**: **Fixed**, this session (config change, verified through a real step)
- **Severity**: blocking, and recurring across retries (verl's own auto-resume could not fix it — see notes)
- **Symptom**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 900.00 MiB.` (later recurred as "1.96 GiB") on one replica mid-step, killing that replica and cascading into a full trainer crash.
- **Root cause**: gpt-oss-120b's Triton MoE kernel (`gpt_oss_triton_kernels_moe.py` → `matmul_ogs.py:340`) does a raw `torch.empty()` workspace allocation **outside vLLM's own memory pool** — not bounded by `gpu_memory_utilization`, and scales with how many tokens are in that forward call (`max_num_batched_tokens`/`max_num_seqs`). At the original settings (32768/128) this pushed usage to ~72-73 GiB, colliding with the trainer's own resident footprint on the same GPU.
- **Fix applied**: `IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS` 32768→8192, `IF_LLM_VERIFIER_MAX_NUM_SEQS` 128→64. Did **not** lower `gpu_memory_utilization` (infeasible — already barely above the 65.2 GB weight floor at 0.85).
- **Verified cost**: re-run step 1 completed cleanly, 0 OOM occurrences, ~5% slower reward phase, ~1.7% slower overall step. Cheap fix.
- **Note on verl's auto-resume**: `resume_mode=auto`/`IF_MAX_RETRIES=10` (built into the shared entrypoint script) could not recover from this on its own — it only restarts the Python trainer process, not the separately-managed verifier vLLM servers, so it kept re-hitting the same OOM on other replicas across retries rather than healing. Had to manually kill the whole process tree and apply the config fix. **Worth flagging upstream if this class of failure (external service, not the trainer itself, dies) recurs** — the auto-resume loop's scope doesn't cover it.
- **Follow-up needed**: see ISSUE-6 — this fix has a real throughput cost that's worth revisiting once/if there's more memory headroom.

---

## ISSUE-5: original launch scripts and configs for 2 of 4 checkpoints were never committed to this repo

- **Type**: process gap
- **Status**: **Open** — worked around, not fixed
- **Severity**: medium — cost real investigation time and leaves two facts permanently unverifiable
- **Description**: neither a Tulu3-8B-DPO "constraint-only" script nor a Tulu3-8B-DPO + gpt-oss-120b-judge script existed in this repo when this task started (only analogous scripts for *different* policy models). Both had to be reconstructed from scratch by analogy to sibling scripts. Separately, the anchor script (`llama31_tulu3_8b_dpo_anchor_grpo_nonreason.sh`) references `validate_upload_tulu3_anchor_cache.py`, which does not exist anywhere in this repo and would fail immediately if actually invoked (`TULU3_VALIDATE_ANCHOR_BEFORE_TRAIN` must be `false`, which is not this script's own default).
- **Concrete cost of this gap**: `ab6ix6bq`'s (the original gpt-oss-120b judge run) `wandb-metadata.json` shows it was launched from a **different git remote/commit** (`ssangyeon/if-rlvr-rlvr` @ `a3033f0543b5d2d7369dddb0da1950cb6371de9b`) than what's checked out here. Its exact verifier-server CLI flags (`max_num_seqs`, `gpu_memory_utilization`, replica topology beyond the 4-URL count visible in wandb config) are permanently unrecoverable — no server logs survive, and the launch script itself lives only in that other checkout. This directly blocked fully resolving ISSUE-7 below.
- **Recommendation**: commit the actual launch script for any run whose checkpoint is meant to be reused later, even if it's "just a copy with some env vars changed" from an existing script. The two new scripts written this session (`llama31_tulu3_8b_dpo_constraint_only_nonreason.sh`, `llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh`) close this gap **going forward** for these two configurations specifically, but do not recover the historical ones.

---

## ISSUE-6: gpt-oss-120b judge reward phase is dominated by config-tunable factors, not fundamental generation cost alone

- **Type**: performance / tuning opportunity
- **Status**: **Open** — root-caused with direct evidence, not yet acted on
- **Severity**: low urgency (training works correctly as-is), high value if addressed (directly shortens the 64.7h/4-epoch estimate for this variant, the slowest of the four)
- **Description**: measured directly from the live vLLM engine's own request-accounting log during one reward-computation window: ~81% of the window is the judge fleet genuinely saturated at its configured `max_num_seqs` ceiling (continuously busy, not idle); ~19% is a long-tail drain where a shrinking number of individual slow requests (unbounded reasoning length) hold up the batch. Both numbers are set by config:
  - The saturated-phase duration is set by `max_num_seqs`/`max_num_batched_tokens` — **the same knob ISSUE-4's fix just halved**, for an unrelated (OOM-safety) reason.
  - The tail length is set by `max_tokens=8192` + `omit_max_tokens=true` + unset `reasoning_effort` — currently unbounded. The sibling Qwen3-30B-A3B judge script already caps this (`max_tokens=1024`) and documents the exact tail-shortening rationale in its own comments.
- **Proposed fix** (not yet applied — needs a decision on acceptable judge-verdict-quality trade-off first): set `IF_LLM_VERIFIER_MAX_TOKENS` to a few hundred and/or `IF_LLM_VERIFIER_REASONING_EFFORT=low` for gpt-oss-120b, directly targeting the tail. Separately, once headroom allows (e.g. if the OOM's root cause in ISSUE-4 gets a real fix upstream), revisit raising `max_num_seqs` back up to shrink the saturated phase.
- **Why this is flagged as "not fundamental"**: a single anchor-style forward pass over the same token volume costs ~40s (§6.3 of the review doc); the judge's *equivalent* work is inherently more expensive (autoregressive decode, per-request prefill with no cross-request batching) but not by the full 10-16x observed — that magnitude is inflated by the two tunable knobs above, not by physics alone.

---

## ISSUE-7: original run used 4 gpt-oss-120b replicas, this continuation uses 8 — unresolved whether/how much this matters

- **Type**: open question
- **Status**: **Unresolved**, likely low-impact but not proven
- **Severity**: low (doesn't block anything; flagged for completeness)
- **Description**: the original production run (`ab6ix6bq`) logged exactly 4 `if_llm_verifier_base_urls`, not 8 — a materially different topology than this continuation's 8×TP1. This wasn't knowable in advance (see ISSUE-5 — the real launch script isn't in this repo). Whether the original used TP=2×4-replicas (matching the user's *original* stated plan, which this task's own research overrode based on a different precedent script for a different policy model) or some other split is unconfirmed.
- **Why not chased further**: the server-side concurrency settings for that historical run (`max_num_seqs`, `gpu_memory_utilization`) are vLLM CLI flags to a separately-launched process, never captured in the wandb-logged Hydra config, and no server logs survive. ISSUE-6's finding (the dominant cost is genuinely capacity-bound decode/prefill volume, and this continuation's 8×TP1 setup lands within ~8% of the original's per-step pace anyway) suggests the *concurrency ceiling value* matters more than how it's split across replica count — but this is inference, not direct measurement of the original's replica-level throughput.
- **Follow-up needed**: none blocking. If a future run of this same experiment is set up, worth deciding replica topology deliberately (informed by ISSUE-6) rather than defaulting to either historical choice.

---

## Summary table

| # | Title | Status | Severity |
|---|---|---|---|
| 1 | numba/numpy ABI break | Fixed (this session) | blocking |
| 2 | `HF_HOME` on quota-limited mount | Fixed (workaround) | blocking |
| 3 | `openai_harmony` cache race | Fixed (permanent) | blocking |
| 4 | CUDA OOM in gpt-oss-120b MoE kernel | Fixed (config change) | blocking |
| 5 | Original launch scripts never committed | Open (worked around) | medium |
| 6 | Judge reward-phase tunable inefficiency | Open (root-caused, not applied) | low urgency / high value |
| 7 | 4-vs-8-replica topology discrepancy | Unresolved | low |
