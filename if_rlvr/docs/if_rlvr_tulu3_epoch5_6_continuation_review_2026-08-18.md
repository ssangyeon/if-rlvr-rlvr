# Tulu3-8B-DPO Epoch 5–6 Continuation: Infrastructure Fixes and Reward-Path Timing Analysis

*Working document for internal review — 2026-08-18.*
*Every claim is tagged **[confirmed]** (directly measured/observed in this investigation, with a citation), **[inferred]** (reasoned from confirmed evidence but not independently measured), or **[unresolved]** (open question — flagged so it isn't mistaken for a settled fact later).*

---

## Table of contents

1. [Task and scope](#1-task-and-scope)
2. [TL;DR](#2-tldr)
3. [Continuation design](#3-continuation-design)
4. [Infrastructure incidents: root cause and fix, in order encountered](#4-infrastructure-incidents-root-cause-and-fix-in-order-encountered)
5. [Corrections to the original plan](#5-corrections-to-the-original-plan)
6. [Wandb archaeology: how long did the original 4-epoch runs actually take](#6-wandb-archaeology-how-long-did-the-original-4-epoch-runs-actually-take)
7. [Root cause: why is the LLM-judge reward path so much slower than the anchor's PPL pass](#7-root-cause-why-is-the-llm-judge-reward-path-so-much-slower-than-the-anchors-ppl-pass)
8. [Recommendations](#8-recommendations)
9. [Appendix: file inventory, run-ID registry, exact commands](#9-appendix)

---

## 1. Task and scope

Continue 4 already-4-epoch-trained `allenai/Llama-3.1-Tulu-3-8B-DPO` IF-RLVR/GRPO checkpoints for 2 more epochs each (→ 6 epochs total), sequentially, on one 8×H100 80GB node:

| # | Reward mechanism | Source checkpoint (epoch 4) | Target repo (epoch 5–6, appended) |
|---|---|---|---|
| 1 | openai/gpt-oss-120b LLM judge, threshold 5 | `just1nseo/llama31-tulu3-8b-dpo-if-rlvr-judge-only` @ `global_step_364` | same repo |
| 2 | constraint-only (rule-based IFEval) | `just1nseo/llama31-tulu3-8b-dpo-if-rlvr-constraint-only` @ `global_step_364` | same repo |
| 3 | anchor (`p(y\|x)` PPL band, coeff 0.1) | `sangyon/llama31_tulu3_8b_dpo_grpo_nonthink_anchor_pyx01_b1024_c1_t1_2k` @ `global_step_364` | new repo `just1nseo/llama31-tulu3-8b-dpo-if-rlvr-anchor-only` |
| 4 | Qwen/Qwen3-30B-A3B LLM judge, threshold 7 | `just1nseo/tulu3-8b-dpo-grpo-q30ba3b-t7` @ `global_step_364` | same repo |

No local FSDP checkpoint (model/optimizer/extra shards) survives on this box for any of the 4 runs — only the bf16 `hf_model` export made it to the Hub via `push_checkpoints_to_hf.py`. Each run therefore downloads `global_step_364` and warm-starts a **fresh trainer instance** (`TOTAL_EPOCHS=2`, reset Adam state) rather than a true `trainer.resume_mode=resume_path` — acceptable because LR is constant (`5e-7`, no schedule to resume mid-curve). `push_checkpoints_to_hf.py --step-offset 364` (added for this task, §9.1) keeps the new run's own step-91/182-ish local numbering from colliding with the already-uploaded epoch 1–4 steps in the same repo.

This document covers: the infrastructure incidents hit while bringing run 1 up (§4), the corrections made to the user's original stated plan after checking it against repo evidence (§5), and — the bulk of this document — a deep, evidence-based investigation into why the four reward mechanisms have such different wall-clock costs, triggered by a user challenge that a naive "judge = slower than anchor" explanation was insufficient (§6–7).

---

## 2. TL;DR

- Bringing up run 1 (gpt-oss-120b judge) required four separate fixes before it trained stably: a numba/numpy ABI break, an HF cache pointed at a quota-limited network mount, an 8-way concurrent-write race corrupting a shared tiktoken cache file, and a CUDA OOM in gpt-oss-120b's Triton MoE kernel. All four are documented in §4 with root cause and the exact fix, because at least three of them (numba, the quota mount, the OOM tuning) are environment/config properties of this box and will very likely recur on a fresh instance.
- One trainer code change was made, opt-in and off by default: `IF_REF_POLICY_MODEL_PATH_OVERRIDE` (§4.4, §9.2) lets the anchor run's `ref` policy stay pinned to the original base model even though `actor_rollout_ref.model.path` now points at the epoch-4 checkpoint. Without it, verl's architecture leaves no way to give `ref` an independent weight source — it is *always* a copy of whatever the actor uses.
- Two numeric corrections to the user's stated plan, both verified against the actual repo/HF state rather than memory: the LLM-judge thresholds were reversed (gpt-oss-120b is 5, Qwen3-30B-A3B is 7, not the other way around), and gpt-oss-120b does not need 2-GPU tensor-parallel sharding — its MXFP4 weights measure 65.2 GB, fit one 80 GB H100, and two other scripts already in this repo already run it that way (§5).
- The original 4-epoch runs' wall-clock costs, pulled from wandb history (not memory, not summary last-values — see the methodological trap in §6.1): constraint-only ≈16.4h, anchor ≈36.4h (one run had a 1.9-hour freak stall — excluded as a real anomaly, not representative), Qwen3-30B-A3B judge ≈38.0h, gpt-oss-120b judge ≈64.7h, all for a full 364-step (4-epoch) run.
- The anchor/Qwen-judge near-tie (~36–38h) is real but coincidental — traced to two unrelated mechanisms (anchor: cheap forward-pass scoring; Qwen-judge: cheap-per-call generation, but response length nearly doubles over training) that happen to land in the same range (§6.3–6.4).
- The gpt-oss-120b judge's reward phase, measured directly from the live vLLM engine's own request-accounting logs, spends ~80% of its time genuinely saturated at its configured concurrency ceiling (not idle) and ~20% draining a long tail of slow individual requests. Both numbers are set by config, not physics: the concurrency ceiling is `max_num_seqs` (which this task's own OOM fix cut in half), and the tail length is bounded by `max_tokens`/`omit_max_tokens`, which are currently unbounded (§7).

---

## 3. Continuation design

### 3.1 Sequencing
One orchestrator script (`if_rlvr/exps/bidirectional/run_tulu3_8b_dpo_epoch5_6_continuation.sh`) runs the 4 experiments strictly sequentially — download checkpoint → launch training → external HF-push watcher → wait for exit → next — so the whole node is available to each run in turn. A run's nonzero exit stops the sequence rather than proceeding onto the next run on top of an unresolved failure.

### 3.2 Why a fresh trainer instead of a real resume
verl's `trainer.resume_mode=resume_path` requires local FSDP shards (`model`/`optimizer`/`extra` — see `verl/trainer/config/config.py:38`) which no longer exist for any of the 4 runs. Only the `hf_model` bf16 export survives (on the Hub). A fresh trainer with `TOTAL_EPOCHS=2` is therefore used, accepting a cold-started Adam optimizer — a non-issue given the constant LR schedule (`5e-7`, `num_warmup_steps=0` confirmed in the live run's own log).

### 3.3 New files added for this task
See §9.1 for the full inventory. In brief: `download_hf_checkpoint_step.py` (materializes one `global_step_<N>` locally, avoiding the bare-repo-id assumption baked into the existing scripts' own weight-prefetch logic), `--step-offset` on `push_checkpoints_to_hf.py`, two new leaf scripts for configurations that didn't previously exist as committed files for this policy model (`llama31_tulu3_8b_dpo_constraint_only_nonreason.sh`, `llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh`), and the orchestrator itself.

---

## 4. Infrastructure incidents: root cause and fix, in order encountered

All four were hit bringing up run 1 (gpt-oss-120b judge) between 2026-08-17 16:33 UTC and 17:25 UTC. Each caused a full restart of the orchestrator (cheap — the checkpoint download and gpt-oss-120b weight download are both cached after the first attempt).

### 4.1 numba/numpy ABI break

**Symptom** [confirmed]: preflight step failed within ~20s of launch:
```
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.5.
```
raised from `vllm.v1.worker.gpu_worker` → `v1/spec_decode/ngram_proposer` → `numba`.

**Root cause** [confirmed]: the `verl` conda env (`/root/miniconda3/envs/verl`) had `numba==0.61.2` installed, which hard-asserts `numpy<=2.2` at import time; the env's actual numpy is `2.5.2`. A comment already present in `llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh` (lines ~36–38) claims *"numba 0.67.0 is installed instead"* as a known workaround for this exact box — that workaround had not survived on this particular environment/image.

**Fix** [confirmed]: `pip install --upgrade numba==0.67.0` (pulls `llvmlite` 0.49.0). This produces a pip dependency-resolver warning (`vllm 0.11.0 requires numba==0.61.2 ... but you have numba 0.67.0`) which is safe to ignore — vLLM's only numba usage on this path is speculative-decoding's n-gram proposer, which this setup never exercises.

**Recurrence risk**: high on any freshly provisioned instance of this environment. Worth pinning `numba>=0.67.0` in the environment spec rather than fixing ad hoc again.

### 4.2 HF cache on a quota-limited network mount

**Symptom** [confirmed]: ~7 minutes into the second launch attempt, mid gpt-oss-120b weight download:
```
OSError: I/O error: IO Error: Disk quota exceeded (os error 122)
```
44 GB into the 65.2 GB transfer.

**Root cause** [confirmed]: the shell's inherited `HF_HOME=/workspace/.cache/huggingface/` resolves to a network mount (`mfs#ca-mtl-1.runpod.net:9421`). `df -h /workspace` reports `965T size / 209T avail` — but this is the **cluster-wide** aggregate across (presumably) many tenants of that MooseFS volume, not this pod's actual allocation. The existing scripts' own free-space safety check (`if_rlvr/exps/bidirectional/llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh`'s `free_gb()` helper, requiring ≥120 GB via `df`) passed cleanly on this path precisely because `df` cannot see the real per-tenant quota — it only reports the shared filesystem's aggregate.

**Fix** [confirmed]: force `HF_HOME`/`HF_HUB_CACHE`/`HF_DATASETS_CACHE` onto the local root overlay (`${VERL_DIR}/.cache/huggingface`, confirmed genuinely local — `df -h /` reports `overlay` filesystem, 1000G size / 957G avail, not network-backed) at the top of the orchestrator, before any leaf script's own (df-based, and therefore fooled the same way) cache-placement logic runs. The 44 GB partial download was deleted from `/workspace` to free quota headroom before retrying.

**Generalizable lesson**: a `df`-based free-space check is not a reliable safety check on any mount where multi-tenant quotas exist below the filesystem's own reported capacity. This pattern (network mount reporting misleadingly large "available" space) is worth checking for explicitly on any new box before trusting inherited `HF_HOME`.

### 4.3 `openai_harmony` tiktoken-cache concurrent-write race

**Symptom** [confirmed]: ~7 minutes after the HF_HOME fix, one of 8 gpt-oss-120b verifier replicas (GPU4, port 21204) crashed during API-server startup — *after* its engine had already finished loading and CUDA-graph-capturing (250+ seconds of otherwise-successful work):
```
openai_harmony.HarmonyError: invalid tiktoken vocab file: expected token and rank, could not split on ' ' at line 67054
```
(`vllm/entrypoints/harmony_utils.py:56` → `openai_harmony/__init__.py:689` → `load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)`). This single dead replica then broke the trainer's post-step "sleep all 8 judges" call (`ConnectionRefusedError` from `verl/experimental/reward_loop/reward_loop.py:361 _set_external_verifier_sleep`), which is not resilient to a dead endpoint and crashed the whole `TaskRunner` Ray actor — which then presented as "all 8 replicas died," even though only 1 of 8 was the actual cause; the other 7 were killed by this script's own `cleanup()` trap reacting correctly to the one genuine failure.

**Root cause** [confirmed]: `openai_harmony`'s encoding loader caches its parsed tiktoken vocab in a content-addressed file, `/tmp/tiktoken-rs-cache/<sha1-of-something>` (observed: `fb374d419588a4632f3f557e76b4b70aebbca790`, 3,613,922 bytes, 199,998 lines when correctly written). Because all 8 `vllm serve` processes request the *same* encoding at approximately the same time, and this cache write is not atomic (no write-to-temp-then-rename observed), multiple processes writing the same path concurrently corrupt it for whichever ones lose the race. Confirmed by direct inspection: after deleting and single-process-rewriting the file, a byte-identical-sized (3,613,922-byte) file was produced in 1.3 seconds and reloaded successfully twice in a row.

**Fix** [confirmed]: pre-warm the cache once, sequentially, with a single Python process, *before* launching any of the 8 replica processes:
```python
from openai_harmony import load_harmony_encoding, HarmonyEncodingName
load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
```
Now baked permanently into `llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh` (weight-prefetch section), not a one-off workaround.

**Generalizable lesson**: any N-way-concurrent-process launch of a model that lazily populates a shared, content-addressed, non-tenant-isolated cache file on first use is a race by construction. Pre-warming any such cache from a single process before fanning out is the general fix; this will recur for any future model whose runtime does similar first-use caching (tokenizer downloads, compiled-kernel caches, etc.) if it isn't already pre-warmed.

### 4.4 CUDA OOM in gpt-oss-120b's Triton MoE kernel

**Symptom** [confirmed]: after all 8 replicas booted cleanly (post-4.3 fix) and step 1 was actively serving judge requests, one replica raised:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 900.00 MiB.
GPU 0 has a total capacity of 79.19 GiB of which 679.38 MiB is free.
Including non-PyTorch memory, this process has 72.66 GiB memory in use.
Process 42728 has 3.66 GiB memory in use. Process 49892 has 2.02 GiB memory in use.
```
Traceback: `vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py:157 triton_kernel_fused_experts` → `triton_kernels/matmul_ogs.py:340 apply_allocation` → `output = torch.empty(allocation.output[0], ...)`.

**Root cause** [confirmed]: gpt-oss-120b's MoE forward pass uses a Triton kernel (`matmul_ogs`) whose output workspace is a **raw `torch.empty()` call outside vLLM's own pre-sized memory pool**. `gpu_memory_utilization=0.85` sizes vLLM's *own* pool (weights + KV cache + tracked activations), but this workspace allocation is not part of that accounting — it competes directly with whatever else is resident on the card at that instant, in this case the trainer's own leftover FSDP process residency (3.66 + 2.02 = 5.68 GiB across two processes on that GPU — itself higher than the ~1.4 GiB this repo's own comments elsewhere describe as the expected fully-offloaded actor residual, though this investigation did not fully resolve why this particular instance ran higher; see §7 note below on config-dependence). Crucially: the failed 900 MiB request was tiny relative to the 65.2 GB base footprint — this is workspace-scratch pressure at the margin, not a fundamentally undersized budget.

**Fix** [confirmed]: reduce the size of the MoE forward pass this workspace scales with — `IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS` 32768→8192, `IF_LLM_VERIFIER_MAX_NUM_SEQS` 128→64 — rather than lowering `gpu_memory_utilization` further (infeasible: 0.85×79.19 GiB ≈ 67.3 GiB is already barely above the 65.2 GB weight floor; going materially lower risks failing to even allocate the weights).

**Verified cost of the fix** [confirmed]: re-run of step 1 after the fix completed cleanly — 0 OOM occurrences, `timing_s/reward` 386.7s vs. 366.8s pre-fix-attempt (~5% slower), full step 699s vs. 687s (~1.7% slower). Negligible throughput cost for eliminating the crash. See §7.2 for how this specific setting also governs the *dominant* cost of the reward phase, not just OOM safety — the two are the same knob.

---

## 5. Corrections to the original plan

The user's initial instructions specified two facts from memory that turned out to conflict with the actual repo/HF state once checked; both were corrected against source-of-truth evidence rather than either silently complied with or silently overridden.

### 5.1 Judge thresholds were stated backwards

**Stated**: "oss and qwen30b judge... thresholds... are 7 and 5 each."
**Actual** [confirmed], three independent sources agree:
- `llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh:266`: `IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-7}`, and its own HF repo name (`tulu3-8b-dpo-grpo-q30ba3b-t7`) encodes this.
- `qwen3_17b_llmverifier_gptoss120b_bonus01_nonreason.sh:54`: `IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-5}` (gpt-oss-120b judge, different policy model, same judge/threshold pairing).
- `if_rlvr/docs/if_rlvr_anchor_reward_review_2026-08-11.md:47`: *"LLM judge: `IF + 0.1` if gpt-oss-120b rates the answer ≥ 5/10."*

**Corrected value used**: gpt-oss-120b judge = threshold 5; Qwen3-30B-A3B judge = threshold 7.

### 5.2 gpt-oss-120b does not need 2-GPU sharding

**Stated**: "since the model is larger than our GPU's vram, you should shard it to two GPUs per model, 4 models on the server in parallel."
**Actual** [confirmed]: `HfApi(files_metadata=True)` on `openai/gpt-oss-120b` measures the top-level (MXFP4, vLLM-loadable) safetensors at **65.2 GB** — under one 80 GB H100's capacity with no sharding. Two scripts already in this repo (`qwen3_17b_llmverifier_gptoss120b_bonus01_nonreason.sh`, `qwen3_4b_t4b_anchor_pyx01_llmverifier_gptoss120b_bonus01_reasoning.sh`) already run this exact model at `TP=1` on other policy models.

**Wrinkle discovered later** [confirmed, §6.2/§9.3]: the *actual* original production run for this exact experiment (`ab6ix6bq`) used **4 replicas**, not 8, per its logged `if_llm_verifier_base_urls` (4 URLs). This was not knowable in advance — the real launch script was never committed to this repo (see §6.2's git-remote finding) — but means the "corrected" 8×TP1 topology used for this continuation is *not* proven identical in aggregate throughput to the historical baseline; see §7.3 for why this matters and why it turned out not to be the dominant factor anyway.

**Corrected value used**: `TP=1`, 8 replicas, one gpt-oss-120b instance per GPU.

---

## 6. Wandb archaeology: how long did the original 4-epoch runs actually take

Triggered by the user's stated recollection that "the 4 epoch tulu run didn't take that long," which needed to be checked against actual wandb history rather than accepted or dismissed from memory.

### 6.1 Methodology, including a mistake made and corrected mid-investigation

Initial approach: `dict(run.summary)["timing_s/step"]` — wandb's stored **last logged value** for that key. This is wrong for any run whose per-step cost is not constant, and it produced a materially misleading first-pass comparison (anchor "363s/step" vs. Qwen-judge "598s/step," implying a ~65% gap) [confirmed error, self-identified after a direct user challenge]. Corrected method, used for all figures below: `run.history(keys=["timing_s/step", ...], pandas=True)` — the complete per-step series — then `.describe()` for the true mean/median/std, plus an explicit anomaly pass: flag any step `> 3× the run's own median` as a discrete anomaly, and separately compute `corr(timing_s/step, response_length/mean)` to distinguish *real, gradual* training dynamics (rising response length — must stay in any estimate) from *discrete infrastructure flukes* (a single stalled step — should not).

### 6.2 Per-run findings

All 4 wandb runs below crashed at step 362/364 except the Qwen-judge run, which finished cleanly at 364/364. "Cleaned" = mean after excluding discrete (`>3×median`) anomalies; "none found" means the raw mean already is the reasonable estimate.

| Variant | wandb run ID | Result | Steps logged | Raw mean (s/step) | Discrete anomaly excluded | Cleaned mean (s/step) | `corr(step_time, response_len)` | Reasonable full-364-step estimate |
|---|---|---|---|---|---|---|---|---|
| constraint-only | `1gyg97hj` | crashed @362 | 362 | 162.2 | none | 162.2 | **0.977** | **16.4 h** |
| anchor (pyx=0.1) | `myf68cle` | crashed @362 | 362 | 377.9 | step 107: 6,850s (reward phase only) | **359.9** | −0.023 | **36.4 h** |
| Qwen3-30B-A3B judge (t7) | `1n2avbdq` | **finished @364** | 364 | 376.3 | none (smooth trend, not a spike — max is only 1.59× median) | 376.3 | **0.963** | **38.0 h** |
| gpt-oss-120b judge (t5) | `ab6ix6bq` | crashed @362 | 362 | 639.5 | none (max is only 1.48× median) | 639.5 | 0.554 | **64.7 h** |

For reference, the current continuation's run 1 (`h02owfkp`, in progress at time of writing, gpt-oss-120b judge with the OOM fix from §4.4 applied): 69 logged steps, raw mean 696.0s/step, cleaned (excluding its own 2 known blips at steps 16 and ~55) ≈688s/step — **7.6% slower** than the original `ab6ix6bq`'s 639.5s, consistent with (and fully attributable to) the halved `max_num_seqs`/`max_num_batched_tokens` from §4.4, not an unexplained regression.

### 6.3 Anchor's step-107 anomaly, in full detail

Full per-component breakdown around the spike (all in seconds; `training/global_step` from `myf68cle`'s history):

| global_step | total | gen | **reward** | ref | old_log_prob | update_actor |
|---|---|---|---|---|---|---|
| 105 | 358.3 | 60.0 | 40.4 | 52.9 | 51.6 | 140.7 |
| 106 | 354.8 | 60.4 | 40.0 | 50.1 | 54.1 | 142.5 |
| **107** | **6849.99** | 69.6 | **6357.8** | 209.0 | 69.0 | 137.5 |
| 108 | 337.9 | 57.2 | 38.7 | 50.9 | 49.1 | 133.5 |
| 109 | 354.1 | 60.1 | 39.9 | 50.0 | 54.3 | 141.6 |

The anomaly is confined almost entirely to the `reward` component (which, for the anchor arm, includes the live `_compute_anchor_ppl_with_ref_policy` forward pass) — every other component that step is within normal range. The run recovers fully at step 108 with no further incident anywhere else in its 362 logged steps. This is [confirmed] a single isolated infrastructure stall (disk I/O, NCCL, GC pause, or similar — the specific external trigger is [unresolved]), not a systematic property of the anchor mechanism.

### 6.4 Qwen3-30B-A3B judge's response-length-driven trend, in full detail

Unlike the anchor's spike, this is a **real, gradual training dynamic**, not a fluke — `corr(timing_s/step, response_length/mean) = 0.963` across the whole 364-step run. Tail of the run:

| global_step | total (s) | response_length/mean (tokens) | timing_s/reward (s) | judge calls |
|---|---|---|---|---|
| 355 | 446.1 | 655.7 | 100.4 | 7,388 |
| 358 | 461.5 | 708.9 | 92.3 | 7,157 |
| 361 | 519.5 | 832.2 | 101.9 | 6,963 |
| 364 | 597.6 | 919.8 | 99.7 | 6,602 |

`timing_s/reward` (the judge phase specifically) stays **flat** at ~92–102s throughout this window while judge call *count* actually falls slightly — the growing cost is entirely in generation/log-prob/update, which scale with sequence length, not in judge serving. [confirmed]: the policy's average response length grew ~40% in just the last 10 logged steps (656→920 tokens). Whether this reflects genuine improvement (longer, more thorough answers legitimately satisfying more constraints) or length-based reward hacking is [unresolved] and out of scope for this document, but the *mechanical* driver of the timing trend is settled.

---

## 7. Root cause: why is the LLM-judge reward path so much slower than the anchor's PPL pass

Raised as a direct challenge: an LLM judge must *sample* (generate) a response, while the anchor's PPL score is a single *forward pass* over already-generated tokens — so some gap is expected, but is a 10–16× gap (§6.2) really explained by that alone, or is it substantially an implementation/config artifact?

### 7.1 Mechanistic starting point [confirmed via code + timing data cross-reference]

The anchor's live scoring (`_compute_anchor_ppl_with_ref_policy` → `_compute_ref_log_prob` → `ref_policy_wg.compute_ref_log_prob`) is architecturally identical in kind to the standard KL-loss `ref` log-prob pass every GRPO/PPO run already pays for (`actor_rollout_ref.actor.use_kl_loss=True`, all 4 variants) — a single teacher-forced forward pass over the full batch, parallelized across all 8 GPUs via FSDP + dynamic batching, no autoregressive dependency because the tokens are already known. This is consistent with the anchor run's own `reward` bucket costing only ~40s (§6.3's steps 105/106/108/109) — in the same cost class as the `ref`/`old_log_prob` buckets (~50s each) that *every* variant pays regardless of reward mechanism.

An LLM judge, by contrast, must generate its response (reasoning + `{"Score": N}`) token by token — inherently sequential per output token, and additionally requires the judge model to be woken from sleep mode, served through vLLM's OpenAI-compatible stack as ~7,500 independent HTTP requests (one per constraint-positive row), each paying its own prefill for the judge prompt (rubric + x + y, up to ~4,096 tokens; prefix caching mitigates but does not eliminate this — 27–52% hit rates observed live) with no cross-request batching of that prefill the way the anchor's single forward pass gets for free.

**This portion of the gap is fundamental** — it does not go away with better tuning, because it is intrinsic to "the judge must produce novel text" as a design choice.

### 7.2 Direct evidence: the judge fleet's own request-accounting log

Rather than reason abstractly about "generation is slower," the live vLLM engine's own periodic stats line (`Running: N reqs, Waiting: M reqs, ... generation throughput: T tok/s`) was read directly for one replica across one full reward-computation window (2026-08-18, GPU0/port 21200, `logs/verifier/gpt_oss_120b_gpu0_21200.log`):

**Phase 1 — saturated, 11:53:50 → 11:58:50 (~300s):**
```
11:53:50  Running: 64  Waiting: 853  gen throughput:  262 tok/s  KV: 10.8%
11:55:00  Running: 64  Waiting: 658  gen throughput:  781 tok/s  KV: 27.4%
11:57:00  Running: 64  Waiting: 299  gen throughput:  800 tok/s  KV: 31.0%
11:58:40  Running: 64  Waiting:  21  gen throughput:  704 tok/s  KV: 40.1%
```
`Running` sits **pegged at 64** (the configured `max_num_seqs` — see §4.4) for the *entire* 5-minute window while `Waiting` drains 853→0. This is not idle time; the replica is continuously busy at its configured concurrency ceiling.

**Phase 2 — long-tail drain, 11:58:50 → 12:00:00 (~70s):**
```
11:58:50  Running: 60  Waiting: 0
11:59:10  Running: 37
11:59:30  Running:  6
11:59:50  Running:  0
```
Once the backlog clears, concurrency collapses because a shrinking set of individual requests are still generating (long reasoning chains) while their peers already finished — the classic straggler pattern, and one this repo's own tuning notes (`llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh:270–277`) already documented for the *more bounded* Qwen3-30B-A3B judge: *"the bulk of ~3,300 judge calls finishes in ~30s and then ONE straggler holds all 8 GPUs idle for ~110s."*

**Split, this window**: ≈300s (≈81%) genuinely capacity-bound work, ≈70s (≈19%) straggler-drain. [confirmed] for this one window; presented as illustrative of the mechanism, not claimed as an exact fleet-wide average.

### 7.3 Decomposition: fundamental vs. implementation-tunable

| Factor | Fundamental or tunable | Evidence |
|---|---|---|
| Judge must autoregressively generate at all | **Fundamental** | §7.1 — no implementation choice removes this for an LLM-judge design |
| Per-request prefill (no cross-request batching of x+y) | **Fundamental**, partially mitigated | Prefix caching (default-on) gives 27–52% hit rate but the unique x+y portion always re-prefills |
| `max_num_seqs` / replica count → sets Phase-1 (§7.2) duration directly | **Tunable**, and *this task lowered it* | §4.4: 128→64 for OOM safety. Halving concurrency roughly doubles how long the backlog takes to drain, holding everything else fixed. |
| `max_tokens=8192`, `omit_max_tokens=true`, unset (default) `reasoning_effort` → sets Phase-2 tail length | **Tunable**, currently unbounded | Nothing in the current config caps how long a single straggler can run; Qwen3-30B-A3B's script caps `max_tokens=1024` and documents exactly the tail-shortening effect of doing so. |
| Client-side dispatch concurrency | Checked, not a bottleneck | `RewardLoopWorker.compute_score_batch` (`verl/experimental/reward_loop/reward_loop.py:142–147`) fires every assigned row as a concurrent `asyncio.gather` task with no additional semaphore — the server's `max_num_seqs`, not the client, is the throttle. |

**Verdict**: the user's intuition was correct that this is "not fundamental, close to an implementation issue" — for the *magnitude* of the gap, though not for its *existence*. Generation-vs-forward-pass is a real, irreducible-at-the-margin cost. But the specific 10–16× figure in §6.2 is inflated well past that irreducible floor by two config knobs that are both currently set in the more-expensive direction: a concurrency ceiling this very task cut in half for an unrelated reason (OOM safety), and an output-length cap that was never set at all.

### 7.4 Open question: the 4-vs-8-replica discrepancy [unresolved]

§5.2/§6.2 noted the original `ab6ix6bq` run used only 4 gpt-oss-120b replicas (ports stride-10: `22900/22910/22920/22930`) against this continuation's 8. Two facts limit how far this can be chased: (a) the server-side concurrency settings (`max_num_seqs`, `gpu_memory_utilization`, `enforce_eager`) for that historical run are CLI flags to a separately-launched `vllm serve` process, not part of the Hydra config wandb captured, and no verifier server logs from that run survive on disk; (b) `ab6ix6bq`'s `wandb-metadata.json` records `git.remote: https://github.com/ssangyeon/if-rlvr-rlvr.git` at commit `a3033f0543b5d2d7369dddb0da1950cb6371de9b` — a different fork/checkout than the one this document and the continuation scripts live in, so the actual launch script that produced it is not recoverable from this repo at all. Given §7.2–7.3's finding that the dominant cost is genuinely capacity-bound decode/prefill volume rather than topology per se, and that this continuation's 8×TP1 setup lands within ~8% of the original's per-step pace anyway (§6.2's `h02owfkp` comparison), this discrepancy is flagged for awareness but does not appear to be the dominant lever — the concurrency *ceiling value* (§7.3) matters more than whether it's split across 4 or 8 replicas to reach it.

---

## 8. Recommendations

For any future gpt-oss-120b-judge (or any long-generation LLM-judge) run on this codebase:

1. **Bound the judge's own output length explicitly.** Set `IF_LLM_VERIFIER_MAX_TOKENS` to something in the low hundreds and/or set `IF_LLM_VERIFIER_REASONING_EFFORT=low` rather than leaving it unset. This directly targets §7.2's Phase-2 tail, which is pure config, not physics. Qwen3-30B-A3B's own script already does exactly this (`max_tokens=1024`) and its tail is correspondingly shorter.
2. **Treat `max_num_seqs`/`max_num_batched_tokens` as a throughput dial, not just an OOM-safety dial**, and re-tune upward if headroom allows once §4.4's workspace-allocation risk is otherwise mitigated (e.g., if a future vLLM/Triton release fixes the out-of-pool allocation, or if the trainer-side resident footprint is independently reduced).
3. **Pre-warm any lazily-populated, content-addressed shared cache** before fanning out N concurrent replicas of anything (§4.3's fix generalizes beyond `openai_harmony`).
4. **Never trust `df` alone for cache-placement safety checks** on a network-mounted `HF_HOME` without confirming there isn't a lower per-tenant quota underneath the reported filesystem-wide capacity (§4.2).
5. **Commit the actual launch script for any run that produces a checkpoint intended for reuse.** Both the anchor's `validate_upload_tulu3_anchor_cache.py` (referenced, never committed) and `ab6ix6bq`'s true launch command (lived only in a collaborator's uncommitted fork checkout) were unrecoverable during this investigation, which materially slowed root-causing §5.2 and §7.4.

---

## 9. Appendix

### 9.1 Files added/modified for this task

| File | Change |
|---|---|
| `if_rlvr/exps/bidirectional/push_checkpoints_to_hf.py` | added `--step-offset` (default 0) |
| `if_rlvr/exps/bidirectional/download_hf_checkpoint_step.py` | new |
| `if_rlvr/exps/bidirectional/llama31_tulu3_8b_dpo_constraint_only_nonreason.sh` | new |
| `if_rlvr/exps/bidirectional/llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh` | new |
| `if_rlvr/exps/bidirectional/run_tulu3_8b_dpo_epoch5_6_continuation.sh` | new (orchestrator) |
| `verl/workers/engine_workers.py` | `ActorRolloutRefWorker.init_model`: added `IF_REF_POLICY_MODEL_PATH_OVERRIDE` (§9.2) |

### 9.2 `IF_REF_POLICY_MODEL_PATH_OVERRIDE` — exact mechanism

`HFModelConfig.__post_init__` (`verl/workers/config/model.py:148`) eagerly resolves `local_path`/`hf_config`/`tokenizer`/`generation_config` from `.path` at *construction* time — mutating `.path` after construction (even where the field's own `_mutable_fields` allowlist would permit it) would leave those derived fields stale. The fix therefore does not mutate the existing `model_config`; when the env var is set, it deep-copies the *OmegaConf node* (`self.config.model`), overwrites `.path` on the copy, and re-runs `omega_conf_to_dataclass` on it — a fresh construction that re-triggers `__post_init__` correctly. Threaded to the Ray worker via `+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_POLICY_MODEL_PATH_OVERRIDE=...` rather than relying on ambient OS-environment inheritance into the Ray actor, matching this codebase's own established pattern for exactly this concern (c.f. `IF_APPLY_ENABLE_THINKING_KWARG`). Default (unset) behavior is byte-identical to pre-change code — verified by `git diff` showing the original `deepcopy(model_config)` line preserved unconditionally in the `else` branch.

### 9.3 Wandb run-ID registry (project `ifif/verl_if_rlvr`)

Used in this document:

| Run ID | Name (experiment) | State | Steps | Runtime (h) |
|---|---|---|---|---|
| `1gyg97hj` | `llama31_tulu3_8b_dpo_grpo_nonthink_constraint_only_b1024_c1_t1_2k` | crashed | 362 | 16.41 |
| `myf68cle` | `llama31_tulu3_8b_dpo_grpo_nonthink_anchor_pyx01_b1024_c1_t1_2k` | crashed | 362 | 38.16 |
| `ab6ix6bq` | `llama31_tulu3_8b_dpo_grpo_nonthink_llmverifier_gptoss120b_bonus01_threshold5_b1024_c1_t1_2k` | crashed | 362 | 64.54 |
| `1n2avbdq` | `llama31_tulu3_8b_dpo_grpo_nonthink_llmverifier_qwen3_30ba3b_nonthink_bonus01_threshold7_b1024_c1_t1_2k` | **finished** | 364 | 38.13 |
| `h7ni08i8` | `tulu3_gptoss120b_judge_t5_epoch5to6` (this task, attempt 1 — killed after §4.1) | finished (1 step only) | 1 | 0.31 |
| `h02owfkp` | `tulu3_gptoss120b_judge_t5_epoch5to6` (this task, current) | running | 69+ | 13.35+ |

Found but explicitly **not** used as a like-for-like comparison, for the reasons noted:

| Run ID | Name | Why excluded |
|---|---|---|
| `5ziiw4bk` | `..._anchor_pyx01_llmverifier_gptoss120b_fallback_bonus01_threshold5_..._t1_2k` | anchor+judge **hybrid** variant, not the pure judge-only or pure anchor arm this document compares |
| `ncfp7isg` | (name does not contain `tulu3`) | matched an unfiltered `"gptoss120b"` name search but is very likely a different policy-model experiment (this repo trains multiple base models under the same project) — flagged here specifically to correct an earlier in-conversation misattribution of this run as the source of the Tulu3 judge-only HF checkpoint |
| `wmib554l`, `41p6737u`, `f7pgrwh8` | judge-only / hybrid, various | crashed at launch, 0 logged steps |
| `crmnqbr8`, `0hhym6yt`, `s51cluxd` | anchor, early attempts | crashed at launch, 0 logged steps, predecessors of `myf68cle` |
| `llbcrjix`, `web2mb2j`, `wi5t08yu`, `pi4noqf2` | judge-only (gptoss120b), various | earlier crashed attempts predating `ab6ix6bq`, included in the original terminal-session survey but not separately analyzed here |
| `eruch8nf`, `i3rrrynl`, `9hh4ho36`, `u96p26zr`, `9dvsnw5d` | anchor+qwen30ba3b-fallback hybrid, early attempts | crashed within the first few steps, predecessors of `1n2avbdq` (which switched to the pure-judge naming once the hybrid attempts kept failing) |

### 9.4 Exact CUDA OOM traceback (§4.4), for grep-ability

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 900.00 MiB.
GPU 0 has a total capacity of 79.19 GiB of which 679.38 MiB is free.
Including non-PyTorch memory, this process has 72.66 GiB memory in use.
Process 42728 has 3.66 GiB memory in use. Process 49892 has 2.02 GiB memory in use.
Of the allocated memory 71.60 GiB is allocated by PyTorch, with 2.98 GiB allocated
in private pools (e.g., CUDA Graphs), and 1.35 GiB is reserved by PyTorch but unallocated.
  File ".../vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py", line 157,
    in triton_kernel_fused_experts
  File ".../triton_kernels/matmul_ogs.py", line 340, in apply_allocation
    output = torch.empty(allocation.output[0], device=allocation.device, dtype=allocation.output[1])
```
