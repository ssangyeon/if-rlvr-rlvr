# IF-RLVR reward cost benchmark

This benchmark reuses one exact training-step corpus (1,024 prompts x 8
rollouts = 8,192 responses) to compare:

1. `Qwen/Qwen3-4B` final-answer `p(y|x)` PPL reward
2. `openai/gpt-oss-120b` verifier
3. `Qwen/Qwen3-30B-A3B` non-reasoning verifier
4. `Qwen/Qwen3-4B` non-reasoning verifier

Each model runs as four independent vLLM replicas, one per B200 (`TP=1`,
effective `DP=4`). Model loading and one warmup request per replica are excluded
from the measured reward-inference wall time.

This is a controlled **vLLM capacity/cost comparison**: all four methods use
persistent HTTP sessions and the same per-GPU request concurrency. The PPL
request replays the exact `x + y` token IDs with vLLM prompt log-probabilities;
its wall time is therefore not the wall time of VERL's in-training FSDP
`_compute_ref_log_prob` implementation. Logical FLOPs are comparable, while the
backend timing answers the requested TP=1, DP=4 serving question.

## 1. Save one exact rollout step

The regular `trainer.rollout_data_dir` JSONL loses `ppl_prompt` and exact token
IDs. Use VERL rollout-skip dumping for the step to benchmark:

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/IFIF/if-rlvr

export PATH=/NHNHOME/WORKSPACE/26msit001_A/IFIF/.miniforge3/envs/verl/bin:$PATH
PYTHONPATH=$PWD \
MODEL_PATH=Qwen/Qwen3-4B \
ENABLE_THINKING=true \
TRAIN_BATCH_SIZE=1024 \
PPO_MINI_BATCH_SIZE=1024 \
ROLLOUT_N=8 \
PY_GIVEN_X_REWARD_COEFF=0 \
PX_GIVEN_Y_REWARD_COEFF=0 \
IF_REF_ANCHOR_PRECOMPUTE=false \
IF_REF_POLICY_ANCHOR_PPL=false \
IF_MAX_RETRIES=1 \
bash if_rlvr/exps/bidirectional/qwen3_4b_01_00_const1_ref_anchor_reasoning_workspace.sh \
  trainer.total_training_steps=1 \
  trainer.save_freq=0 \
  trainer.test_freq=0 \
  trainer.logger='["console"]' \
  actor_rollout_ref.rollout.skip.enable=true \
  actor_rollout_ref.rollout.skip.dump_dir=/NHNHOME/WORKSPACE/26msit001_A/IFIF/reward_cost_rollouts \
  actor_rollout_ref.rollout.skip.max_dump_step=1 \
  actor_rollout_ref.rollout.skip.action=cache
```

The dump is placed below a generated directory like:

```text
/NHNHOME/WORKSPACE/26msit001_A/IFIF/reward_cost_rollouts/
  <experiment>_<project>/GBS1024_N8_in..._out.../genstep_000001/
    new_batch.dp
    gen_batch.dp
```

If the run already produced such a dump, do not generate it again.

If the trainer exits nonzero after both `.dp` files were written (for example,
a validation/DataLoader teardown failure), resume directly from the dump:

```bash
GPU_IDS=0,1,2,3 \
JUDGE_MODE=all \
VLLM_PREFIX_CACHING=disabled \
MAX_RETRIES=0 \
LOCAL_FILES_ONLY=1 \
bash if_rlvr/benchmarks/reward_cost/resume_reward_cost_after_rollout.sh \
  "$STEP_DIR" "$CORPUS" "$OUTPUT_ROOT"
```

The resume script validates `new_batch.dp` and `gen_batch.dp`, atomically
creates the corpus if absent, and skips corpus regeneration when an existing
8,192-row corpus validates successfully.

## 2. Export a compact shared corpus

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/IFIF/if-rlvr

PYTHON=/NHNHOME/WORKSPACE/26msit001_A/IFIF/.miniforge3/envs/verl/bin/python
export NLTK_DATA=/NHNHOME/WORKSPACE/26msit001_A/IFIF/IFBench/.nltk_data
STEP_DIR=/NHNHOME/WORKSPACE/26msit001_A/IFIF/reward_cost_rollouts/<...>/genstep_000001
CORPUS=/NHNHOME/WORKSPACE/26msit001_A/IFIF/reward_cost_rollouts/qwen3_4b_reasoning_step1_8192.jsonl

PYTHONPATH=$PWD "$PYTHON" if_rlvr/benchmarks/reward_cost/prepare_rollout_step.py \
  "$STEP_DIR" \
  --output "$CORPUS" \
  --tokenizer Qwen/Qwen3-4B \
  --nltk-data "$NLTK_DATA" \
  --expected-responses 8192 \
  --expected-prompts 1024 \
  --rollouts-per-prompt 8 \
  --require-think-end
```

The exporter prioritizes the exact `ppl_x_prompt_ids` and
`ppl_y_final_answer_ids` saved by the agent loop. It stores:

- the full decoded reasoning response;
- the final answer after `</think>` for all judges (using the training fallback
  when the marker is absent);
- constraint-free `x` and final-answer `y` token IDs for PPL;
- the training IFEval constraint score for optional verifier gating.

Empty/aborted responses stay in the 8,192-row corpus. They are marked
`ppl_eligible=false` and incur no PPL model request, matching the current
ref-policy scoring batch, which filters empty prefix/continuation pairs.

## 3. Run all four measurements

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/IFIF/if-rlvr

GPU_IDS=0,1,2,3 \
JUDGE_MODE=all \
bash if_rlvr/benchmarks/reward_cost/run_reward_cost_benchmark.sh \
  /NHNHOME/WORKSPACE/26msit001_A/IFIF/reward_cost_rollouts/qwen3_4b_reasoning_step1_8192.jsonl \
  /NHNHOME/WORKSPACE/26msit001_A/IFIF/reward_cost_results/qwen3_4b_reasoning_step1
```

`JUDGE_MODE=all` forces all 8,192 responses through every verifier and is the
clean model-cost comparison requested here. To reproduce the current training
manager's **constraint-positive call selection** instead, use:

```bash
JUDGE_MODE=constraint-positive \
GPU_IDS=0,1,2,3 \
bash if_rlvr/benchmarks/reward_cost/run_reward_cost_benchmark.sh "$CORPUS" "$OUTPUT_ROOT"
```

That mode calls an LLM verifier only when the rule-based constraint score is
positive. With a reasoning actor, `--require-think-end` also makes responses
without a completed `</think>` block fail the gate, matching training. Client
session/concurrency and the standardized vLLM backend still follow this
benchmark harness, not the trainer's exact Python overhead.

Useful overrides include:

```bash
LOCAL_FILES_ONLY=1                 # fail instead of downloading an absent model
PORT_BASE=28600                    # four consecutive ports per model
CONCURRENCY_PER_GPU=128
REQUEST_TIMEOUT=600
MAX_RETRIES=0                       # keep timed FLOPs and attempts one-to-one
VLLM_MAX_MODEL_LEN=40960
VLLM_PREFIX_CACHING=disabled        # controlled default
RUN_PPL=1
RUN_GPT_OSS=1
RUN_QWEN30=1
RUN_QWEN4_VERIFIER=1
```

The launcher disables vLLM prefix caching. This is intentional: the eight
responses for each prompt otherwise make cache hit order and request routing a
large confounder. PPL prompt-logprob requests are full-sequence scoring requests
in either case. Set `VLLM_PREFIX_CACHING=enabled` for a second, training-like
cache-on measurement; the report uses API `cached_tokens` when adjusting its
logical FLOPs estimate.

Qwen3-4B PPL and Qwen3-4B verifier measurements use separately started replica
sets. That keeps one method from pre-warming the other's server state; startup
and model-loading time remain excluded from both measurements.

## Outputs

Each method writes `results.jsonl` and `metrics.json`. The combined files are:

```text
<output>/report/report.json
<output>/report/report.csv
<output>/report/report.md
```

The main timing columns are:

- `4-GPU wall`: elapsed time after warmup until all selected requests finish;
- `1-GPU eq replica-sum`: sum of each replica's first-request-to-last-response
  makespan; this includes server queue and HTTP/CPU overhead;
- `1-GPU eq allocated`: `4 x 4-GPU wall`, including tail idle;
- `replica span / allocation`: replica-sum divided by allocated time.

The requested 1-GPU-normalized number is `1-GPU eq replica-sum`. It is a
projected serial cost, not measured GPU kernel-active time and not a direct
one-GPU rerun: single-GPU batching and queueing can change wall time.

The launcher disables retries by default. A retry would add elapsed inference
work that cannot be reconstructed exactly from only the final OpenAI response;
any nonzero retry count is therefore displayed prominently in the report.

## FLOPs definition

FLOPs are analytical logical matrix-multiplication FLOPs with one fused
multiply-add counted as two operations. Every one of the 8,192 samples is
estimated separately before aggregation; an average length is never substituted
into the quadratic attention term.

Included:

- Q/K/V/O projections;
- dense SwiGLU or active top-k MoE experts plus router;
- causal attention score and value products;
- language-model head positions actually needed by PPL/generation.

Excluded:

- norm, RoPE, activations, softmax, top-k and sampling;
- memory traffic, HTTP/CPU overhead and CUDA graph/kernel padding;
- output-head work that vLLM may compute and discard at intermediate
  chunked-prefill boundaries;
- inactive MoE experts.

For GPT-OSS-120B, the estimate uses 18 full-attention and 18 sliding-window
layers (window 128) and four active experts. For Qwen3-30B-A3B it uses eight
active experts. GPT-OSS expert quantization changes runtime and bandwidth, not
the logical operation count. The PPL result covers only the per-step policy
`p(y|x)` pass; precomputed anchor construction/cache-miss passes are excluded.

Installed vLLM 0.11 does not expose a separate GPT-OSS reasoning-token count in
`usage`; its `completion_tokens` still includes both analysis and final-output
generation, so total time and FLOPs are unaffected.
