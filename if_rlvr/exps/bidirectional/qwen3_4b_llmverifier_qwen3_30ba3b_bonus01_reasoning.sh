#!/usr/bin/env bash
# GRPO | Qwen/Qwen3-4B reasoning policy
#   + external Qwen/Qwen3-30B-A3B non-thinking LLM-verifier bonus at score >= 7
# JUDGE-ONLY ARM: constraint reward + judge bonus. NO anchor shaping.
# sized for ONE node of 8x H100 80GB.
#
# ---------------------------------------------------------------------------
# Per-rollout reward
# ---------------------------------------------------------------------------
#   1. policy rollout on GPUs 0-7 (vLLM, TP=1 -> 8 data-parallel replicas)
#   2. IFEval constraint reward; rows with constraint=0 stop here
#   3. every constraint-positive row gets ONE Qwen3-30B-A3B judge call; a score
#      >= IF_LLM_VERIFIER_THRESHOLD adds +IF_LLM_VERIFIER_BONUS
#   That is the whole reward. Max per rollout is 1.1, never 1.2 - there is NO
#   anchor shaping in this arm; see IF_REF_ANCHOR_PRECOMPUTE below. The anchor
#   cache is still referenced, but only to filter the training split to the
#   same 93,882 rows the anchor arms use, so the arms stay comparable.
#
# ---------------------------------------------------------------------------
# GPU plan - trainer and judge time-share all 8 GPUs, they are never resident
# together
# ---------------------------------------------------------------------------
#   * 4 judge replicas (Qwen3-30B-A3B at TP=2, one NVLink-local pair each) are
#     started first on empty GPUs and put to sleep (level 1: weights spill to
#     host RAM, KV cache is freed) before the trainer profiles its memory.
#   * Training then owns the GPUs. verl sleeps the policy rollout engines
#     immediately after generate_sequences(), and RewardLoopManager wakes the
#     judge only for the reward phase and sleeps it again in a `finally` block.
#   * TP=2 rather than TP=1 is the load-bearing choice here. Qwen3-30B-A3B is
#     56.8 GiB of bf16 weights. At TP=1 every GPU holds a full copy, which both
#     leaves almost no KV next to the trainer's residual AND makes the level-1
#     sleep/wake move 56.8 GiB per GPU twice per step. TP=2 halves both: 28.4
#     GiB of weights per GPU and half the per-step host<->device traffic, on a
#     pair that talks over NVLink. Aggregate judge FLOPs are unchanged - all 8
#     GPUs still run the judge, just as 4 replicas instead of 8.
#   * IF_LLM_VERIFIER_SLEEP_LEVEL=2 is NOT a valid way to cut the transfer cost:
#     level 2 discards the weights and expects an external weight update on
#     wake, which a standalone judge server cannot provide.
#
# Per-GPU budget against the card's 81,559 MiB:
#   judge awake : 0.60 * 79.6 GiB = 47.8 GiB (28.4 weights + ~2 overhead +
#                 ~17 GiB KV, i.e. ~370k tokens per replica - the measured judge
#                 prompt is <4k tokens, so this is deliberately over-provisioned)
#   trainer at judge-wake time: rollout engine asleep at level 2 (~4 GiB) plus
#                 the resident fp32 FSDP shard (~3 GiB at 4B over 8 ranks).
#                 ~7 GiB, leaving >20 GiB of slack.
#   judge asleep: 2.57 GiB of CUDA context per GPU that never goes away. This is
#                 a permanent tax on the POLICY engine's budget - it is why
#                 ROLLOUT_GPU_MEM_UTIL here is lower than in sibling runs that
#                 have no co-resident judge. See that knob for the full ledger.
#   Because that slack exists, actor param/optimizer offload stays OFF: it only
#   moves memory, but it pays a host round-trip on every log-prob pass and on
#   the update. Raise IF_LLM_VERIFIER_GPU_MEM_UTIL or ROLLOUT_GPU_MEM_UTIL and
#   you have to turn the two ACTOR_*_OFFLOAD knobs back on.
#
# ---------------------------------------------------------------------------
# Judge sampling: Qwen3 non-thinking mode, per the model card
# ---------------------------------------------------------------------------
#   https://huggingface.co/Qwen/Qwen3-30B-A3B -> "Best Practices".  Qwen3 is a
#   hybrid-reasoning model; enable_thinking=false makes the chat template emit a
#   closed, empty <think></think> block so the reply is just the {"Score": N}
#   object. The card's non-thinking preset is temperature 0.7, top_p 0.8,
#   top_k 20, min_p 0, and it explicitly says "DO NOT use greedy decoding, as it
#   can lead to performance degradation and endless repetitions" - so this run
#   does NOT judge at temperature 0. presence_penalty stays 0 (the card offers
#   0-2 only as a repetition escape hatch, and warns it can cause language
#   mixing).
#
# Usage:  bash qwen3_4b_llmverifier_qwen3_30ba3b_bonus01_reasoning.sh
# Any extra arguments are forwarded verbatim as Hydra overrides.

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
mkdir -p "${CACHE_ROOT}"
cd "${VERL_DIR}"

########################### model / cache placement ###########################
# The ambient HF_HOME on this box points at the quota-capped /workspace volume
# (~50 GB), which silently fails mid-download once Qwen3-30B-A3B (57 GB) plus
# Qwen3-4B (8 GB) are in flight. Pin the cache to the repo volume instead.
export HF_HOME=${HF_HOME_OVERRIDE:-${CACHE_ROOT}/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
# The base launcher re-derives NLTK_DATA from NLTK_DATA_DIR, so both must be set
# or its default (<repo>/../IFBench/.nltk_data) silently wins.
export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${CACHE_ROOT}/nltk_data}
export NLTK_DATA=${NLTK_DATA:-${NLTK_DATA_DIR}}
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${NLTK_DATA_DIR}"

IF_HF_CACHE_MIN_GB=${IF_HF_CACHE_MIN_GB:-120}
free_gb() {
    local dir=$1
    while [[ ! -d "${dir}" && "${dir}" != "/" ]]; do dir=$(dirname -- "${dir}"); done
    df -Pk "${dir}" 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}' | grep -E '^[0-9]+$' || echo 0
}
if [[ "$(free_gb "${HF_HOME}")" -lt "${IF_HF_CACHE_MIN_GB}" ]]; then
    echo "ERROR: ${HF_HOME} has $(free_gb "${HF_HOME}") GB free (< ${IF_HF_CACHE_MIN_GB} GB)." >&2
    echo "       It must hold Qwen3-4B (8 GB) + Qwen3-30B-A3B (57 GB) and the checkpoints." >&2
    exit 1
fi

########################### run isolation / ports ###########################
export RUN_SLOT=${RUN_SLOT:-0}
# 25000, not the 20000+2000*slot the other launchers use, and not 40000: this
# box's ephemeral range is 32768-60999 (/proc/sys/net/ipv4/ip_local_port_range)
# and the run starts several hundred Ray processes that each bind a random
# ephemeral port. _reserve_port_block() needs VLLM_RESERVED_PORT_COUNT
# consecutive free ports inside each replica's VLLM_PORT_STRIDE window, so a
# base inside the ephemeral range is a dice roll at every engine start.
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-$((25000 + RUN_SLOT * 2000))}
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-$((IF_RLVR_PORT_BASE + 200))}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

# Keep per-worker native thread pools small; Ray already creates many workers.
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

########################### runtime preflight ###########################
export IF_LLM_VERIFIER_PYTHON=${IF_LLM_VERIFIER_PYTHON:-${CONDA_PREFIX:-/root/miniconda3/envs/verl}/bin/python}
if [[ ! -x "${IF_LLM_VERIFIER_PYTHON}" ]]; then
    echo "ERROR: verifier Python is not executable: ${IF_LLM_VERIFIER_PYTHON}" >&2
    exit 1
fi
# Every engine in this run - 8 policy replicas and 4 judge replicas - imports
# the v1 worker stack, which pulls in numba. vLLM 0.11 pins numba==0.61.2, which
# requires numpy<2.3; a mismatch kills each engine minutes after launch with the
# real cause buried in a dozen separate logs. Surface it here in ten seconds.
# The IFEval verifier's own deps (nltk data, langdetect) are checked too, since
# IFLLMVerifierRewardManager raises on every reward-loop worker without them.
PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${IF_LLM_VERIFIER_PYTHON}" - <<'PY'
import os
import sys

try:
    from vllm.v1.worker.gpu_worker import Worker  # noqa: F401
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"[preflight] vLLM engine worker stack is not importable: {type(exc).__name__}: {exc}\n"
        "            Every vLLM engine in this run would fail the same way "
        "(vLLM 0.11 pins numba==0.61.2, which requires numpy<2.3)."
    )

sys.path.insert(0, os.path.join(os.environ["VERL_DIR"], "if_rlvr"))
try:
    import langdetect  # noqa: F401

    from ifeval_oi import instructions_util

    instructions_util.nltk.word_tokenize("This is a sentence. Here is another one.")
    instructions_util.count_words("two words")
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"[preflight] IFEval reward dependencies are broken: {type(exc).__name__}: {exc}\n"
        "            Install `langdetect immutabledict nltk` and the nltk punkt/punkt_tab "
        f"data under NLTK_DATA={os.environ.get('NLTK_DATA')}."
    )
print("[preflight] vLLM worker stack and IFEval reward dependencies OK")
PY

########################### verifier servers ###########################
export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-Qwen/Qwen3-30B-A3B}
export IF_LLM_VERIFIER_REVISION=${IF_LLM_VERIFIER_REVISION:-ad44e777bcd18fa416d9da3bd8f70d33ebb85d39}
# Qwen3 is a hybrid-reasoning model; judge in non-thinking mode so the reply is
# just the {"Score": N} JSON object.
export IF_LLM_VERIFIER_ENABLE_THINKING=${IF_LLM_VERIFIER_ENABLE_THINKING:-false}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-0,1,2,3,4,5,6,7}
export IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP:-2}
export IF_LLM_VERIFIER_HOST=${IF_LLM_VERIFIER_HOST:-127.0.0.1}
# The policy's own vLLM replicas reserve VLLM_MASTER_PORT_BASE + rank * 100,
# i.e. base+200 .. base+999 for 8 replicas. Start the judge above that block.
export IF_LLM_VERIFIER_PORT=${IF_LLM_VERIFIER_PORT:-$((IF_RLVR_PORT_BASE + 1200))}
export IF_LLM_VERIFIER_START_SERVER=${IF_LLM_VERIFIER_START_SERVER:-true}
export IF_LLM_VERIFIER_LOG_DIR=${IF_LLM_VERIFIER_LOG_DIR:-${VERL_DIR}/logs/verifier}

# Wake/sleep alternation with the trainer.
export IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE:-true}
export IF_LLM_VERIFIER_MANAGE_SLEEP=${IF_LLM_VERIFIER_MANAGE_SLEEP:-true}
export IF_LLM_VERIFIER_SLEEP_LEVEL=${IF_LLM_VERIFIER_SLEEP_LEVEL:-1}
export IF_LLM_VERIFIER_DEV_MODE=${IF_LLM_VERIFIER_DEV_MODE:-1}
export IF_LLM_VERIFIER_CONTROL_TIMEOUT=${IF_LLM_VERIFIER_CONTROL_TIMEOUT:-600}
export IF_LLM_VERIFIER_WAIT_TIMEOUT=${IF_LLM_VERIFIER_WAIT_TIMEOUT:-2400}

# Engine sizing - see the per-GPU budget in the header.
export IF_LLM_VERIFIER_DTYPE=${IF_LLM_VERIFIER_DTYPE:-bfloat16}
export IF_LLM_VERIFIER_GPU_MEM_UTIL=${IF_LLM_VERIFIER_GPU_MEM_UTIL:-0.60}
# Judge prompt = rubric (~250 tok) + x (<=2048 tok) + the rollout's final answer
# with the reasoning section stripped (<=8192 tok if stripping fails and the raw
# response is used). 16384 covers the worst case with room to spare.
export IF_LLM_VERIFIER_MAX_MODEL_LEN=${IF_LLM_VERIFIER_MAX_MODEL_LEN:-16384}
export IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS=${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS:-16384}
export IF_LLM_VERIFIER_MAX_NUM_SEQS=${IF_LLM_VERIFIER_MAX_NUM_SEQS:-128}
# The judge decodes ~15 tokens, so CUDA graphs buy nothing and cost both capture
# time (x4 replicas) and the memory the KV cache wants.
export IF_LLM_VERIFIER_ENFORCE_EAGER=${IF_LLM_VERIFIER_ENFORCE_EAGER:-true}
export IF_LLM_VERIFIER_TRUST_REMOTE_CODE=${IF_LLM_VERIFIER_TRUST_REMOTE_CODE:-false}
# vLLM's default TP all-reduce path rendezvouses through PyTorch symmetric
# memory, which needs CUDA multicast/VMM privileges a container may not grant;
# TP>1 then dies in torch_symm_mem.rendezvous. Falling back to NCCL costs
# nothing on an NVLink pair, and is a no-op at TP=1.
export IF_LLM_VERIFIER_USE_SYMM_MEM=${IF_LLM_VERIFIER_USE_SYMM_MEM:-0}

# Reward-side judging policy.
export IF_LLM_VERIFIER_BONUS=${IF_LLM_VERIFIER_BONUS:-0.1}
export IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-5.3}
# Qwen3 non-thinking preset from the model card; see the header.
export IF_LLM_VERIFIER_TEMPERATURE=${IF_LLM_VERIFIER_TEMPERATURE:-0.7}
export IF_LLM_VERIFIER_TOP_P=${IF_LLM_VERIFIER_TOP_P:-0.8}
export IF_LLM_VERIFIER_TOP_K=${IF_LLM_VERIFIER_TOP_K:-20}
export IF_LLM_VERIFIER_MIN_P=${IF_LLM_VERIFIER_MIN_P:-0}
# A well-formed answer is `{"Score": N}` - about 8 tokens under the json_object
# response format. A large cap does not make good answers longer, it only lets
# the handful of rambling rows per step run to the cap, and a lone request
# decodes slowly enough that ONE straggler can hold every GPU idle for minutes.
# 128 bounds that tail and is still >10x what a valid answer needs;
# extract_judge_score also has a regex fallback for truncated JSON.
export IF_LLM_VERIFIER_MAX_TOKENS=${IF_LLM_VERIFIER_MAX_TOKENS:-128}
export IF_LLM_VERIFIER_OMIT_MAX_TOKENS=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS:-false}
export IF_LLM_VERIFIER_RESPONSE_FORMAT=${IF_LLM_VERIFIER_RESPONSE_FORMAT:-true}
export IF_LLM_VERIFIER_REASONING_EFFORT=${IF_LLM_VERIFIER_REASONING_EFFORT:-}
# The whole reward phase is issued at once, so a request can wait behind several
# thousand others; time out generously rather than silently dropping bonuses.
export IF_LLM_VERIFIER_TIMEOUT=${IF_LLM_VERIFIER_TIMEOUT:-900}
export IF_LLM_VERIFIER_MAX_RETRIES=${IF_LLM_VERIFIER_MAX_RETRIES:-2}
export IF_LLM_VERIFIER_REWARD_WORKERS=${IF_LLM_VERIFIER_REWARD_WORKERS:-64}

########################### actor training ###########################
# Whole-node run: 8 policy replicas at TP=1, no rollout TP collectives.
export GPU_SET=${GPU_SET:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export ROLLOUT_TP=${ROLLOUT_TP:-1}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-Qwen/Qwen3-4B}
export ENABLE_THINKING=${ENABLE_THINKING:-true}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-true}

# Optimization - unchanged from the 4-GPU definition of this experiment. The
# global batch is 1024 prompts x n=8; FSDP just shards it 8 ways instead of 4
# and token-mean aggregation normalizes by a globally all-reduced token count,
# so moving to 8 GPUs does not change the objective.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
# 93,882 cached rows / 1024 = 91 steps per epoch -> one checkpoint per epoch,
# and the last one lands exactly on the final step (91 / 182 / 273).
export SAVE_FREQ=${SAVE_FREQ:-91}
export TEST_FREQ=${TEST_FREQ:-1000}   # base-launcher default: no in-training validation

# Throughput knobs. None of these change the optimization.
#   PPO_MAX_TOKEN_LEN_PER_GPU: the base launcher's 98304 is a B200 profile. At
#   Qwen3's 151,936-token vocab a microbatch costs ~0.283 MiB/token in bf16
#   logits AND the same again in the logits gradient - a model-size-independent
#   term - so 98304 tokens is ~60 GB of logits alone. 32768 is the value
#   measured safe at 4B on this box.
#   LOG_PROB_MAX_TOKEN_LEN_PER_GPU: 3x the actor cap, not the usual "same". The
#   old_log_prob / ref / anchor-PPL passes are forward-only under no_grad and use
#   flash-attn's fused cross-entropy, so they never materialize a full-vocab fp32
#   tensor nor retain activations - the step peak comes from update_actor's
#   backward, not from these. They are overhead-bound, so cutting the pass count
#   is close to free; they were 239 s of the measured 1325 s step at 65536.
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-98304}
#   The binding constraint is the FSDP -> vLLM weight sync. It is only dangerous
#   when it runs while the engine is AWAKE, which happens after a _validate()
#   call or the anchor-precompute block. This arm has neither (PRECOMPUTE=false,
#   TEST_FREQ=1000), so every sync lands with the engine asleep at level 2:
#     worker gather ~10.5 + sleeping engine ~4 + sleeping judge 2.57 = ~17 GiB.
#   That frees the rollout phase to be the constraint instead:
#     engine awake @ 0.85 ... 67.3 GiB
#     worker resident ....... ~3.0 GiB (Adam is offloaded, see below)
#     sleeping judge ctx .... ~2.6 GiB
#     -------------------------------------
#     ~72.9 of 79.18 GiB, matching the 75 GB/GPU measured at 0.85 on this box
#     before the offload was added. 0.90 would measure ~81 of 81.6 - no margin
#     for the CUDA-graph capture spike, and the judge tax is on top of that.
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
#   verl leaves rollout.max_model_len null, so vLLM falls back to Qwen3-4B's
#   max_position_embeddings (40960) - 4x the block-table width this run can use.
export ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
#   Qwen3-4B KV is 2 x 36 layers x 8 kv-heads x 128 head-dim x 2 B = 144 KiB per
#   token. At util 0.85 that is 67.3 - 7.5 (bf16 weights) - ~4 (engine overhead)
#   = ~55.8 GiB ~= 397k tokens of KV per replica. Measured mean sequence is 225
#   prompt + 2519 response = ~2.7k tokens, so ~150 fit; 160 is KV-matched.
#   Raising it further does NOT raise steady-state throughput - decode here is
#   KV-bandwidth-bound, not weight-bound (at 128 concurrent x 3k ctx the engine
#   reads ~56 GB of KV against 8 GB of weights per forward), so bigger batches
#   buy <10%. What the extra slots do buy is a shorter tail: vLLM schedules
#   FCFS, and with 1024 requests per replica a long rollout queued late is what
#   stretches the step. Oversubscribing past KV capacity instead triggers
#   recompute-preemption, which is strictly wasted work.
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-160}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
#   Each AgentLoopWorker is a Ray actor driving its chunk of the 1024 prompts;
#   this box has 224 cores.
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-256}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-32}

# Host residency for the actor.
#   PARAM_OFFLOAD stays off: the parameters are touched by every forward and
#   backward, so offloading them buys ~2 GiB for three host round-trips a step.
#   OPTIMIZER_OFFLOAD is ON: the fp32 Adam moments are ~4 GiB per GPU and are
#   touched exactly once per step inside optimizer.step(), so one round-trip at
#   ~10-20 GB/s is under a second against a multi-minute step. It matters
#   because the weight-sync budget above was measured BEFORE the Adam state
#   existed; without this the worker is ~4 GiB fatter at the step-91 validation
#   resync than it was at the step-0 one.
export ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
export ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
#   The weight sync stages through one contiguous GPU buffer of this size (verl
#   default 2048 MB). It only has to clear the largest single tensor - the
#   151,936 x 2,560 bf16 embedding, 778 MiB - so 1024 MB gives back 1 GiB of
#   transient memory at the exact moment that OOM'd.
export UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-1024}

########################### anchor reward ###########################
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.1}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_PPL_REWARD_STRATEGY=${IF_PPL_REWARD_STRATEGY:-anchor}
export IF_PPL_ANCHOR_REWARD_MODE=${IF_PPL_ANCHOR_REWARD_MODE:-both}
# THIS FLAG SELECTS THE ARM. Leave it false.
#
# IF_REF_ANCHOR_PRECOMPUTE is the cache-LOADING gate in the trainer, not just a
# "generate the anchors" switch: with it false, _precompute_if_ref_anchor_cache()
# clears the cache and _attach_if_ref_anchor_cache() returns early, so
# p_y_ref_given_x / p_y_ref_xc_given_x are never attached and
# apply_if_ppl_anchor_reward() returns the reward untouched. That is exactly what
# this arm wants: IFEval constraint + Qwen3-30B-A3B judge bonus, no anchor.
# The sibling judge-only arm (qwen3_4b_llmverifier_qwen3_4b_bonus01_reasoning.sh)
# sets it the same way. The hybrid arm is the one whose FILENAME carries
# t4b_anchor_pyx01; it sets this true AND if_llm_verifier_anchor_fallback_only.
#
# Consequences, so nobody "fixes" this again:
#   * PY_GIVEN_X_REWARD_COEFF and IF_PPL_REWARD_STRATEGY below are inert here -
#     they multiply anchor fields that are never attached. They are kept so this
#     script stays a one-flag diff from the hybrid.
#   * EXPERIMENT_NAME still reads pyx01_t4banchor. That is copy-paste residue
#     shared with the sibling judge-only arm, NOT a claim that the anchor is on.
#   * IF_REF_ANCHOR_CACHE_PATH / IF_REF_ANCHOR_TRAIN_CACHED_ONLY stay set for a
#     different reason: they filter the training split to the same 93,882
#     cached-anchor rows the anchor arms train on, keeping the arms comparable.
export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-false}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-true}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-true}
# This cache was produced with Qwen's standard (x-only) prefix and final-answer
# NLL scope, before those two fields were recorded in the metadata. Without the
# legacy flag the trainer reads the missing keys as a PPL-semantics mismatch and
# throws the whole 93,882-row cache away.
export IF_PPL_PREFIX_MODE=${IF_PPL_PREFIX_MODE:-standard}
export IF_REF_ANCHOR_ALLOW_LEGACY_FINAL_ANSWER_CACHE=${IF_REF_ANCHOR_ALLOW_LEGACY_FINAL_ANSWER_CACHE:-true}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.json}
export IF_REF_ANCHOR_HF_REPO=${IF_REF_ANCHOR_HF_REPO:-sangyon/anchor_cache}
export IF_REF_ANCHOR_CACHE_FILENAME=${IF_REF_ANCHOR_CACHE_FILENAME:-if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.json}
export IF_REF_PPL_BASELINE=${IF_REF_PPL_BASELINE:-0}
export IF_REF_PPL_ANCHOR=${IF_REF_PPL_ANCHOR:-0}

if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "[anchor] downloading hf://datasets/${IF_REF_ANCHOR_HF_REPO}/${IF_REF_ANCHOR_CACHE_FILENAME}"
    "${IF_LLM_VERIFIER_PYTHON}" - <<'PY'
import os
import shutil

from huggingface_hub import hf_hub_download

dest = os.environ["IF_REF_ANCHOR_CACHE_PATH"]
src = hf_hub_download(
    os.environ["IF_REF_ANCHOR_HF_REPO"],
    filename=os.environ["IF_REF_ANCHOR_CACHE_FILENAME"],
    repo_type="dataset",
)
os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
if os.path.realpath(src) != os.path.realpath(dest):
    shutil.copyfile(src, dest)
print(f"[anchor] downloaded {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
PY
fi
if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "ERROR: missing Qwen3-4B reasoning anchor cache: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    exit 1
fi

# Assert the cache is the legacy v2 final-answer artefact the flags above claim
# it is, and that it actually has complete rows. IF_REF_ANCHOR_TRAIN_CACHED_ONLY
# filters the training set down to these keys, so an empty or wrong cache is a
# silently 0-row run rather than a crash.
"${IF_LLM_VERIFIER_PYTHON}" - <<'PY'
import json
import math
import os
import sys

path = os.environ["IF_REF_ANCHOR_CACHE_PATH"]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
metadata = payload.get("metadata", {})

problems = []
if int(metadata.get("version", -1)) != 2:
    problems.append(f"version={metadata.get('version')!r}, expected 2 (legacy final-answer cache)")
if metadata.get("model_path") != os.environ["MODEL_PATH"]:
    problems.append(f"model_path={metadata.get('model_path')!r}, expected {os.environ['MODEL_PATH']!r}")
if "ppl_prefix_mode" in metadata or "ppl_nll_scope" in metadata:
    problems.append("cache already records ppl_prefix_mode/ppl_nll_scope; drop the legacy flag")
if os.environ["IF_PPL_PREFIX_MODE"] != "standard":
    problems.append(f"IF_PPL_PREFIX_MODE={os.environ['IF_PPL_PREFIX_MODE']!r}; the legacy path requires 'standard'")
for key, expected in (
    ("if_dataset_seed", 1),
    ("if_dataset_val_size", 512),
    ("max_prompt_length", int(os.environ["MAX_PROMPT_LENGTH"])),
):
    if metadata.get(key) != expected:
        problems.append(f"{key}={metadata.get(key)!r}, expected {expected!r}")
if metadata.get("apply_chat_template_kwargs") != {"enable_thinking": True}:
    problems.append(f"apply_chat_template_kwargs={metadata.get('apply_chat_template_kwargs')!r}, expected reasoning-on")
if problems:
    raise SystemExit("[anchor preflight] " + "; ".join(problems))

items = payload.get("items", {})
complete = sum(
    1
    for item in items.values()
    if item.get("y0")
    and item.get("y1")
    and int(item.get("ref0_token_count", 0) or 0) > 0
    and int(item.get("ref1_token_count", 0) or 0) > 0
    and math.isfinite(float(item.get("ref0_nll", float("inf"))))
    and math.isfinite(float(item.get("ref1_nll", float("inf"))))
)
if complete == 0:
    raise SystemExit(f"[anchor preflight] no complete anchor rows in {path}")
steps = complete // int(os.environ["TRAIN_BATCH_SIZE"])
print(
    f"[anchor preflight] OK: {complete}/{len(items)} complete rows -> {steps} steps/epoch, "
    f"{steps * int(os.environ['TOTAL_EPOCHS'])} steps total "
    "(build-host tokenizer_class/response_length ignored as informational)"
)
sys.exit(0)
PY

########################### checkpoints + wandb ###########################
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
# Tag derived from the threshold so the run name cannot drift from the config.
IF_LLM_VERIFIER_THRESHOLD_TAG=${IF_LLM_VERIFIER_THRESHOLD/./}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_think_pyx01_t4banchor_llmverifier_qwen3_30ba3b_nonthink_bonus01_threshold${IF_LLM_VERIFIER_THRESHOLD_TAG}_b1024_c1}

CKPT_DIR=${CKPT_DIR:-${VERL_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
mkdir -p "${CKPT_DIR}"

if [[ -z "${WANDB_API_KEY:-}" && ! -s "${HOME}/.netrc" ]]; then
    echo "ERROR: wandb logging is required but no WANDB_API_KEY and no ~/.netrc were found." >&2
    echo "       export WANDB_API_KEY=... (or run 'wandb login') before starting this run." >&2
    exit 1
fi

# fp32 FSDP model shards (~16 GB) + Adam state (~32 GB) per saved step.
IF_CKPT_MIN_GB=${IF_CKPT_MIN_GB:-200}
if [[ "$(free_gb "${CKPT_DIR}")" -lt "${IF_CKPT_MIN_GB}" ]]; then
    echo "ERROR: ${CKPT_DIR} has $(free_gb "${CKPT_DIR}") GB free; ${TOTAL_EPOCHS} checkpoints need ~$((TOTAL_EPOCHS * 50)) GB." >&2
    echo "       Set IF_CKPT_MIN_GB=0 to override, or point CKPT_DIR at a bigger volume." >&2
    exit 1
fi

########################### banner ###########################
cat >&2 <<BANNER
[run] experiment      : ${EXPERIMENT_NAME}
[run] policy          : ${MODEL_PATH} (reasoning, ${MAX_PROMPT_LENGTH}+${MAX_RESPONSE_LENGTH} tokens)
[run] judge           : ${IF_LLM_VERIFIER_MODEL} @ ${IF_LLM_VERIFIER_REVISION}
[run] judge sampling  : thinking=${IF_LLM_VERIFIER_ENABLE_THINKING} temp=${IF_LLM_VERIFIER_TEMPERATURE} top_p=${IF_LLM_VERIFIER_TOP_P} top_k=${IF_LLM_VERIFIER_TOP_K} min_p=${IF_LLM_VERIFIER_MIN_P}
[run] judge reward    : threshold=${IF_LLM_VERIFIER_THRESHOLD} bonus=${IF_LLM_VERIFIER_BONUS}
[run] anchor          : pyx=${PY_GIVEN_X_REWARD_COEFF} cache=${IF_REF_ANCHOR_CACHE_PATH}
[run] gpus            : train=${GPU_SET} (tp=${ROLLOUT_TP}) judge=${IF_LLM_VERIFIER_GPU_SET} (tp=${IF_LLM_VERIFIER_TP})
[run] batch           : ${TRAIN_BATCH_SIZE} x n=${ROLLOUT_N}, ${TOTAL_EPOCHS} epochs, save/test every ${SAVE_FREQ} steps
[run] checkpoints     : ${CKPT_DIR}
[run] hf cache        : ${HF_HUB_CACHE}
[run] wandb           : ${WANDB_ENTITY}/${PROJECT_NAME}
BANNER

########################### weight prefetch ###########################
# Four vLLM servers starting at once would otherwise race on the same 57 GB
# download. Fetch both models exactly once, up front.
echo "[setup] prefetching model weights into ${HF_HUB_CACHE}"
"${IF_LLM_VERIFIER_PYTHON}" - <<'PY'
import os

from huggingface_hub import snapshot_download

IGNORE = ["*.pth", "*.bin", "*.bin.index.json", "original/*", "consolidated*"]

targets = [(os.environ["MODEL_PATH"], None)]
verifier = os.environ["IF_LLM_VERIFIER_MODEL"]
if not os.path.isdir(verifier):
    # The judge servers are launched with --revision, so pin the same snapshot.
    targets.append((verifier, os.environ.get("IF_LLM_VERIFIER_REVISION") or None))

for repo_id, revision in targets:
    if os.path.isdir(repo_id):
        continue
    path = snapshot_download(repo_id, revision=revision, ignore_patterns=IGNORE, max_workers=16)
    print(f"[setup] ready: {repo_id} -> {path}")
PY

########################### verifier launch ###########################
VERIFIER_PIDS=()
VERIFIER_LOG_FILES=()
VERIFIER_BASE_URL_ARRAY=()

cleanup_verifier() {
    local status=$?
    trap - EXIT INT TERM
    if ((${#VERIFIER_PIDS[@]})); then
        for verifier_pid in "${VERIFIER_PIDS[@]}"; do
            kill -- "-${verifier_pid}" 2>/dev/null || kill "${verifier_pid}" 2>/dev/null || true
        done
        for verifier_pid in "${VERIFIER_PIDS[@]}"; do
            wait "${verifier_pid}" 2>/dev/null || true
        done
        VERIFIER_PIDS=()
    fi
    exit "${status}"
}
trap cleanup_verifier EXIT INT TERM

wait_for_verifier() {
    local verifier_base_url=$1
    local verifier_timeout=$2
    local verifier_pid=${3:-}
    "${IF_LLM_VERIFIER_PYTHON}" - "${verifier_base_url}" "${verifier_timeout}" "${verifier_pid}" <<'PY'
import os
import sys
import time
import urllib.request

base_url = sys.argv[1].rstrip("/")
timeout = float(sys.argv[2])
pid = int(sys.argv[3]) if sys.argv[3] else None
url = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"

deadline = time.time() + timeout
last_error = None
while time.time() < deadline:
    if pid is not None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"IF LLM verifier process {pid} exited before becoming ready.", file=sys.stderr)
            sys.exit(2)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status < 500:
                print(f"[IF LLM verifier] ready: {url}")
                sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
    time.sleep(2)

print(f"Timed out waiting for IF LLM verifier at {url}: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

IFS=',' read -r -a VERIFIER_GPU_ARRAY <<< "${IF_LLM_VERIFIER_GPU_SET//[[:space:]]/}"
if ((${#VERIFIER_GPU_ARRAY[@]} % IF_LLM_VERIFIER_TP != 0)); then
    echo "ERROR: IF_LLM_VERIFIER_GPU_SET (${#VERIFIER_GPU_ARRAY[@]} GPUs) is not divisible by IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP}." >&2
    exit 1
fi
VERIFIER_REPLICAS=$((${#VERIFIER_GPU_ARRAY[@]} / IF_LLM_VERIFIER_TP))

for ((replica = 0; replica < VERIFIER_REPLICAS; replica++)); do
    VERIFIER_BASE_URL_ARRAY+=("http://${IF_LLM_VERIFIER_HOST}:$((IF_LLM_VERIFIER_PORT + replica))/v1")
done
IF_LLM_VERIFIER_BASE_URLS=$(IFS=,; echo "${VERIFIER_BASE_URL_ARRAY[*]}")
export IF_LLM_VERIFIER_BASE_URLS
export IF_LLM_VERIFIER_BASE_URL="${VERIFIER_BASE_URL_ARRAY[0]}"

if [[ "${IF_LLM_VERIFIER_START_SERVER}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    mkdir -p "${IF_LLM_VERIFIER_LOG_DIR}"
    for ((replica = 0; replica < VERIFIER_REPLICAS; replica++)); do
        replica_gpus=()
        for ((lane = 0; lane < IF_LLM_VERIFIER_TP; lane++)); do
            replica_gpus+=("${VERIFIER_GPU_ARRAY[$((replica * IF_LLM_VERIFIER_TP + lane))]}")
        done
        replica_gpu_csv=$(IFS=,; echo "${replica_gpus[*]}")
        replica_port=$((IF_LLM_VERIFIER_PORT + replica))
        replica_log="${IF_LLM_VERIFIER_LOG_DIR}/qwen3_30ba3b_gpu${replica_gpu_csv//,/_}_${replica_port}.log"

        VERIFIER_CMD=(
            "${IF_LLM_VERIFIER_PYTHON}" -m vllm.entrypoints.cli.main serve "${IF_LLM_VERIFIER_MODEL}"
            --served-model-name "${IF_LLM_VERIFIER_MODEL}"
            --host "${IF_LLM_VERIFIER_HOST}"
            --port "${replica_port}"
            --tensor-parallel-size "${IF_LLM_VERIFIER_TP}"
            --dtype "${IF_LLM_VERIFIER_DTYPE}"
            --gpu-memory-utilization "${IF_LLM_VERIFIER_GPU_MEM_UTIL}"
            --max-model-len "${IF_LLM_VERIFIER_MAX_MODEL_LEN}"
            --max-num-batched-tokens "${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS}"
            --max-num-seqs "${IF_LLM_VERIFIER_MAX_NUM_SEQS}"
        )
        if [[ -n "${IF_LLM_VERIFIER_REVISION}" && ! -d "${IF_LLM_VERIFIER_MODEL}" ]]; then
            VERIFIER_CMD+=(--revision "${IF_LLM_VERIFIER_REVISION}" --tokenizer-revision "${IF_LLM_VERIFIER_REVISION}")
        fi
        if [[ "${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--enable-sleep-mode)
        fi
        if [[ "${IF_LLM_VERIFIER_ENFORCE_EAGER}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--enforce-eager)
        fi
        if [[ "${IF_LLM_VERIFIER_TRUST_REMOTE_CODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            VERIFIER_CMD+=(--trust-remote-code)
        fi

        setsid env CUDA_VISIBLE_DEVICES="${replica_gpu_csv}" \
            VLLM_SERVER_DEV_MODE="${IF_LLM_VERIFIER_DEV_MODE}" \
            VLLM_ALLREDUCE_USE_SYMM_MEM="${IF_LLM_VERIFIER_USE_SYMM_MEM}" \
            "${VERIFIER_CMD[@]}" >"${replica_log}" 2>&1 &
        verifier_pid=$!
        VERIFIER_PIDS+=("${verifier_pid}")
        VERIFIER_LOG_FILES+=("${replica_log}")
        echo "[IF LLM verifier] pid=${verifier_pid} gpu=${replica_gpu_csv} port=${replica_port} log=${replica_log}" >&2
    done
fi

for replica in "${!VERIFIER_BASE_URL_ARRAY[@]}"; do
    if ! wait_for_verifier "${VERIFIER_BASE_URL_ARRAY[$replica]}" "${IF_LLM_VERIFIER_WAIT_TIMEOUT}" "${VERIFIER_PIDS[$replica]:-}"; then
        replica_log="${VERIFIER_LOG_FILES[$replica]:-}"
        if [[ -n "${replica_log}" && -f "${replica_log}" ]]; then
            echo "========== verifier log: ${replica_log} ==========" >&2
            tail -100 "${replica_log}" >&2
        fi
        exit 1
    fi
done

if [[ "${IF_LLM_VERIFIER_MANAGE_SLEEP}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if [[ ! "${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        echo "ERROR: IF_LLM_VERIFIER_MANAGE_SLEEP=true requires IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=true." >&2
        exit 1
    fi
    # Hand the GPUs back before the trainer profiles its own memory.
    for verifier_base_url in "${VERIFIER_BASE_URL_ARRAY[@]}"; do
        IF_LLM_VERIFIER_CONTROL_URL="${verifier_base_url}" \
            bash "${SCRIPT_DIR}/control_qwen3_30ba3b_verifier.sh" sleep "${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    done
fi
########################### end verifier launch ###########################

REWARD_MANAGER_PATH="${VERL_DIR}/if_rlvr/if_llm_verifier_reward_manager.py"

OVERRIDES=(
    # --- 8-GPU rollout profile (no effect on the objective) ---------------
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MB}"
    trainer.default_local_dir="${CKPT_DIR}"
    # --- ray runtime env ---------------------------------------------------
    # Do NOT add PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: vLLM's
    # sleep mode allocates through a CUDA memory pool and asserts "Expandable
    # segments are not compatible with memory pool", killing every engine.
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=${HF_HOME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=${HF_HUB_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.NLTK_DATA=${NLTK_DATA}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.MODEL_PATH=${MODEL_PATH}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_PPL_PREFIX_MODE=${IF_PPL_PREFIX_MODE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REF_ANCHOR_ALLOW_LEGACY_FINAL_ANSWER_CACHE=${IF_REF_ANCHOR_ALLOW_LEGACY_FINAL_ANSWER_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG}"
    # --- anchor cache semantics -------------------------------------------
    "+if_ppl_prefix_mode=${IF_PPL_PREFIX_MODE}"
    "+if_ref_anchor_allow_legacy_final_answer_cache=${IF_REF_ANCHOR_ALLOW_LEGACY_FINAL_ANSWER_CACHE}"
    # --- reward: IFEval constraint + Qwen3-30B-A3B judge bonus -------------
    reward.num_workers="${IF_LLM_VERIFIER_REWARD_WORKERS}"
    +reward.compute_after_rollout=true
    reward.reward_model.enable=False
    reward.reward_manager.source=importlib
    reward.reward_manager.name=IFLLMVerifierRewardManager
    reward.reward_manager.module.path="${REWARD_MANAGER_PATH}"
    custom_reward_function.path=null
    "+reward.reward_kwargs.verification_reward=1.0"
    "+reward.reward_kwargs.if_llm_verifier_base_url=${IF_LLM_VERIFIER_BASE_URL}"
    "+reward.reward_kwargs.if_llm_verifier_base_urls='${IF_LLM_VERIFIER_BASE_URLS}'"
    "+reward.reward_kwargs.if_llm_verifier_model=${IF_LLM_VERIFIER_MODEL}"
    "+reward.reward_kwargs.if_llm_verifier_enable_thinking=${IF_LLM_VERIFIER_ENABLE_THINKING}"
    "+reward.reward_kwargs.if_llm_verifier_bonus=${IF_LLM_VERIFIER_BONUS}"
    "+reward.reward_kwargs.if_llm_verifier_threshold=${IF_LLM_VERIFIER_THRESHOLD}"
    "+reward.reward_kwargs.if_llm_verifier_temperature=${IF_LLM_VERIFIER_TEMPERATURE}"
    "+reward.reward_kwargs.if_llm_verifier_top_p=${IF_LLM_VERIFIER_TOP_P}"
    "+reward.reward_kwargs.if_llm_verifier_top_k=${IF_LLM_VERIFIER_TOP_K}"
    "+reward.reward_kwargs.if_llm_verifier_min_p=${IF_LLM_VERIFIER_MIN_P}"
    "+reward.reward_kwargs.if_llm_verifier_max_tokens=${IF_LLM_VERIFIER_MAX_TOKENS}"
    "+reward.reward_kwargs.if_llm_verifier_omit_max_tokens=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS}"
    "+reward.reward_kwargs.if_llm_verifier_timeout=${IF_LLM_VERIFIER_TIMEOUT}"
    "+reward.reward_kwargs.if_llm_verifier_max_retries=${IF_LLM_VERIFIER_MAX_RETRIES}"
    "+reward.reward_kwargs.if_llm_verifier_response_format=${IF_LLM_VERIFIER_RESPONSE_FORMAT}"
    "+reward.reward_kwargs.if_llm_verifier_manage_sleep=${IF_LLM_VERIFIER_MANAGE_SLEEP}"
    "+reward.reward_kwargs.if_llm_verifier_sleep_level=${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    "+reward.reward_kwargs.if_llm_verifier_control_timeout=${IF_LLM_VERIFIER_CONTROL_TIMEOUT}"
)
if [[ -n "${IF_LLM_VERIFIER_REASONING_EFFORT}" ]]; then
    OVERRIDES+=("+reward.reward_kwargs.if_llm_verifier_reasoning_effort=${IF_LLM_VERIFIER_REASONING_EFFORT}")
fi

set +e
bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" "${OVERRIDES[@]}" "$@"
status=$?
set -e
exit "${status}"
