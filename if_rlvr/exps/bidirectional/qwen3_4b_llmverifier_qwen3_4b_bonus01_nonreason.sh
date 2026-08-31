#!/usr/bin/env bash
# GRPO | Qwen/Qwen3-4B policy (non-reasoning, enable_thinking=False)
#   + Qwen/Qwen3-4B LLM-verifier bonus (also non-reasoning), +0.1 at score >= T
#   NO anchor / PPL reward shaping - the judge is the only non-rule signal.
#
# Self-judging counterpart of
# llama31_tulu3_8b_dpo_llmverifier_qwen3_30ba3b_bonus01_t7_nonreason.sh: same
# reward structure, same 2k/b1024/n8/4-epoch profile, same sleep-wake GPU
# choreography, but policy AND judge are the SAME non-reasoning Qwen3-4B, and
# the actor LR is 5e-7. Sized for one node of 8x H100 80GB.
#
# Differences from the Tulu reference that are NOT cosmetic:
#   * policy = judge = Qwen/Qwen3-4B. Both are served from ONE prefetched local
#     snapshot, so the judge scores against byte-identical weights and nine
#     engines never race on the same Hub download.
#   * Qwen3 is a hybrid-reasoning model, so non-reasoning is a chat-template
#     argument, not a property of the checkpoint: the policy gets
#     `data.apply_chat_template_kwargs.enable_thinking=false`
#     (IF_APPLY_ENABLE_THINKING_KWARG=true + ENABLE_THINKING=false) and the
#     judge gets `chat_template_kwargs={"enable_thinking": false}` per request.
#     Tulu 3 needed neither - its template has no such kwarg.
#   * IF_REQUIRE_THINK_END_FOR_REWARD=false: with thinking off there is no
#     </think> delimiter to require, and the whole response is the final answer.
#   * The reference's `actor.fsdp_config.model_dtype=fp32` /
#     `.dtype=bfloat16` / `ref.fsdp_config.model_dtype=fp32` overrides are
#     already verl's defaults (verl/workers/config/engine.py), so they are
#     omitted here rather than restated.
#   * Host-side param/optimizer offload is OFF (the reference needed it to fit a
#     61 GB judge; see the GPU plan below). The reference measured 52.2% mean GPU
#     utilisation with both offloads on, so this is a real throughput win.
#
# ---------------------------------------------------------------------------
# Per-rollout reward
# ---------------------------------------------------------------------------
#   1. policy rollout on GPUs 0-7 (vLLM, TP=1 per GPU)
#   2. IFEval constraint reward; rows with constraint=0 stop here
#   3. every constraint-positive rollout gets ONE Qwen3-4B judge call on the
#      constraint-FREE prompt x (`ppl_prompt`) plus the response y;
#      score >= IF_LLM_VERIFIER_THRESHOLD adds +0.1 on top of the constraint
#      reward.
#   Single reward pass, single judge call per eligible row
#   (if_llm_verifier_anchor_fallback_only is NOT set, so the reward manager runs
#   in its "combined" phase).
#
# ---------------------------------------------------------------------------
# Threshold calibration (REQUIRED before the real run)
# ---------------------------------------------------------------------------
# The threshold is read once at reward-manager construction, so it is a
# launch-time parameter. It is deliberately NOT defaulted here: an LLM judge's
# score scale is model-specific, and a threshold far from the score distribution
# makes the bonus either free (everyone passes) or dead (nobody does).
#
#   # 1. one throwaway training step, judge scores dumped, no checkpoint, no wandb
#   IF_JUDGE_CALIBRATE=1 bash qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh
#   #    -> prints if_llm_verifier/score_mean, the full 1..10 histogram, and the
#   #       pass rate at every candidate threshold. Recommended value is
#   #       round(score_mean).
#
#   # 2. the real 4-epoch run
#   IF_LLM_VERIFIER_THRESHOLD=<T> bash qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh
#
# Outcome on 2026-08-27 (Qwen3-4B judging Qwen3-4B, non-thinking both sides):
# score_mean 5.1265 over 6,442 parsed judgments of 6,542 calls on 8,192 rollouts
# (79.9% constraint-positive) -> T=5. The distribution is BIMODAL, not centred:
# scores 1-3 hold 39.5% and 7-10 hold 39.5%, with only 20.9% in 4-6 and troughs
# at 4 (3.96%) and 10 (3.96%) - this judge decides "bad" or "good" and rarely
# hedges, so the mean lands in a sparse valley. T=5 passes 56.5% of judged
# rollouts, near the even split that maximises within-group variance for GRPO,
# and the choice is insensitive: T=5 -> 6 moves the pass rate only 8.8pp.
# 1.53% of calls answered {"Score": 0}, outside the rubric, so extract_judge_score
# rejects them and they earn no bonus - counting them as 0 or 1 gives ~5.05,
# still T=5.
#
# Note that score_mean is threshold-INDEPENDENT (see collect_if_llm_verifier_metrics
# in verl/trainer/ppo/ray_trainer.py: it averages llm_verifier_score over every
# called, finite, >=1 judgment), which is what makes a single probe step enough.
# The probe still performs one actor update; it is discarded, which is why it
# writes no checkpoint and logs to console only.
#
# ---------------------------------------------------------------------------
# GPU plan - one model per GPU, no tensor sharding anywhere
# ---------------------------------------------------------------------------
# Per GPU, against the card's 81,559 MiB. Qwen3-4B: 4.02B params (tied
# embeddings) = 8.05 GB of bf16 weights; 36 layers x 8 KV heads x 128 head_dim
# = 144 KiB of KV per token.
#
# Figures marked MEASURED come from the 2026-08-27 calibration probe on this box
# (wandb-free, logs/judge_calibration/); the rest are the same arithmetic.
#
#   A. startup   8 judge replicas, TP=1, one per GPU, profiled on empty GPUs.
#                MEASURED at util 0.50: 30.68 GiB of KV = 223,424 tokens per
#                replica = 18.2x concurrency at the full 12,288-token request
#                length (~89 concurrent at a realistic ~2.5k-token judge prompt).
#                Each is then slept at level 1 (weights -> host RAM, KV freed)
#                BEFORE the trainer profiles its own memory; MEASURED asleep
#                footprint 891 MiB per GPU, i.e. bare CUDA context. Host cost:
#                8 x 8.0 GB = 64 GB of 2 TB.
#   B. rollout   trainer's vLLM engine profiles with the judges asleep. At util
#                0.75 it budgets 61,169 MiB and overshoots by ~2,100 MiB of
#                non-torch allocations -> ~63,300 MiB, ~12 GiB clear of the
#                trainer's resident params + Adam. 1024 prompts x n=8 = 8192
#                sequences over 8 replicas. MEASURED at 0.60: gen 87.0s,
#                response_length mean 508 / max 2048, 5.4% hitting the cap.
#   C. judging   rollout engine asleep at level 2 (weights AND KV released), judge
#                wakes beside the trainer's resident shard (2.0 fp32 params +
#                4.0 Adam; grads are not yet allocated and ref params live on the
#                host). MEASURED with the judge at 0.50: 47.2 GB per GPU total at
#                93-99% util, ~33 GB clear, whole phase 17.8s for 6,542 judge
#                calls. The reward loop sleeps the judge again in a `finally`, so
#                judge weights and the update's activation peak are never
#                resident together.
#   D. update    judge asleep, rollout asleep, trainer alone. 32,768-token
#                micro-batches -> 32,768 x 151,936 x 2 B = ~10 GB of bf16 logits
#                (verl's entropy_from_logits_with_chunking defaults to false) in
#                ~70 GB of free memory. MEASURED at 24,576: update_actor 95.0s,
#                old_log_prob 43.3s, ref 41.3s, step total 295.0s = 2,542
#                tokens/s/GPU.
#
#   Two environment facts inherited from the reference, kept because they cost
#   nothing at TP=1:
#     * VLLM_ALLREDUCE_USE_SYMM_MEM=0 - the container denies the CUDA multicast
#       privileges vLLM's default all-reduce rendezvous wants.
#     * the preflight below fails fast if the v1 engine worker stack stops
#       importing (vLLM 0.11 pins numba==0.61.2, which cannot import against
#       numpy>=2.3; a mismatch otherwise kills all nine engines minutes after
#       launch, with the cause buried in eight separate server logs).
#   Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - vLLM's sleep
#   mode allocates through a CUDA memory pool and asserts against it.
#
# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
#   95,373 rows - 512 held-out val = 94,861 train rows; 980 are dropped by
#   data.filter_overlong_prompts at max_prompt_length=2048 under the Qwen3
#   non-thinking template (measured, not estimated) -> 93,881 rows.
#   93,881 / 1024 = 91 steps per epoch (drop_last=True drops 697), so
#   SAVE_FREQ=91 gives exactly one checkpoint per epoch and 4 epochs = 364 steps.
#   The same 91 that llama31_tulu3_8b_dpo_constraint_only_nonreason.sh uses; the
#   reference script's own SAVE_FREQ=92 is a stale comment ("~94.9k / 1024 = 92")
#   that its sibling scripts already corrected.
#
#   data.seed is left unset (verl's create_rl_sampler then uses an unseeded
#   generator), matching every sibling arm. The train/val SPLIT is still
#   deterministic via IF_DATA_SEED=1.
#
# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
#   * wandb: trainer.logger=[console,wandb], ${WANDB_ENTITY}/${PROJECT_NAME}.
#   * Hugging Face Hub: `hf_model` is in the actor checkpoint contents, so each
#     saved step is a ready-to-load bf16 HF model. push_checkpoints_to_hf.py
#     uploads each completed step to ${HF_CHECKPOINT_REPO} under
#     `global_step_<N>/`, with a final sweep on exit. Load one with:
#         AutoModelForCausalLM.from_pretrained(repo, subfolder="global_step_91")
#
# Usage:
#   IF_JUDGE_CALIBRATE=1 bash qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh
#   IF_LLM_VERIFIER_THRESHOLD=7 bash qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh
#   IF_PREFLIGHT_ONLY=1 IF_JUDGE_CALIBRATE=1 bash ...   # env check, no GPU claimed
#   IF_CONFIG_CHECK=1 bash ...                          # + Hydra compose check, no GPU
# Any extra arguments are forwarded verbatim as Hydra overrides.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
DATA_ROOT=$(dirname -- "${VERL_DIR}")
export CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
cd "${VERL_DIR}"

########################### calibration mode ###########################
# 1 = throwaway probe step that only measures the judge score distribution.
IF_JUDGE_CALIBRATE=${IF_JUDGE_CALIBRATE:-0}
case "${IF_JUDGE_CALIBRATE}" in
    1 | true | TRUE | yes | YES | on | ON) IF_JUDGE_CALIBRATE=1 ;;
    0 | false | FALSE | no | NO | off | OFF) IF_JUDGE_CALIBRATE=0 ;;
    *) echo "ERROR: IF_JUDGE_CALIBRATE must be 0 or 1, got: ${IF_JUDGE_CALIBRATE}" >&2; exit 1 ;;
esac

# 1 = compose the Hydra config and exit, without claiming a GPU. Validates every
# override below against verl's schema (a missing `+`, a typo'd key, or a
# duplicate that Hydra will not accept) in seconds instead of after nine vLLM
# engines have loaded.
IF_CONFIG_CHECK=${IF_CONFIG_CHECK:-0}
case "${IF_CONFIG_CHECK}" in
    1 | true | TRUE | yes | YES | on | ON) IF_CONFIG_CHECK=1 ;;
    0 | false | FALSE | no | NO | off | OFF) IF_CONFIG_CHECK=0 ;;
    *) echo "ERROR: IF_CONFIG_CHECK must be 0 or 1, got: ${IF_CONFIG_CHECK}" >&2; exit 1 ;;
esac
if ((IF_CONFIG_CHECK)); then
    IF_LLM_VERIFIER_START_SERVER=false
    HF_CHECKPOINT_PUSH=false
    # Placeholder only, so the check does not demand a calibrated threshold; it
    # is NOT a default for real runs (see the fail-closed branch below).
    IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-7}
fi

if ((IF_JUDGE_CALIBRATE)); then
    # The probe's own threshold only shifts a reward we throw away; score_mean
    # does not depend on it. Park it at the top of the scale so the discarded
    # update is as close to constraint-only as possible.
    IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-10}
elif [[ -z "${IF_LLM_VERIFIER_THRESHOLD:-}" ]]; then
    cat >&2 <<'MSG'
ERROR: IF_LLM_VERIFIER_THRESHOLD is required and has no default.

Calibrate it first (one throwaway step, ~10 min, no checkpoint, no wandb run):

    IF_JUDGE_CALIBRATE=1 bash if_rlvr/exps/bidirectional/qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh

then launch the real run with the threshold it recommends:

    IF_LLM_VERIFIER_THRESHOLD=<T> bash if_rlvr/exps/bidirectional/qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh
MSG
    exit 1
fi
if [[ ! "${IF_LLM_VERIFIER_THRESHOLD}" =~ ^([1-9]|10)$ ]]; then
    echo "ERROR: IF_LLM_VERIFIER_THRESHOLD must be an integer in 1..10 (the judge rubric's range), got: ${IF_LLM_VERIFIER_THRESHOLD}" >&2
    exit 1
fi
export IF_LLM_VERIFIER_THRESHOLD

########################### model / cache placement ###########################
# One Qwen3-4B snapshot (~8 GB) serves the policy and all 8 judge replicas, plus
# the ~250 MB IF dataset. Small container-default cache mounts silently break the
# run at download time, so fall back to a roomier disk instead of failing later.
IF_HF_CACHE_MIN_GB=${IF_HF_CACHE_MIN_GB:-60}
HF_HOME=${HF_HOME:-${DATA_ROOT}/.cache/huggingface}
free_gb() {
    local dir=$1
    while [[ ! -d "${dir}" && "${dir}" != "/" ]]; do dir=$(dirname -- "${dir}"); done
    df -Pk "${dir}" 2>/dev/null | awk 'NR==2 {print int($4 / 1048576)}' | grep -E '^[0-9]+$' || echo 0
}
if [[ "$(free_gb "${HF_HOME}")" -lt "${IF_HF_CACHE_MIN_GB}" ]]; then
    HF_HOME_FALLBACK="${DATA_ROOT}/.cache/huggingface"
    if [[ "${HF_HOME%/}" != "${HF_HOME_FALLBACK%/}" && "$(free_gb "${HF_HOME_FALLBACK}")" -ge "${IF_HF_CACHE_MIN_GB}" ]]; then
        echo "[setup] HF_HOME=${HF_HOME} has $(free_gb "${HF_HOME}") GB free (< ${IF_HF_CACHE_MIN_GB} GB);" >&2
        echo "[setup] switching to ${HF_HOME_FALLBACK}. Set IF_HF_CACHE_MIN_GB=0 to keep the original." >&2
        HF_HOME="${HF_HOME_FALLBACK}"
    else
        echo "ERROR: no HF cache location with >= ${IF_HF_CACHE_MIN_GB} GB free (checked ${HF_HOME})." >&2
        echo "       Point HF_HOME at a disk that can hold Qwen3-4B (8 GB) + the IF dataset." >&2
        exit 1
    fi
fi
export HF_HOME
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${CACHE_ROOT}"

########################### run isolation ###########################
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-qwen3_4b_q4b_judge_slot${RUN_SLOT}}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-$((20000 + RUN_SLOT * 2000))}
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

# Whole-node run: claim all 8 GPUs before _concurrent_run_env.sh derives a set.
export GPU_SET=${GPU_SET:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# nltk punkt/punkt_tab: the base launcher and every reward worker resolve these
# through nltk.data.path, which always includes ${HOME}/nltk_data. Point at it
# when it is already populated so the launcher skips the download entirely.
if [[ -d "${HOME}/nltk_data/tokenizers/punkt_tab" ]]; then
    export NLTK_DATA_DIR=${NLTK_DATA_DIR:-${HOME}/nltk_data}
fi

########################### interpreter ###########################
# Prefer the environment that is already active when it can run vLLM; the
# sibling Tulu/Qwen scripts hardcode a ${IFIF_ROOT}/.miniforge3 layout that does
# not exist everywhere.
if [[ -z "${IF_PYTHON_BIN:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]] \
        && "${CONDA_PREFIX}/bin/python" -c 'import vllm' >/dev/null 2>&1; then
        IF_PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif command -v python3 >/dev/null && python3 -c 'import vllm' >/dev/null 2>&1; then
        IF_PYTHON_BIN=$(command -v python3)
    else
        echo "ERROR: no Python on PATH (or in CONDA_PREFIX) can import vLLM." >&2
        echo "       Activate the training environment first, or set IF_PYTHON_BIN=/path/to/python." >&2
        exit 1
    fi
fi
export IF_PYTHON_BIN
export PATH="$(dirname -- "${IF_PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

########################### model pinning + prefetch ###########################
export QWEN3_4B_MODEL_ID=${QWEN3_4B_MODEL_ID:-Qwen/Qwen3-4B}
# Snapshot pinned so the policy and the judge cannot drift apart across restarts.
export QWEN3_4B_REVISION=${QWEN3_4B_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}

echo "[setup] resolving ${QWEN3_4B_MODEL_ID} @ ${QWEN3_4B_REVISION} into ${HF_HUB_CACHE}"
QWEN3_4B_SNAPSHOT=$("${IF_PYTHON_BIN}" - <<'PY'
import os
import sys

from huggingface_hub import snapshot_download

model_id = os.environ["QWEN3_4B_MODEL_ID"]
revision = os.environ["QWEN3_4B_REVISION"] or None
if os.path.isdir(model_id):
    print(model_id)
    sys.exit(0)

# Both repos ship complete safetensors; skip the duplicated .bin/.pth copies.
path = snapshot_download(
    model_id,
    revision=revision,
    ignore_patterns=["*.pth", "*.bin", "*.bin.index.json", "original/*", "consolidated*"],
)
for required in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
    if not os.path.isfile(os.path.join(path, required)):
        sys.exit(f"incomplete snapshot for {model_id}: missing {required} under {path}")
print(path)
PY
) || { echo "ERROR: could not prefetch ${QWEN3_4B_MODEL_ID} (see above)." >&2; exit 1; }
echo "[setup] ready: ${QWEN3_4B_MODEL_ID} -> ${QWEN3_4B_SNAPSHOT}"

# Nine engines (1 policy rollout group + 8 judges) read this local path instead
# of the Hub ID, which is what keeps them off the Hub's per-IP 429 limit.
export MODEL_PATH=${MODEL_PATH:-${QWEN3_4B_SNAPSHOT}}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-${MODEL_PATH}}

########################### preflight ###########################
"${IF_PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    from vllm.v1.worker.gpu_worker import Worker  # noqa: F401
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"[preflight] vLLM engine worker stack is not importable: {type(exc).__name__}: {exc}\n"
        "            All nine vLLM engines in this run would fail the same way. Fix the "
        "environment first\n"
        "            (e.g. vLLM 0.11 pins numba==0.61.2, which requires numpy<2.3)."
    )

import transformers
import vllm
from transformers import AutoTokenizer
from vllm.model_executor.models.registry import ModelRegistry

if "Qwen3ForCausalLM" not in ModelRegistry.get_supported_archs():
    sys.exit(f"[preflight] vLLM {vllm.__version__} does not register Qwen3ForCausalLM")

tokenizer = AutoTokenizer.from_pretrained(os.environ["MODEL_PATH"], local_files_only=True)
if not tokenizer.chat_template:
    sys.exit("[preflight] Qwen3-4B tokenizer has no embedded chat template")

# Non-reasoning is a chat-template argument for Qwen3, not a property of the
# checkpoint, so assert BOTH branches. Qwen3's template does not omit the think
# block when thinking is off - it PRE-FILLS an empty, already-closed one:
#   enable_thinking=False -> '...<|im_start|>assistant\n<think>\n\n</think>\n\n'
#   enable_thinking=True  -> '...<|im_start|>assistant\n'
# That is what makes IF_REQUIRE_THINK_END_FOR_REWARD=false correct here: the
# </think> lives in the PROMPT, so the response never contains one and requiring
# it would zero every reward. It is also why IF_PPL_PREFIX_MODE stays 'standard'
# ('empty_think' exists for templates that end on an OPEN <think>).
kwargs = dict(add_generation_prompt=True, tokenize=False)
off = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Runtime preflight."}], enable_thinking=False, **kwargs
)
on = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Runtime preflight."}], enable_thinking=True, **kwargs
)
if off == on:
    sys.exit(
        "[preflight] enable_thinking has no effect on this chat template; the policy would "
        "train in Qwen3's DEFAULT mode, which is thinking.\n"
        f"            rendered: {off!r}"
    )
if "<|im_start|>assistant\n" not in off:
    sys.exit(f"[preflight] unexpected Qwen3 chat-template rendering: {off!r}")
if not off.rstrip().endswith("</think>"):
    sys.exit(
        "[preflight] enable_thinking=False does not leave the think block closed in the "
        "prompt.\n"
        "            IF_REQUIRE_THINK_END_FOR_REWARD=false assumes it does; the response "
        "would be scored\n"
        f"            as if it had already finished reasoning. rendered: {off!r}"
    )
if "</think>" in on:
    sys.exit(f"[preflight] enable_thinking=True unexpectedly closes the think block: {on!r}")

print(
    f"[preflight] transformers={transformers.__version__} vllm={vllm.__version__} "
    f"architecture=Qwen3ForCausalLM tokenizer={tokenizer.__class__.__name__}"
)
print(f"[preflight] enable_thinking=False generation prefix: {off.split('<|im_start|>assistant')[-1]!r}")

# IFLLMVerifierRewardManager._verify_runtime_deps raises on every Ray reward
# worker if any of these is missing - i.e. several minutes into the run, once per
# worker. Same check, here, in one second.
sys.path.insert(0, os.path.join(os.environ["VERL_DIR"], "if_rlvr"))
try:
    import langdetect  # noqa: F401

    from ifeval_oi import instructions_util
    from ifeval_oi.verifier import score_ifeval  # noqa: F401

    instructions_util.nltk.word_tokenize("This is a sentence. Here is another one.")
    instructions_util.count_words("two words")
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"[preflight] reward-manager dependency check failed: {type(exc).__name__}: {exc}\n"
        "            Install `langdetect immutabledict nltk` and the nltk punkt/punkt_tab data "
        "on EVERY\n"
        "            ray worker node (or point NLTK_DATA at shared storage)."
    )
print("[preflight] reward-manager deps (langdetect, ifeval_oi, nltk punkt) OK")
PY

########################### policy: non-reasoning Qwen3 ###########################
# ENABLE_THINKING=false + IF_APPLY_ENABLE_THINKING_KWARG=true is what makes the
# base launcher emit `+data.apply_chat_template_kwargs.enable_thinking=false`.
# Dropping the kwarg instead (as the Tulu scripts do) would silently train in
# Qwen3's DEFAULT mode, which is thinking.
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}
export IF_ALLOW_MISSING_THINK_FINAL_ANSWER=${IF_ALLOW_MISSING_THINK_FINAL_ANSWER:-true}
export IF_PPL_PREFIX_MODE=${IF_PPL_PREFIX_MODE:-standard}

########################### training hyper-parameters ###########################
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export ROLLOUT_N=${ROLLOUT_N:-8}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
# 93,881 filtered train rows / 1024 = 91 steps per epoch -> 1 ckpt/epoch.
export SAVE_FREQ=${SAVE_FREQ:-91}
export TEST_FREQ=${TEST_FREQ:-1000}
# Sequences are <= 2048 + 2048, so these are throughput knobs with headroom, not
# capacity limits (see the GPU plan, phase D). MEASURED at 24576 on this box:
# 5,997,996 batch tokens = ~750k per GPU per pass = ~30 micro-batches; 32768
# brings that to ~23. Both engines are asleep during the update, so the ~10 GiB
# of bf16 logits (32768 x 151,936 x 2 B) sits in ~70 GiB of free memory. Do not
# expect much from it: update_actor measured 95.0s and is compute-bound, so this
# only removes ~7 per-micro-batch FSDP all-gathers.
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-32}
export IF_DATA_SEED=${IF_DATA_SEED:-1}
export IF_VAL_SIZE=${IF_VAL_SIZE:-512}

# Rollout engine sizing; see the GPU plan, phase B. 0.60 was measured safe end
# to end (step 1 completed, 99% util, no OOM). 0.75 = 61,169 MiB of budget plus
# vLLM's ~2,100 MiB non-torch overshoot = ~63,300 MiB, and the only trainer
# growth after vLLM's profile point is the 4,096 MiB of Adam state that appears
# on the first optimizer step -> ~12 GiB still clear during generation.
#
# Honest expectation: near zero. Generation is STRAGGLER-bound here, not
# KV-bound - measured agent_loop/generate_sequences max 71.5s against a mean of
# 29.5s inside an 87.0s phase, i.e. one 2048-token response sets the phase
# length, and 0.60 already afforded ~190 concurrent sequences per replica.
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}
# Matches the Tulu reference arm this run is compared against (temperature 1.0,
# top_p 0.95). Note the other Qwen3-4B arms in this directory silently used
# verl's top_p=1.0 default instead.
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
export PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}

# Qwen3-4B's shard is small enough to keep parameters and Adam state resident
# (~8 GB/GPU, phase C), which the reduced utilisations above already fund. Raise
# either utilisation and these must go back to True or the rollout phase OOMs.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}

########################### reward: constraint + LLM verifier only ###########################
# No anchor/PPL shaping: PY/PX coefficients are 0 and the anchor cache is never
# loaded, so the FULL train split is used rather than a cached-anchor subset.
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.0}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-false}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-false}
export IF_REF_PPL_GATE=${IF_REF_PPL_GATE:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_PPL_BASELINE=${IF_REF_PPL_BASELINE:-0}
export IF_REF_PPL_ANCHOR=${IF_REF_PPL_ANCHOR:-0}

########################### LLM verifier ###########################
# Served from the same local snapshot as the policy, but advertised under the
# canonical Hub id: the reward manager sends {"model": IF_LLM_VERIFIER_MODEL},
# which must match --served-model-name, and a 100-char path in every request and
# log line buys nothing.
export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-${QWEN3_4B_MODEL_ID}}
export IF_LLM_VERIFIER_MODEL_PATH=${IF_LLM_VERIFIER_MODEL_PATH:-${QWEN3_4B_SNAPSHOT}}
# Judge in non-thinking mode so the reply is just the {"Score": N} JSON object.
export IF_LLM_VERIFIER_ENABLE_THINKING=${IF_LLM_VERIFIER_ENABLE_THINKING:-false}
export IF_LLM_VERIFIER_GPU_SET=${IF_LLM_VERIFIER_GPU_SET:-${GPU_SET}}
export IF_LLM_VERIFIER_TP=${IF_LLM_VERIFIER_TP:-1}
export IF_LLM_VERIFIER_HOST=${IF_LLM_VERIFIER_HOST:-127.0.0.1}
# The policy's own vLLM replicas reserve VLLM_MASTER_PORT_BASE + rank * 100, i.e.
# base+200 .. base+999 for 8 replicas. Start the judge servers above that block.
export IF_LLM_VERIFIER_PORT=${IF_LLM_VERIFIER_PORT:-$((IF_RLVR_PORT_BASE + 1200))}
export IF_LLM_VERIFIER_START_SERVER=${IF_LLM_VERIFIER_START_SERVER:-true}
export IF_LLM_VERIFIER_PYTHON=${IF_LLM_VERIFIER_PYTHON:-${IF_PYTHON_BIN}}
export IF_LLM_VERIFIER_LOG_DIR=${IF_LLM_VERIFIER_LOG_DIR:-${VERL_DIR}/logs/verifier}

# Wake/sleep alternation with the trainer.
export IF_LLM_VERIFIER_ENABLE_SLEEP_MODE=${IF_LLM_VERIFIER_ENABLE_SLEEP_MODE:-true}
export IF_LLM_VERIFIER_MANAGE_SLEEP=${IF_LLM_VERIFIER_MANAGE_SLEEP:-true}
export IF_LLM_VERIFIER_SLEEP_LEVEL=${IF_LLM_VERIFIER_SLEEP_LEVEL:-1}
export IF_LLM_VERIFIER_DEV_MODE=${IF_LLM_VERIFIER_DEV_MODE:-1}
export IF_LLM_VERIFIER_CONTROL_TIMEOUT=${IF_LLM_VERIFIER_CONTROL_TIMEOUT:-600}
export IF_LLM_VERIFIER_WAIT_TIMEOUT=${IF_LLM_VERIFIER_WAIT_TIMEOUT:-1800}

# Engine sizing; see the GPU plan, phases A and C.
export IF_LLM_VERIFIER_DTYPE=${IF_LLM_VERIFIER_DTYPE:-bfloat16}
# 0.50 measured: 30.68 GiB of KV = 223,424 tokens per replica, and the whole
# judge phase - wake, 6,542 calls, sleep - took 17.8s of a 295s step. 0.60 makes
# an already-cheap phase cheaper; it profiles on an empty GPU (~51,000 MiB awake)
# and the trainer holds only ~6 GiB at that point, leaving ~24 GiB clear.
export IF_LLM_VERIFIER_GPU_MEM_UTIL=${IF_LLM_VERIFIER_GPU_MEM_UTIL:-0.60}
# Judge prompt = rubric (~250 tok) + x (<=2048) + y (<=2048) = ~4.4k, plus a
# 2048-token generation cap -> ~6.4k worst case. 12288 leaves ~2x margin, and
# vLLM rejects any request over this with HTTP 400 rather than truncating.
export IF_LLM_VERIFIER_MAX_MODEL_LEN=${IF_LLM_VERIFIER_MAX_MODEL_LEN:-12288}
export IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS=${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS:-16384}
export IF_LLM_VERIFIER_MAX_NUM_SEQS=${IF_LLM_VERIFIER_MAX_NUM_SEQS:-128}
# The judge decodes ~20 tokens per call, so CUDA graphs buy little and cost both
# capture time (x8 replicas) and the GPU memory the KV cache wants.
export IF_LLM_VERIFIER_ENFORCE_EAGER=${IF_LLM_VERIFIER_ENFORCE_EAGER:-true}
export IF_LLM_VERIFIER_TRUST_REMOTE_CODE=${IF_LLM_VERIFIER_TRUST_REMOTE_CODE:-false}
# vLLM's default TP all-reduce path rendezvouses through PyTorch symmetric
# memory, which needs CUDA multicast/VMM privileges this container does not
# grant. Harmless at TP=1.
export IF_LLM_VERIFIER_USE_SYMM_MEM=${IF_LLM_VERIFIER_USE_SYMM_MEM:-0}

# Reward-side judging policy.
export IF_LLM_VERIFIER_BONUS=${IF_LLM_VERIFIER_BONUS:-0.1}
export IF_LLM_VERIFIER_TEMPERATURE=${IF_LLM_VERIFIER_TEMPERATURE:-0.0}
export IF_LLM_VERIFIER_TOP_P=${IF_LLM_VERIFIER_TOP_P:-1.0}
# 2048 per this run's spec. A well-formed non-thinking answer is ~17 tokens and
# response_format=json_object constrains decoding to a JSON object, so the
# straggler tail the reference bounded at 1024 is mostly structural here rather
# than length-limited; extract_judge_score also has a regex fallback that
# recovers the score from truncated JSON.
export IF_LLM_VERIFIER_MAX_TOKENS=${IF_LLM_VERIFIER_MAX_TOKENS:-2048}
export IF_LLM_VERIFIER_OMIT_MAX_TOKENS=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS:-false}
export IF_LLM_VERIFIER_RESPONSE_FORMAT=${IF_LLM_VERIFIER_RESPONSE_FORMAT:-true}
# A whole phase is issued at once, so a request can wait behind the rest of the
# batch; time out generously rather than silently dropping bonuses.
export IF_LLM_VERIFIER_TIMEOUT=${IF_LLM_VERIFIER_TIMEOUT:-900}
export IF_LLM_VERIFIER_MAX_RETRIES=${IF_LLM_VERIFIER_MAX_RETRIES:-0}
export IF_LLM_VERIFIER_REWARD_WORKERS=${IF_LLM_VERIFIER_REWARD_WORKERS:-64}

########################### checkpoints, wandb, Hub ###########################
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
IF_LLM_VERIFIER_BONUS_TAG=${IF_LLM_VERIFIER_BONUS/./}
EXPERIMENT_TAG=threshold${IF_LLM_VERIFIER_THRESHOLD}
((IF_JUDGE_CALIBRATE)) && EXPERIMENT_TAG=calib
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_nonthink_llmverifier_qwen3_4b_nonthink_bonus${IF_LLM_VERIFIER_BONUS_TAG}_${EXPERIMENT_TAG}_b${TRAIN_BATCH_SIZE}_c1_t1_2k}

CKPT_DIR=${CKPT_DIR:-${VERL_DIR}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
((IF_CONFIG_CHECK)) || mkdir -p "${CKPT_DIR}"

CALIB_DUMP_DIR="${VERL_DIR}/logs/judge_calibration/${EXPERIMENT_NAME}"
if ((IF_JUDGE_CALIBRATE)); then
    HF_CHECKPOINT_PUSH=false
    rm -rf "${CALIB_DUMP_DIR}"
    mkdir -p "${CALIB_DUMP_DIR}"
elif [[ -z "${WANDB_API_KEY:-}" && ! -s "${HOME}/.netrc" ]]; then
    echo "ERROR: wandb logging is required but no WANDB_API_KEY and no ~/.netrc were found." >&2
    echo "       export WANDB_API_KEY=... (or run 'wandb login') before starting this run." >&2
    exit 1
fi

HF_CHECKPOINT_PUSH=${HF_CHECKPOINT_PUSH:-true}
HF_CHECKPOINT_REPO_PRIVATE=${HF_CHECKPOINT_REPO_PRIVATE:-false}
HF_CHECKPOINT_POLL_SECONDS=${HF_CHECKPOINT_POLL_SECONDS:-120}
HF_CHECKPOINT_FINAL_SWEEP=${HF_CHECKPOINT_FINAL_SWEEP:-true}
HF_CHECKPOINT_PUSHER="${SCRIPT_DIR}/push_checkpoints_to_hf.py"
# EXPERIMENT_NAME is far longer than the Hub's 96-character repo-name limit, so
# the repo gets its own short, stable name.
export HF_CHECKPOINT_REPO_NAME=${HF_CHECKPOINT_REPO_NAME:-qwen3-4b-grpo-q4b-t${IF_LLM_VERIFIER_THRESHOLD}}
if [[ "${HF_CHECKPOINT_PUSH}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    if [[ -z "${HF_CHECKPOINT_REPO:-}" ]]; then
        # Default to <hub-user>/<short-name>; the token must be able to write it.
        HF_CHECKPOINT_REPO=$("${IF_PYTHON_BIN}" - <<'PY'
import os
import sys

from huggingface_hub import HfApi

try:
    user = HfApi(token=os.getenv("HF_TOKEN") or None).whoami()["name"]
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"cannot resolve the Hugging Face account for checkpoint upload: {type(exc).__name__}: {exc}\n"
        "Export a write-scoped HF_TOKEN, or set HF_CHECKPOINT_REPO=<owner>/<name>, "
        "or disable uploading with HF_CHECKPOINT_PUSH=false."
    )
print(f"{user}/{os.environ['HF_CHECKPOINT_REPO_NAME'][:96]}")
PY
        ) || {
            echo "ERROR: could not determine the Hugging Face checkpoint repo (see the message above)." >&2
            exit 1
        }
    fi
    export HF_CHECKPOINT_REPO
    echo "[hf-push] checkpoints -> https://huggingface.co/${HF_CHECKPOINT_REPO} (private=${HF_CHECKPOINT_REPO_PRIVATE})"
else
    HF_CHECKPOINT_REPO=""
    echo "[hf-push] disabled (HF_CHECKPOINT_PUSH=${HF_CHECKPOINT_PUSH})"
fi

########################### banner ###########################
cat >&2 <<BANNER
[run] experiment      : ${EXPERIMENT_NAME}
[run] mode            : $( ((IF_JUDGE_CALIBRATE)) && echo "THRESHOLD CALIBRATION (1 step, discarded)" || echo "full training" )
[run] policy          : ${QWEN3_4B_MODEL_ID} @ ${QWEN3_4B_REVISION} (thinking=${ENABLE_THINKING})
[run] verifier        : ${IF_LLM_VERIFIER_MODEL} (threshold=${IF_LLM_VERIFIER_THRESHOLD}, bonus=${IF_LLM_VERIFIER_BONUS}, thinking=${IF_LLM_VERIFIER_ENABLE_THINKING})
[run] snapshot        : ${QWEN3_4B_SNAPSHOT}
[run] reward          : IFEval constraint + judge bonus (no anchor/PPL shaping)
[run] gpus            : train=${GPU_SET} verifier=${IF_LLM_VERIFIER_GPU_SET} (tp=${IF_LLM_VERIFIER_TP})
[run] lengths         : prompt<=${MAX_PROMPT_LENGTH} response<=${MAX_RESPONSE_LENGTH} judge_out<=${IF_LLM_VERIFIER_MAX_TOKENS}
[run] optim           : lr=${ACTOR_LR}, batch ${TRAIN_BATCH_SIZE} x n=${ROLLOUT_N}, temp=${ROLLOUT_TEMPERATURE} top_p=${ROLLOUT_TOP_P}
[run] schedule        : ${TOTAL_EPOCHS} epochs, save every ${SAVE_FREQ} steps
[run] checkpoints     : ${CKPT_DIR}
[run] wandb           : $( ((IF_JUDGE_CALIBRATE)) && echo "disabled (console only)" || echo "${WANDB_ENTITY}/${PROJECT_NAME}" )
BANNER

# Everything above touches no GPU: it validates the environment, prefetches the
# snapshot both roles load, and resolves the Hub repo. IF_PREFLIGHT_ONLY=1 stops
# here, which is the cheap way to check a box (or a busy node) before committing
# to a run.
if [[ "${IF_PREFLIGHT_ONLY:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    echo "[preflight] IF_PREFLIGHT_ONLY set; stopping before any GPU is claimed." >&2
    exit 0
fi

########################### verifier servers ###########################
VERIFIER_PIDS=()
VERIFIER_LOG_FILES=()
VERIFIER_BASE_URL_ARRAY=()
HF_PUSH_PID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "${HF_PUSH_PID}" ]]; then
        kill "${HF_PUSH_PID}" 2>/dev/null || true
        wait "${HF_PUSH_PID}" 2>/dev/null || true
        HF_PUSH_PID=""
    fi
    if ((${#VERIFIER_PIDS[@]})); then
        for pid in "${VERIFIER_PIDS[@]}"; do
            kill -- "-${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
        done
        for pid in "${VERIFIER_PIDS[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
        VERIFIER_PIDS=()
    fi
    if [[ -n "${HF_CHECKPOINT_REPO:-}" && \
          "${HF_CHECKPOINT_FINAL_SWEEP}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        echo "[hf-push] final sweep for ${CKPT_DIR}" >&2
        local -a sweep_args=(
            --ckpt-dir "${CKPT_DIR}"
            --repo-id "${HF_CHECKPOINT_REPO}"
            --run-name "${EXPERIMENT_NAME}"
            --base-model "${QWEN3_4B_MODEL_ID}"
            --once
        )
        if [[ "${HF_CHECKPOINT_REPO_PRIVATE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
            sweep_args+=(--private)
        fi
        "${IF_PYTHON_BIN}" "${HF_CHECKPOINT_PUSHER}" "${sweep_args[@]}" \
            || echo "[hf-push] final sweep failed; re-run push_checkpoints_to_hf.py manually." >&2
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

wait_for_verifier() {
    local base_url=$1
    local timeout=$2
    local pid=${3:-}
    "${IF_PYTHON_BIN}" - "${base_url}" "${timeout}" "${pid}" <<'PY'
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
    if [[ ! -x "${IF_LLM_VERIFIER_PYTHON}" ]]; then
        echo "ERROR: verifier Python is not executable: ${IF_LLM_VERIFIER_PYTHON}" >&2
        exit 1
    fi
    mkdir -p "${IF_LLM_VERIFIER_LOG_DIR}"

    for ((replica = 0; replica < VERIFIER_REPLICAS; replica++)); do
        replica_gpus=()
        for ((lane = 0; lane < IF_LLM_VERIFIER_TP; lane++)); do
            replica_gpus+=("${VERIFIER_GPU_ARRAY[$((replica * IF_LLM_VERIFIER_TP + lane))]}")
        done
        replica_gpu_csv=$(IFS=,; echo "${replica_gpus[*]}")
        replica_port=$((IF_LLM_VERIFIER_PORT + replica))
        replica_log="${IF_LLM_VERIFIER_LOG_DIR}/qwen3_4b_judge_gpu${replica_gpu_csv//,/_}_${replica_port}.log"

        VERIFIER_CMD=(
            "${IF_LLM_VERIFIER_PYTHON}" -m vllm.entrypoints.cli.main serve "${IF_LLM_VERIFIER_MODEL_PATH}"
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
    ((IF_CONFIG_CHECK)) && break
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
    # Hand the GPUs back to the trainer before it profiles its own memory.
    for base_url in "${VERIFIER_BASE_URL_ARRAY[@]}"; do
        ((IF_CONFIG_CHECK)) && break
        IF_LLM_VERIFIER_CONTROL_URL="${base_url}" \
            bash "${SCRIPT_DIR}/control_qwen3_30ba3b_verifier.sh" sleep "${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    done
fi
########################### end verifier servers ###########################

########################### checkpoint -> Hub watcher ###########################
if [[ "${HF_CHECKPOINT_PUSH}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    HF_PUSH_ARGS=(
        --ckpt-dir "${CKPT_DIR}"
        --repo-id "${HF_CHECKPOINT_REPO}"
        --run-name "${EXPERIMENT_NAME}"
        --base-model "${QWEN3_4B_MODEL_ID}"
        --poll-seconds "${HF_CHECKPOINT_POLL_SECONDS}"
    )
    if [[ "${HF_CHECKPOINT_REPO_PRIVATE}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        HF_PUSH_ARGS+=(--private)
    fi
    mkdir -p "${VERL_DIR}/logs"
    "${IF_PYTHON_BIN}" "${HF_CHECKPOINT_PUSHER}" "${HF_PUSH_ARGS[@]}" \
        >"${VERL_DIR}/logs/hf_push_${EXPERIMENT_NAME}.log" 2>&1 &
    HF_PUSH_PID=$!
    echo "[hf-push] watcher pid=${HF_PUSH_PID} log=${VERL_DIR}/logs/hf_push_${EXPERIMENT_NAME}.log" >&2
fi

########################### actor training ###########################
REWARD_MANAGER_PATH="${VERL_DIR}/if_rlvr/if_llm_verifier_reward_manager.py"

OVERRIDES=(
    # --- rollout profile --------------------------------------------------
    "+if_ppl_prefix_mode=${IF_PPL_PREFIX_MODE}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    # --- GPU residency: see the GPU plan, phase C ------------------------
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
    # --- checkpoints: keep the resumable shards AND export an HF model ----
    trainer.default_local_dir="${CKPT_DIR}"
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]"
    "actor_rollout_ref.actor.checkpoint.load_contents=[model,optimizer,extra]"
    # --- ray runtime env --------------------------------------------------
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=${HF_HOME}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=${HF_HUB_CACHE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
    # Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here. vLLM's
    # sleep mode allocates through a CUDA memory pool and asserts
    # "Expandable segments are not compatible with memory pool", killing every
    # rollout engine at init. verl itself toggles it off around weight sync.
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_APPLY_ENABLE_THINKING_KWARG=\"${IF_APPLY_ENABLE_THINKING_KWARG}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_ALLOW_MISSING_THINK_FINAL_ANSWER=\"${IF_ALLOW_MISSING_THINK_FINAL_ANSWER}\""
    # --- reward: IFEval constraint + LLM verifier bonus ------------------
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
    "+reward.reward_kwargs.if_llm_verifier_max_tokens=${IF_LLM_VERIFIER_MAX_TOKENS}"
    "+reward.reward_kwargs.if_llm_verifier_omit_max_tokens=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS}"
    "+reward.reward_kwargs.if_llm_verifier_timeout=${IF_LLM_VERIFIER_TIMEOUT}"
    "+reward.reward_kwargs.if_llm_verifier_max_retries=${IF_LLM_VERIFIER_MAX_RETRIES}"
    "+reward.reward_kwargs.if_llm_verifier_response_format=${IF_LLM_VERIFIER_RESPONSE_FORMAT}"
    "+reward.reward_kwargs.if_llm_verifier_manage_sleep=${IF_LLM_VERIFIER_MANAGE_SLEEP}"
    "+reward.reward_kwargs.if_llm_verifier_sleep_level=${IF_LLM_VERIFIER_SLEEP_LEVEL}"
    "+reward.reward_kwargs.if_llm_verifier_control_timeout=${IF_LLM_VERIFIER_CONTROL_TIMEOUT}"
)

if ((IF_JUDGE_CALIBRATE)); then
    # One step, nothing kept: no checkpoint (save_freq<=0 skips even the
    # last-step save), no last-step validation (test_freq<=0), no wandb run, no
    # resume from a stale probe, and every judged rollout dumped to JSONL so the
    # full score distribution - not just its mean - is available.
    #
    # Keep the base launcher's retry loop ARMED (3 attempts, not 1). Bringing up
    # 16 vLLM engine processes at once - 8 judges plus 8 rollout replicas - races
    # on ephemeral ports: vLLM's get_open_port() picks a free port, closes the
    # probe socket, and only later binds it in TCPStore, so a sibling process can
    # take it in between. Measured on this box: attempt 1 of the first probe lost
    # exactly that race (7 of 8 rollout engines up, one dead with
    # "DistNetworkError ... port: 41631 ... EADDRINUSE") and, with retries
    # disabled, the whole probe exited rc=1 after 4.5 minutes. Riding this out is
    # what the autoresume loop exists for. Retrying is safe here specifically
    # because resume_mode=disable makes every attempt a clean step-0 restart, and
    # the judge servers are this script's children - they stay up and asleep
    # across attempts, so a retry costs only the trainer's boot.
    export IF_MAX_RETRIES=${IF_MAX_RETRIES:-3}
    OVERRIDES+=(
        trainer.total_training_steps=1
        trainer.save_freq=-1
        trainer.test_freq=-1
        trainer.resume_mode=disable
        'trainer.logger=["console"]'
        "trainer.rollout_data_dir=${CALIB_DUMP_DIR}"
    )
fi

if ((IF_CONFIG_CHECK)); then
    export IF_MAX_RETRIES=1
    echo "[config-check] composing the Hydra config; no GPU is claimed and no training runs." >&2
fi

set +e
bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" "${OVERRIDES[@]}" "$@" \
    $( ((IF_CONFIG_CHECK)) && echo "--cfg job --resolve" )
status=$?
set -e

if ((IF_CONFIG_CHECK)); then
    ((status == 0)) && echo "[config-check] OK: every override composed against verl's schema." >&2
    exit "${status}"
fi

if ((IF_JUDGE_CALIBRATE)) && ((status == 0)); then
    echo
    "${IF_PYTHON_BIN}" "${SCRIPT_DIR}/summarize_judge_scores.py" \
        --dump-dir "${CALIB_DUMP_DIR}" --bonus "${IF_LLM_VERIFIER_BONUS}" \
        || status=$?
fi

exit "${status}"
