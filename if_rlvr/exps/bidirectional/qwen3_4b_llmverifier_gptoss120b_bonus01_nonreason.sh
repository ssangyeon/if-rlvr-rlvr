#!/usr/bin/env bash
# GRPO | Qwen/Qwen3-4B policy (non-reasoning, enable_thinking=False)
#   + openai/gpt-oss-120b LLM-verifier bonus, +0.1 at score >= T
#   NO anchor / PPL reward shaping - the judge is the only non-rule signal.
#
# Judge-model swap of qwen3_4b_llmverifier_qwen3_4b_bonus01_nonreason.sh: same
# policy, same 2k/b1024/n8/4-epoch/lr-5e-7 profile, same sleep-wake GPU
# choreography, same threshold-calibration workflow. Only the judge and the
# memory budget it forces differ. The judge side follows the three existing
# gpt-oss-120b precedents in this directory, principally
# llama31_tulu3_8b_dpo_llmverifier_gptoss120b_bonus01_t5_nonreason.sh.
#
# ---------------------------------------------------------------------------
# MXFP4: one 120B judge per GPU, no sharding
# ---------------------------------------------------------------------------
# openai/gpt-oss-120b ships MXFP4-quantized: `quantization_config.quant_method
# = "mxfp4"`, with attention, the MoE router, embeddings and lm_head left
# unquantized. The safetensors total 65.2 GB (verified via HfApi against
# revision b5c939de8f754692c1647ca79fbf85e8c1e70f8a), which fits one 80 GB H100
# with room for a KV cache, so this runs as 8 TP=1 replicas - one judge per GPU,
# NO tensor-parallel sharding. TP=2 would only halve the replica count for no
# memory benefit. Same conclusion the three sibling gpt-oss scripts reached.
#
# The repo ALSO ships a 65.2 GB bf16 copy under `original/` plus a `metal/`
# export; both are excluded at download time. Fetching them would double the
# disk cost and, worse, `original/` is what vLLM would have to dequantize to.
#
# On Hopper the MXFP4 path depends on `triton_kernels` being importable - vLLM
# routes gpt-oss MoE through gpt_oss_triton_kernels_moe.matmul_ogs. Without it
# the weights are upcast to bf16 (~240 GB) and nothing fits, so the preflight
# below asserts it rather than letting eight engines discover it separately.
#
# ---------------------------------------------------------------------------
# Per-rollout reward
# ---------------------------------------------------------------------------
#   1. policy rollout on GPUs 0-7 (vLLM, TP=1 per GPU)
#   2. IFEval constraint reward; rows with constraint=0 stop here
#   3. every constraint-positive rollout gets ONE gpt-oss-120b judge call on the
#      constraint-FREE prompt x (`ppl_prompt`) plus the response y;
#      score >= IF_LLM_VERIFIER_THRESHOLD adds +0.1.
#
# gpt-oss answers through its harmony format (analysis channel + final channel).
# The reward manager reads the FINAL channel only (_message_final_content, which
# never falls back to reasoning_content), so no `enable_thinking`
# chat_template_kwarg is sent - that is a Qwen3-hybrid knob and this script
# deliberately omits the override the Qwen3-judge sibling passes. Reasoning
# depth is controlled by IF_LLM_VERIFIER_REASONING_EFFORT instead (empty =
# gpt-oss default).
#
# ---------------------------------------------------------------------------
# Expected wall-clock: this arm is SLOW
# ---------------------------------------------------------------------------
# A reasoning judge with no max_tokens cap emits chain-of-thought for every one
# of ~6,700 judged rollouts per step. The repo's own like-for-like record
# (if_rlvr/docs/if_rlvr_tulu3_epoch5_6_continuation_review_2026-08-18.md 9.3,
# identical policy/batch/epochs) measured:
#     constraint-only            362 steps  16.41 h  (163 s/step)
#     Qwen3-30B-A3B judge  t7    364 steps  38.13 h  (377 s/step)
#     gpt-oss-120b judge   t5    362 steps  64.54 h  (642 s/step)
# i.e. the gpt-oss judge cost 1.7x the Qwen3-30B-A3B judge. The Qwen3-4B-judge
# sibling of THIS script measured 251.6 s/step with a 17.0 s judge phase, so
# budget roughly 50-75 h here rather than ~25 h. Set
# IF_LLM_VERIFIER_REASONING_EFFORT=low to trade judge quality for a materially
# shorter CoT (deviates from the precedents, which used the default).
#
# ---------------------------------------------------------------------------
# Threshold calibration (REQUIRED before the real run)
# ---------------------------------------------------------------------------
# The threshold is read once at reward-manager construction, so it is a
# launch-time parameter, and an LLM judge's score scale is model-specific: the
# Qwen3-4B judge on this exact policy calibrated to score_mean 5.13 -> T=5,
# which says nothing about where gpt-oss-120b centres. The three gpt-oss
# precedents in this repo all used T=5, which is a reasonable fallback, but
# calibrate rather than assume:
#
#   # 1. one throwaway training step, judge scores dumped, no checkpoint, no wandb
#   IF_JUDGE_CALIBRATE=1 bash qwen3_4b_llmverifier_gptoss120b_bonus01_nonreason.sh
#   #    -> prints if_llm_verifier/score_mean, the full 1..10 histogram, and the
#   #       pass rate at every candidate threshold. Recommended = round(score_mean).
#
#   # 2. the real 4-epoch run
#   IF_LLM_VERIFIER_THRESHOLD=<T> bash qwen3_4b_llmverifier_gptoss120b_bonus01_nonreason.sh
#
# score_mean is threshold-INDEPENDENT (collect_if_llm_verifier_metrics in
# verl/trainer/ppo/ray_trainer.py averages llm_verifier_score over every called,
# finite, >=1 judgment), which is what makes a single probe step enough. Note
# the probe is far from free here - one step is ~10 min of judge time plus a
# multi-minute 8x65 GB engine load.
#
# ---------------------------------------------------------------------------
# GPU plan - one model per GPU, no tensor sharding anywhere
# ---------------------------------------------------------------------------
# Per GPU against the card's 81,559 MiB. gpt-oss-120b: 65.2 GB of MXFP4 weights
# = 62,179 MiB. 36 layers ALTERNATE sliding_attention (window 128) and
# full_attention, so only 18 layers hold growing KV: 18 x 2 x 8 KV heads x 64
# head_dim x 2 B = 36 KiB per token, four times cheaper than Qwen3-4B's 144 KiB.
#
#   A. startup   8 judge replicas, TP=1, one per GPU, profiled on empty GPUs at
#                util 0.85 -> 69,325 MiB of vLLM-managed budget, of which 62,179
#                is weights. Then slept at level 1 (weights -> host RAM, KV
#                freed) BEFORE the trainer profiles its own memory.
#                Host cost: 8 x 65.2 GB = 522 GB of the 2,015 GB on this box.
#                Engine load is slow (8 replicas x 65 GB); WAIT_TIMEOUT is 2400 s.
#   B. rollout   trainer's vLLM engine profiles with the judges asleep, util 0.72
#                (the value the gpt-oss precedent used; the Qwen3-4B-judge
#                sibling runs 0.75 but its asleep judge is only 864 MiB, whereas
#                a CUDA-graph-capturing gpt-oss replica's asleep footprint is
#                not yet measured on this box - the calibration probe will show
#                it, and this can be raised afterwards).
#   C. judging   rollout engine asleep at level 2, judge wakes. THE BINDING
#                CONSTRAINT, and the one that has already OOM'd a live run in
#                this repo. gpt-oss's Triton `matmul_ogs` MoE kernel allocates
#                its output workspace with a raw torch.empty() OUTSIDE vLLM's
#                memory pool, sized by the token count of that forward call. At
#                max_num_batched_tokens=32768 / max_num_seqs=128 the precedent
#                measured real usage at ~72-73 GiB - past the nominal 0.85
#                budget - and the run died with torch.OutOfMemoryError the
#                moment the trainer's resident footprint landed on the same GPU.
#                util cannot go much below 0.85 without starving the 62 GB of
#                weights, so the fix is on the other side: 8192/64 keeps the MoE
#                forward small enough to leave real headroom. Do not raise them.
#   D. update    both engines asleep, trainer alone.
#
#   Because phase C is this tight, ACTOR_PARAM_OFFLOAD and
#   ACTOR_OPTIMIZER_OFFLOAD are True here - the OPPOSITE of the Qwen3-4B-judge
#   sibling, where they are False. That sibling measured the trainer holding
#   8,410 MiB during generation with them off; 8.4 GiB on top of a ~70 GiB awake
#   gpt-oss replica leaves ~1 GiB, which is where the precedent OOM'd. Offloaded,
#   the wake-time residual is the ~2-6 GiB the precedent observed. This costs
#   throughput (host round-trips for the fp32 parameters and Adam state) and is
#   a real part of why this arm is slower.
#
#   No --enforce-eager for the judge: both other gpt-oss precedents leave CUDA
#   graphs on for this model and graph capture was not implicated in the OOM.
#   Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True - vLLM's sleep
#   mode allocates through a CUDA memory pool and asserts against it.
#
# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
#   Identical to the Qwen3-4B-judge sibling: 95,373 rows - 512 val = 94,861, of
#   which 980 exceed max_prompt_length=2048 under the Qwen3 non-thinking
#   template (measured) -> 93,881 rows / 1024 = 91 steps per epoch, so
#   SAVE_FREQ=91 gives one checkpoint per epoch and 4 epochs = 364 steps.
#
# Usage:
#   IF_JUDGE_CALIBRATE=1 bash qwen3_4b_llmverifier_gptoss120b_bonus01_nonreason.sh
#   IF_LLM_VERIFIER_THRESHOLD=5 bash qwen3_4b_llmverifier_gptoss120b_bonus01_nonreason.sh
#   IF_PREFLIGHT_ONLY=1 bash ...   # env + weights check, no GPU claimed
#   IF_CONFIG_CHECK=1 bash ...     # + Hydra compose check, no GPU
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
    # is NOT a default for real runs (see the fail-closed branch below). 5 is the
    # value the three gpt-oss precedents used, purely so the composed names read
    # sensibly during a check.
    IF_LLM_VERIFIER_THRESHOLD=${IF_LLM_VERIFIER_THRESHOLD:-5}
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

    IF_JUDGE_CALIBRATE=1 bash if_rlvr/exps/bidirectional/qwen3_4b_llmverifier_gptoss120b_bonus01_nonreason.sh

then launch the real run with the threshold it recommends:

    IF_LLM_VERIFIER_THRESHOLD=<T> bash if_rlvr/exps/bidirectional/qwen3_4b_llmverifier_gptoss120b_bonus01_nonreason.sh
MSG
    exit 1
fi
if [[ ! "${IF_LLM_VERIFIER_THRESHOLD}" =~ ^([1-9]|10)$ ]]; then
    echo "ERROR: IF_LLM_VERIFIER_THRESHOLD must be an integer in 1..10 (the judge rubric's range), got: ${IF_LLM_VERIFIER_THRESHOLD}" >&2
    exit 1
fi
export IF_LLM_VERIFIER_THRESHOLD

########################### model / cache placement ###########################
# Qwen3-4B (8 GB) plus gpt-oss-120b's MXFP4 safetensors (65.2 GB) plus the
# ~250 MB IF dataset need ~80 GB of HF cache. Small container-default cache
# mounts silently break the run at download time, so fall back to a roomier disk
# instead of failing 20 minutes in.
IF_HF_CACHE_MIN_GB=${IF_HF_CACHE_MIN_GB:-120}
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
        echo "       Point HF_HOME at a disk that can hold Qwen3-4B (8 GB) + gpt-oss-120b (65 GB)." >&2
        exit 1
    fi
fi
export HF_HOME
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${CACHE_ROOT}"

########################### run isolation ###########################
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-qwen3_4b_gptoss120b_judge_slot${RUN_SLOT}}
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
export QWEN3_4B_REVISION=${QWEN3_4B_REVISION:-1cfa9a7208912126459214e8b04321603b3df60c}
export GPTOSS_MODEL_ID=${GPTOSS_MODEL_ID:-openai/gpt-oss-120b}
# Pinned: the revision whose MXFP4 safetensors were verified at 65.2 GB, and the
# same one the Tulu gpt-oss precedent pins.
export GPTOSS_REVISION=${GPTOSS_REVISION:-b5c939de8f754692c1647ca79fbf85e8c1e70f8a}

# Eight judge replicas starting at once would otherwise race on the same 65 GB
# download. Fetch each model exactly once, up front, and hand the resolved local
# snapshot to every engine so none of them touches the Hub at init.
resolve_snapshot() {
    IF_RESOLVE_MODEL_ID="$1" IF_RESOLVE_REVISION="$2" "${IF_PYTHON_BIN}" - <<'RESOLVE_PY'
import os
import sys

from huggingface_hub import snapshot_download

model_id = os.environ["IF_RESOLVE_MODEL_ID"]
revision = os.environ.get("IF_RESOLVE_REVISION") or None
if os.path.isdir(model_id):
    print(model_id)
    sys.exit(0)

# `original/` is a full bf16 duplicate of the MXFP4 weights (another 65.2 GB)
# and `metal/` is an unrelated export; both are dead weight, and pulling
# `original/` is what would let vLLM dequantize instead of using the MXFP4 path.
path = snapshot_download(
    model_id,
    revision=revision,
    ignore_patterns=[
        "*.pth", "*.bin", "*.bin.index.json",
        "original/*", "metal/*", "consolidated*",
    ],
)
for required in ("config.json", "tokenizer_config.json"):
    if not os.path.isfile(os.path.join(path, required)):
        sys.exit(f"incomplete snapshot for {model_id}: missing {required} under {path}")
print(path)
RESOLVE_PY
}

echo "[setup] resolving ${QWEN3_4B_MODEL_ID} @ ${QWEN3_4B_REVISION} into ${HF_HUB_CACHE}"
QWEN3_4B_SNAPSHOT=$(resolve_snapshot "${QWEN3_4B_MODEL_ID}" "${QWEN3_4B_REVISION}") \
    || { echo "ERROR: could not prefetch ${QWEN3_4B_MODEL_ID} (see above)." >&2; exit 1; }
echo "[setup] ready: ${QWEN3_4B_MODEL_ID} -> ${QWEN3_4B_SNAPSHOT}"

echo "[setup] resolving ${GPTOSS_MODEL_ID} @ ${GPTOSS_REVISION} (65.2 GB MXFP4) into ${HF_HUB_CACHE}"
GPTOSS_SNAPSHOT=$(resolve_snapshot "${GPTOSS_MODEL_ID}" "${GPTOSS_REVISION}") \
    || { echo "ERROR: could not prefetch ${GPTOSS_MODEL_ID} (see above)." >&2; exit 1; }
echo "[setup] ready: ${GPTOSS_MODEL_ID} -> ${GPTOSS_SNAPSHOT}"

# Exported because the preflight below runs in a separate interpreter and needs
# the judge path to assert its quantization before any engine starts. The
# authoritative IF_LLM_VERIFIER_MODEL_PATH is set later, in the LLM-verifier
# section; honour an explicit override of it here too.
export QWEN3_4B_SNAPSHOT GPTOSS_SNAPSHOT
export IF_LLM_VERIFIER_MODEL_PATH=${IF_LLM_VERIFIER_MODEL_PATH:-${GPTOSS_SNAPSHOT}}

# The policy rollout group reads this local path instead of the Hub ID, which is
# what keeps its 8 replicas off the Hub's per-IP 429 limit.
export MODEL_PATH=${MODEL_PATH:-${QWEN3_4B_SNAPSHOT}}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-${MODEL_PATH}}

# gpt-oss's OpenAI-compatible endpoint loads its harmony tiktoken encoding from a
# content-addressed cache under /tmp/tiktoken-rs-cache/<hash> on first use. That
# write is not atomic against concurrent writers and all 8 replicas hit the same
# hash at once - one partial write corrupts the file for everyone, surfacing
# minutes later as "openai_harmony.HarmonyError: invalid tiktoken vocab file"
# during API-server startup, AFTER the multi-minute engine init has completed.
# Populate it once, sequentially, before any replica starts.
echo "[setup] pre-warming openai_harmony encoding cache (avoids an 8-way concurrent-write race)"
"${IF_PYTHON_BIN}" -c "
from openai_harmony import load_harmony_encoding, HarmonyEncodingName
load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
print('[setup] harmony encoding cache ready')
"

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

for arch in ("Qwen3ForCausalLM", "GptOssForCausalLM"):
    if arch not in ModelRegistry.get_supported_archs():
        sys.exit(f"[preflight] vLLM {vllm.__version__} does not register {arch}")

# On Hopper, gpt-oss MXFP4 MoE runs through triton_kernels.matmul_ogs. If that
# import is missing vLLM upcasts the experts to bf16 (~240 GB) and no replica
# can fit on one 80 GB card - a failure that would otherwise appear eight times
# over, minutes into an engine load.
import importlib.util

for module in (
    "triton_kernels",
    "vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe",
    "vllm.model_executor.layers.quantization.mxfp4",
    "openai_harmony",
):
    if importlib.util.find_spec(module) is None:
        sys.exit(
            f"[preflight] {module} is not importable, so gpt-oss-120b cannot be served in "
            "MXFP4 on this box.\n"
            "            Without it vLLM dequantizes the MoE experts to bf16 (~240 GB) and "
            "every judge replica OOMs."
        )

# Assert the judge checkpoint really is the MXFP4 build, not the bf16 duplicate.
import json

judge_config = os.path.join(os.environ["IF_LLM_VERIFIER_MODEL_PATH"], "config.json")
with open(judge_config, "r", encoding="utf-8") as handle:
    judge_cfg = json.load(handle)
quant = (judge_cfg.get("quantization_config") or {}).get("quant_method")
if quant != "mxfp4":
    sys.exit(
        f"[preflight] judge checkpoint at {judge_config} reports quant_method={quant!r}, "
        "expected 'mxfp4'.\n"
        "            A bf16 gpt-oss-120b needs ~240 GB and cannot run at TP=1."
    )
layer_types = judge_cfg.get("layer_types") or []
full_attn = sum(1 for t in layer_types if t == "full_attention")
print(
    f"[preflight] judge: mxfp4, {judge_cfg.get('num_hidden_layers')} layers "
    f"({full_attn} full-attention, {len(layer_types) - full_attn} sliding) -> "
    f"{full_attn * 2 * judge_cfg.get('num_key_value_heads', 0) * judge_cfg.get('head_dim', 0) * 2 / 1024:.0f} "
    "KiB of KV per token"
)

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

# Rollout engine sizing; see the GPU plan, phase B. 0.72 follows the gpt-oss
# precedent rather than the 0.75 the Qwen3-4B-judge sibling runs, because that
# sibling's asleep judge is only 864 MiB whereas a CUDA-graph-capturing gpt-oss
# replica's asleep footprint has not been measured on this box.
#
# For calibration, the sibling at 0.75 with an 864 MiB asleep judge measured a
# per-GPU peak of 78,067 MiB of 81,559 (3,492 MiB clear), sampled every 5 s
# across two full steps and stable over 16 steps - i.e. 0.75 already runs that
# configuration at ~96% of the card. Every MiB a sleeping gpt-oss replica holds
# above 864 has to come out of this number, which is why this starts lower.
# Measure the asleep footprint during the calibration probe (nvidia-smi while
# the trainer is generating) and raise this only by what is actually free.
export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.72}
# Matches the Tulu reference arm this run is compared against (temperature 1.0,
# top_p 0.95). Note the other Qwen3-4B arms in this directory silently used
# verl's top_p=1.0 default instead.
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
export PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}

# True here, unlike the Qwen3-4B-judge sibling where both are False. See the GPU
# plan, phase C: that sibling measured the trainer holding 8,410 MiB during
# generation with them off, and 8.4 GiB on top of a ~70 GiB awake gpt-oss replica
# leaves ~1 GiB - which is exactly where the gpt-oss precedent OOM'd. Offloaded,
# the wake-time residual is the ~2-6 GiB that precedent observed. This is pure
# memory placement (it does not change the optimization) but it is not free: it
# pays a host round-trip on every log-prob pass and on the actor update.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}

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
# Served from the prefetched local snapshot but advertised under the canonical
# Hub id: the reward manager sends {"model": IF_LLM_VERIFIER_MODEL}, which must
# match --served-model-name.
export IF_LLM_VERIFIER_MODEL=${IF_LLM_VERIFIER_MODEL:-${GPTOSS_MODEL_ID}}
# Already exported next to the prefetch (the preflight needs it earlier than
# this section); repeated here so this block still reads as self-contained.
export IF_LLM_VERIFIER_MODEL_PATH=${IF_LLM_VERIFIER_MODEL_PATH:-${GPTOSS_SNAPSHOT}}
# NOTE: IF_LLM_VERIFIER_ENABLE_THINKING is deliberately NOT set. gpt-oss uses
# harmony channels, not Qwen3's enable_thinking chat-template kwarg, and the
# reward manager reads the final channel itself. Sending the kwarg would put an
# unrecognised chat_template_kwargs in every request. Reasoning depth is
# IF_LLM_VERIFIER_REASONING_EFFORT instead ("" = gpt-oss default; low|medium|high).
export IF_LLM_VERIFIER_REASONING_EFFORT=${IF_LLM_VERIFIER_REASONING_EFFORT:-}
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
# 8 replicas x 65 GB of MXFP4 weights is a slow load even from page cache.
export IF_LLM_VERIFIER_WAIT_TIMEOUT=${IF_LLM_VERIFIER_WAIT_TIMEOUT:-2400}

# Engine sizing - see the GPU plan, phase C. These four values are the ones that
# already cost this repo a live OOM; they are not free parameters.
#   * 0.85 leaves 69,325 MiB of vLLM-managed budget for 62,179 MiB of weights.
#     Lower starves the weights; higher leaves nothing for the trainer.
#   * 8192 / 64 keep the Triton matmul_ogs MoE workspace - a raw torch.empty()
#     OUTSIDE vLLM's pool, sized by tokens-per-forward - small enough that real
#     usage stays under the nominal budget. At 32768 / 128 the precedent measured
#     ~72-73 GiB and died with torch.OutOfMemoryError. Do not raise them.
#   * No --enforce-eager: both other gpt-oss precedents keep CUDA graphs on for
#     this model and graph capture was not implicated in the OOM.
export IF_LLM_VERIFIER_DTYPE=${IF_LLM_VERIFIER_DTYPE:-bfloat16}
export IF_LLM_VERIFIER_GPU_MEM_UTIL=${IF_LLM_VERIFIER_GPU_MEM_UTIL:-0.85}
export IF_LLM_VERIFIER_MAX_MODEL_LEN=${IF_LLM_VERIFIER_MAX_MODEL_LEN:-24576}
export IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS=${IF_LLM_VERIFIER_MAX_NUM_BATCHED_TOKENS:-8192}
export IF_LLM_VERIFIER_MAX_NUM_SEQS=${IF_LLM_VERIFIER_MAX_NUM_SEQS:-64}
export IF_LLM_VERIFIER_ENFORCE_EAGER=${IF_LLM_VERIFIER_ENFORCE_EAGER:-false}
export IF_LLM_VERIFIER_TRUST_REMOTE_CODE=${IF_LLM_VERIFIER_TRUST_REMOTE_CODE:-false}
# vLLM's default TP all-reduce path rendezvouses through PyTorch symmetric
# memory, which needs CUDA multicast/VMM privileges this container does not
# grant. Harmless at TP=1.
export IF_LLM_VERIFIER_USE_SYMM_MEM=${IF_LLM_VERIFIER_USE_SYMM_MEM:-0}

# Reward-side judging policy, matching the gpt-oss precedents. gpt-oss is a
# reasoning judge: it emits an analysis channel before the final {"Score": N},
# so OMIT_MAX_TOKENS=true sends no max_tokens at all and lets it run to
# max_model_len rather than truncating mid-thought. MAX_TOKENS is kept only for
# provenance with those scripts; it is not sent while OMIT is true.
export IF_LLM_VERIFIER_BONUS=${IF_LLM_VERIFIER_BONUS:-0.1}
export IF_LLM_VERIFIER_TEMPERATURE=${IF_LLM_VERIFIER_TEMPERATURE:-0.0}
export IF_LLM_VERIFIER_TOP_P=${IF_LLM_VERIFIER_TOP_P:-1.0}
export IF_LLM_VERIFIER_MAX_TOKENS=${IF_LLM_VERIFIER_MAX_TOKENS:-8192}
export IF_LLM_VERIFIER_OMIT_MAX_TOKENS=${IF_LLM_VERIFIER_OMIT_MAX_TOKENS:-true}
export IF_LLM_VERIFIER_RESPONSE_FORMAT=${IF_LLM_VERIFIER_RESPONSE_FORMAT:-true}
export IF_LLM_VERIFIER_TIMEOUT=${IF_LLM_VERIFIER_TIMEOUT:-300}
export IF_LLM_VERIFIER_MAX_RETRIES=${IF_LLM_VERIFIER_MAX_RETRIES:-2}
export IF_LLM_VERIFIER_REWARD_WORKERS=${IF_LLM_VERIFIER_REWARD_WORKERS:-64}

########################### checkpoints, wandb, Hub ###########################
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
IF_LLM_VERIFIER_BONUS_TAG=${IF_LLM_VERIFIER_BONUS/./}
EXPERIMENT_TAG=threshold${IF_LLM_VERIFIER_THRESHOLD}
((IF_JUDGE_CALIBRATE)) && EXPERIMENT_TAG=calib
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_grpo_nonthink_llmverifier_gptoss120b_bonus${IF_LLM_VERIFIER_BONUS_TAG}_${EXPERIMENT_TAG}_b${TRAIN_BATCH_SIZE}_c1_t1_2k}

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
export HF_CHECKPOINT_REPO_NAME=${HF_CHECKPOINT_REPO_NAME:-qwen3-4b-grpo-gptoss120b-t${IF_LLM_VERIFIER_THRESHOLD}}
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
[run] verifier        : ${IF_LLM_VERIFIER_MODEL} @ ${GPTOSS_REVISION}
[run]                   threshold=${IF_LLM_VERIFIER_THRESHOLD} bonus=${IF_LLM_VERIFIER_BONUS} mxfp4 tp=1 util=${IF_LLM_VERIFIER_GPU_MEM_UTIL} effort=${IF_LLM_VERIFIER_REASONING_EFFORT:-<default>}
[run] snapshots       : policy ${QWEN3_4B_SNAPSHOT}
[run]                   judge  ${GPTOSS_SNAPSHOT}
[run] reward          : IFEval constraint + judge bonus (no anchor/PPL shaping)
[run] gpus            : train=${GPU_SET} verifier=${IF_LLM_VERIFIER_GPU_SET} (tp=${IF_LLM_VERIFIER_TP})
[run] lengths         : prompt<=${MAX_PROMPT_LENGTH} response<=${MAX_RESPONSE_LENGTH} judge_out=uncapped(<=${IF_LLM_VERIFIER_MAX_MODEL_LEN})
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
        replica_log="${IF_LLM_VERIFIER_LOG_DIR}/gpt_oss_120b_gpu${replica_gpu_csv//,/_}_${replica_port}.log"

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

# Only sent when explicitly set; an empty value would make gpt-oss reject the
# request rather than fall back to its default.
if [[ -n "${IF_LLM_VERIFIER_REASONING_EFFORT}" ]]; then
    OVERRIDES+=("+reward.reward_kwargs.if_llm_verifier_reasoning_effort=${IF_LLM_VERIFIER_REASONING_EFFORT}")
fi

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
        --launcher "if_rlvr/exps/bidirectional/qwen3_4b_llmverifier_gptoss120b_bonus01_nonreason.sh" \
        || status=$?
fi

exit "${status}"
