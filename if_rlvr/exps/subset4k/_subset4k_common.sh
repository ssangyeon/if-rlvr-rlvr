#!/usr/bin/env bash
# Shared frame for the Qwen3-4B 4k-subset reward ablations (Stage 1).
#
# Sourced by the per-arm scripts AFTER they export their reward knobs and ARM_TAG.
# Everything except the reward knobs is identical across arms:
#   - same 4,096 rows (v2 subset cache as the row filter, TRAIN_CACHED_ONLY)
#   - b=256 (16 steps/epoch), 4 epochs = 64 steps, per-epoch checkpoints
#   - rollout sampling t1.0 / top_p 0.95 / top_k 20 (matched to the v2 anchors)
#   - budgets 2048/8192, lr 1e-6, n=8 rollouts, no in-training validation
#
# Metadata strictness: the v2 cache records tokenizer_class=Qwen2Tokenizer (B200
# build env) and train_sample_count=93882 (build universe), which can never equal
# this box's/this subset's values, so the full-dict strict check is disabled and
# replaced by the explicit semantic-field preflight below.
set -euo pipefail

SUBSET4K_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=${VERL_DIR:-$(cd -- "${SUBSET4K_DIR}/../../.." && pwd)}
BASE_SCRIPT="${VERL_DIR}/if_rlvr/exps/bidirectional/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh"

: "${ARM_TAG:?arm script must set ARM_TAG}"

export IF_REF_ANCHOR_CACHE_PATH=${SUBSET_CACHE:-${VERL_DIR}/.cache/if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.json}
if [[ ! -s "${IF_REF_ANCHOR_CACHE_PATH}" ]]; then
    echo "[subset4k] missing v2 subset cache: ${IF_REF_ANCHOR_CACHE_PATH}" >&2
    echo "[subset4k] carve it from the completed v2 cache first (see data_subsets/qwen3_4b_reasoning_anchor4k/)." >&2
    exit 1
fi

# Preflight: assert the semantic metadata fields; informational fields
# (tokenizer_class, train_sample_count) are deliberately ignored. When the arm
# does not use anchors (coeff 0, e.g. A0), the cache is only a row filter, so
# only the split identity and row count are asserted — this lets A0 run with the
# v1 subset file (identical indices) before the v2 carve exists.
python3 - "${IF_REF_ANCHOR_CACHE_PATH}" "${PY_GIVEN_X_REWARD_COEFF:-0.1}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
anchors_used = float(sys.argv[2]) != 0.0
md = payload.get("metadata", {})
expect = {"if_dataset_seed": 1, "if_dataset_val_size": 512}
if anchors_used:
    expect.update({
        "rollout_temperature": 1.0, "rollout_top_p": 0.95, "rollout_top_k": 20,
        "response_length": 8192, "max_response_length": 8192, "max_prompt_length": 2048,
        "ppl_prefix_mode": "standard", "ppl_nll_scope": "final_answer_tokens_only",
        "model_path": "Qwen/Qwen3-4B",
    })
bad = {k: (md.get(k), v) for k, v in expect.items() if md.get(k) != v}
if bad:
    raise SystemExit(f"[subset4k preflight] cache metadata mismatch (got, expected): {bad}")
n = len(payload.get("items", {}))
want = 4096
if n != want:
    raise SystemExit(f"[subset4k preflight] expected {want} cache items, found {n}")
print(f"[subset4k preflight] cache OK: {n} items, "
      f"{'semantic metadata verified' if anchors_used else 'row-filter-only checks'} "
      f"(tokenizer_class/train_sample_count ignored as informational)")
PY

# ---- shared frame (env consumed by the base launcher) ----
export HF_HOME=${HF_HOME:-${VERL_DIR}/.cache/huggingface}   # Qwen3-4B + dataset live here
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=true
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=false
export ENABLE_THINKING=true
if [[ "${PY_GIVEN_X_REWARD_COEFF:-0.1}" == "0.0" ]]; then
    # Anchor-free arm (A0): the cache is ONLY the dataset row filter. Keep the
    # trainer's cache loader completely out of the picture (its semantic check
    # would REGENERATE-AND-OVERWRITE a legacy v1-metadata file even with
    # strict=false), and skip the dataset-side prefix-semantics check so the
    # v1 subset file can serve as the row filter before the v2 carve exists.
    export IF_REF_ANCHOR_PRECOMPUTE=false
    unset IF_PPL_PREFIX_MODE
else
    export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-true}
    export IF_PPL_PREFIX_MODE=standard
fi

export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export TRAIN_BATCH_SIZE=256
export PPO_MINI_BATCH_SIZE=256
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=8192
# Memory profile for 8x80GB, mirroring the proven full-scale qwen runs
# (util 0.9 + engine sleep) with one deliberate change: a lower microbatch
# token cap. At b=256 the dynamic-bsz packer can produce worse-balanced
# microbatches than at b=1024, and a 98304-token microbatch OOMs on the
# ~26GB logits of the 151k vocab (observed 2026-08-14). 65536 restores
# headroom at a modest throughput cost.
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.9}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-256}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
export SAVE_FREQ=16          # one checkpoint per epoch (4096/256 = 16 steps)
export TEST_FREQ=-1          # no in-training validation; offline epoch evals instead
export ACTOR_LR=1e-6
export ROLLOUT_N=8
export PRESENCE_PENALTY=0.0

export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
export EXPERIMENT_NAME="qwen3_4b_sub4k_${ARM_TAG}_b256_ep${TOTAL_EPOCHS}"

echo "[subset4k] arm=${ARM_TAG}"
echo "[subset4k]   flip_handling=${IF_PPL_ANCHOR_FLIP_HANDLING:-abstain} floor_action=${IF_PPL_ANCHOR_FLOOR_ACTION:-zero} floor_penalty=${IF_PPL_ANCHOR_FLOOR_PENALTY:-0.1}"
echo "[subset4k]   anchor_mode=${IF_PPL_ANCHOR_REWARD_MODE:-both} pyx_coeff=${PY_GIVEN_X_REWARD_COEFF:-0.1} policy_ppl=${IF_REF_POLICY_ANCHOR_PPL:-true}"
echo "[subset4k]   experiment=${EXPERIMENT_NAME}"

# Checkpoints land in ./checkpoints/<project>/<experiment> relative to CWD —
# pin it to the repo root so every launch context agrees.
cd "${VERL_DIR}"

# Rollout sampling matched to the v2 anchor generation regime; hf_model saved
# per checkpoint so the offline eval battery has ready-to-load HF exports.
exec bash "${BASE_SCRIPT}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]" \
    "$@"
