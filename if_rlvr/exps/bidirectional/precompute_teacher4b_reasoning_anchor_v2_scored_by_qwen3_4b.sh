#!/usr/bin/env bash
# Build the v2 (rebuilt, hygiene-validated) Qwen3-4B REASONING anchor cache.
#
# This replaces if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.json,
# whose forensic audit (if_rlvr/docs/if_rlvr_anchor_reward_review_2026-08-11.md)
# found: 12.99% inverted pairs, 5.4% repetition-loop y1 (one ref1 was 29,842
# tokens of "apple apple apple..."), 1.7%/1.3% near-empty y1/y0, and 41.3% of
# rows carrying at least one structural defect. Sampling was t1.0/top_p1.0/
# top_k -1 with no degeneracy protection and a 16-32k generation budget vs the
# 8192-token training budget.
#
# Deltas vs precompute_teacher4b_reasoning_anchor_scored_by_qwen3_4b.sh, each
# tied to an audit finding:
#   1. SAMPLING t=1.0 top_p=0.95 top_k=20 presence_penalty=0.0
#      - p0.95+k20 measured on the Llama sibling: degenerate generations
#        12.5-16.4% -> ~0. presence_penalty stays 0.0 to MATCH TRAINING
#        rollouts (the old script's 1.5 postdates the old cache and would skew
#        NLLs; penalty is not recorded in metadata v3, hence the filename tag
#        and this comment are its provenance).
#   2. GENERATION BUDGET = TRAINING BUDGET (8192 incl. <think>), constant
#      across retry passes. The old 16k/32k ladder anchored on answers the
#      trained policy cannot produce (3.2% of old y1 exceeded 2048 tokens) and
#      a changing max_response_length would also flip metadata between passes.
#   3. IN-LOOP RETRIES for malformed draws only: empty/short final answers are
#      retried immediately (EMPTY_RESPONSE_RETRIES=2, MIN_TOKENS=10). No
#      compliance-based selection anywhere: ONE kept generation per side.
#   4. HYGIENE LOOP: after each build pass a validator DELETES rows whose y0/y1
#      is a repetition loop (rep-8gram mass > 0.5 or distinct-word ratio < 0.15
#      at n>=50), near-empty (<10 tokens), or byte-identical y0==y1; the next
#      pass regenerates exactly those rows. Capped at IF_ANCHOR_HYGIENE_PASSES;
#      survivors are reported, not silently kept-as-good.
#   5. Metadata v3 is written by the current trainer code (ppl_prefix_mode,
#      ppl_nll_scope, rollout_top_k included). The pilot that consumes this
#      cache MUST train with the same t/p/k -- with
#      IF_REF_ANCHOR_CACHE_METADATA_STRICT=true that match is machine-enforced.
#
# WHERE TO RUN: any box with the verl env + 4-8 GPUs. On the current machine
# the GPUs are occupied by the Tulu judge-only baseline until ~2026-08-14;
# override GPU_SET/NGPUS_PER_NODE for the B200 box as needed.
#
# Cost estimate: 93,882 rows x 2 generations x <=8192 thinking tokens + scoring;
# roughly 1-2 days on 8 H100-class GPUs, hygiene passes included (later passes
# regenerate only the deleted rows, so they are cheap).

set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
CACHE_ROOT=${CACHE_ROOT:-${VERL_DIR}/.cache}
mkdir -p "${CACHE_ROOT}"

export GPU_SET=${GPU_SET:-0,1,2,3,4,5,6,7}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export RUN_SLOT=${RUN_SLOT:-0}

export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-Qwen/Qwen3-4B}
export ENABLE_THINKING=${ENABLE_THINKING:-true}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-true}
export IF_ANCHOR_PRECOMPUTE_FINAL_ANSWER_ONLY=${IF_ANCHOR_PRECOMPUTE_FINAL_ANSWER_ONLY:-true}
# In-loop retries for malformed generations (delta 3).
export IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES=${IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES:-2}
export IF_ANCHOR_PRECOMPUTE_MIN_TOKENS=${IF_ANCHOR_PRECOMPUTE_MIN_TOKENS:-10}

# ---- sampling (delta 1) ----
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-20}
export PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}

# ---- budget (delta 2): matched to the training run, constant across passes ----
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1024}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-98304}
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-98304}
export ROLLOUT_N=${ROLLOUT_N:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-64}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-16}
export IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE=${IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE:-16384}
export IF_REF_ANCHOR_CACHE_SAVE_INTERVAL=${IF_REF_ANCHOR_CACHE_SAVE_INTERVAL:-512}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

export IF_DATA_SEED=${IF_DATA_SEED:-1}
export IF_VAL_SIZE=${IF_VAL_SIZE:-512}
export IF_DATASET_HF=${IF_DATASET_HF:-allenai/IF_multi_constraints_upto5}
# Filename encodes every generation knob metadata v3 cannot hold (pp0 = presence_penalty 0.0).
export IF_REF_ANCHOR_CACHE_PATH=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.json}
# strict=false ONLY so retry passes can resume the partial cache; the finished
# cache's metadata is written by the same code the pilot compares against.
export IF_REF_ANCHOR_CACHE_METADATA_STRICT=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-false}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}

export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-true}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-true}
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.1}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}

export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:--1}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
export IF_MAX_RETRIES=${IF_MAX_RETRIES:-1}

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_precompute_teacher4b_reasoning_anchor_v2_p095_k20_r8192}

PYBIN=${PYBIN:-python3}

# Completeness gate: complete = y0 & y1 present, positive token counts, finite NLLs.
cache_summary() {
    "${PYBIN}" - "${IF_REF_ANCHOR_CACHE_PATH}" "${IF_DATASET_HF}" "${IF_VAL_SIZE}" <<'PY'
import json, math, os, sys
path, dataset_name, val_size_s = sys.argv[1], sys.argv[2], sys.argv[3]
val_size = int(val_size_s)
expected = os.getenv("IF_REF_ANCHOR_EXPECTED_TOTAL", "").strip()
if expected:
    total = int(expected)
else:
    import datasets
    total = max(len(datasets.load_dataset(dataset_name, split="train")) - val_size, 0)
items = {}
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f).get("items", {})
complete = 0
for item in items.values():
    try:
        c0 = int(item.get("ref0_token_count", 0) or 0); c1 = int(item.get("ref1_token_count", 0) or 0)
        n0 = float(item.get("ref0_nll", float("inf"))); n1 = float(item.get("ref1_nll", float("inf")))
    except (TypeError, ValueError):
        continue
    if item.get("y0") and item.get("y1") and c0 > 0 and c1 > 0 and math.isfinite(n0) and math.isfinite(n1):
        complete += 1
missing = max(total - complete, 0)
print(f"complete={complete} total={total} missing={missing} path={path}")
sys.exit(0 if complete >= total else 1)
PY
}

# Hygiene validator (delta 4): DELETE bad rows so the next pass regenerates them.
# Prints "deleted=N kept_bad=M" on stdout; exit 0 always (flow control is on counts).
hygiene_pass() {
    "${PYBIN}" - "${IF_REF_ANCHOR_CACHE_PATH}" "${MODEL_PATH}" <<'PY'
import json, sys, os
from collections import Counter
path, model = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print("deleted=0 kept_bad=0 (no cache yet)"); sys.exit(0)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model)
with open(path, "r", encoding="utf-8") as f:
    payload = json.load(f)
items = payload.get("items", {})

def words(ids):
    try:
        return tok.decode(list(ids), skip_special_tokens=True).split()
    except Exception:
        return []

def is_loop(ids):
    w = words(ids)
    n = len(w)
    if n < 50:
        return False
    distinct = len(set(w)) / n
    if distinct < 0.15:
        return True
    grams = [" ".join(w[i:i+8]) for i in range(n - 7)]
    if not grams:
        return False
    top = Counter(grams).most_common(1)[0][1]
    return top / len(grams) > 0.5

reasons = Counter()
bad_keys = []
for key, item in items.items():
    y0, y1 = item.get("y0") or [], item.get("y1") or []
    why = None
    if len(y0) < 10: why = "y0_short"
    elif len(y1) < 10: why = "y1_short"
    elif list(y0) == list(y1): why = "y0_eq_y1"
    elif is_loop(y1): why = "y1_loop"
    elif is_loop(y0): why = "y0_loop"
    if why:
        bad_keys.append(key); reasons[why] += 1
for key in bad_keys:
    del items[key]
if bad_keys:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
print(f"deleted={len(bad_keys)} kept_bad=0 breakdown={dict(reasons)}")
PY
}

run_build_pass() {
    set +e
    bash "${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh" \
        +if_ref_anchor_precompute_only=true \
        +if_ref_anchor_cache_save_interval="${IF_REF_ANCHOR_CACHE_SAVE_INTERVAL}" \
        actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
        actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
        actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
        actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
        trainer.logger='["console"]' \
        "$@"
    local rc=$?
    set -e
    return "${rc}"
}

echo "[anchor v2] target cache: ${IF_REF_ANCHOR_CACHE_PATH}"
echo "[anchor v2] sampling: t=${ROLLOUT_TEMPERATURE} top_p=${ROLLOUT_TOP_P} top_k=${ROLLOUT_TOP_K} presence_penalty=${PRESENCE_PENALTY} budget=${MAX_RESPONSE_LENGTH}"

max_hygiene_passes=${IF_ANCHOR_HYGIENE_PASSES:-3}
pass_idx=1
while true; do
    echo "[anchor v2] ===== build pass ${pass_idx} ====="
    run_build_pass "$@" || echo "[anchor v2] build pass ${pass_idx} exited nonzero; continuing with saved rows"

    if ! summary=$(cache_summary); then
        echo "[anchor v2] cache incomplete after pass ${pass_idx}: ${summary}; regenerating missing rows"
        pass_idx=$((pass_idx + 1))
        if (( pass_idx > max_hygiene_passes + 3 )); then
            echo "[anchor v2] too many passes without completeness; aborting" >&2
            exit 1
        fi
        continue
    fi
    echo "[anchor v2] completeness OK: ${summary}"

    if (( pass_idx > max_hygiene_passes )); then
        echo "[anchor v2] hygiene pass budget exhausted; running FINAL report only"
        hygiene_report=$(hygiene_pass) || true
        echo "[anchor v2] FINAL residual (deleted in report run -- will stay missing): ${hygiene_report}"
        echo "[anchor v2] NOTE: rows deleted by the final report are absent from the cache;"
        echo "[anchor v2] TRAIN_CACHED_ONLY=true in the pilot simply excludes them."
        break
    fi

    hygiene_out=$(hygiene_pass)
    echo "[anchor v2] hygiene pass ${pass_idx}: ${hygiene_out}"
    deleted=$(sed -n 's/.*deleted=\([0-9]*\).*/\1/p' <<< "${hygiene_out}")
    if [[ "${deleted:-0}" == "0" ]]; then
        echo "[anchor v2] cache complete and clean after pass ${pass_idx}"
        break
    fi
    echo "[anchor v2] ${deleted} rows deleted for regeneration"
    pass_idx=$((pass_idx + 1))
done

final=$(cache_summary) || true
echo "[anchor v2] DONE: ${final}"
echo "[anchor v2] Reminder: the pilot must train with t=${ROLLOUT_TEMPERATURE}/top_p=${ROLLOUT_TOP_P}/top_k=${ROLLOUT_TOP_K}"
echo "[anchor v2] (IF_REF_ANCHOR_CACHE_METADATA_STRICT=true will enforce it)."
