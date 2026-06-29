#!/usr/bin/env bash
# GRPO | Qwen3-4B | IF-RLVR | B200 speed-only profile | reasoning
#
# This is intentionally the same training setup as run_qwen3_4b_if_rlvr.sh.
# The only default changes are throughput knobs:
#   - TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE
#   - PPO_MAX_TOKEN_LEN_PER_GPU / LOG_PROB_MAX_TOKEN_LEN_PER_GPU
#   - ROLLOUT_GPU_MEM_UTIL
#
# Everything rollout-facing is left to the original verl defaults/script values:
# no Liger, no fused kernels, no forced FSDP dtype change, no rollout worker
# count change, no vLLM max_num_batched_tokens/max_num_seqs change, no changed
# validation cadence, and no changed checkpoint experiment name.

set -xeuo pipefail

CACHE_ROOT=/NHNHOME/WORKSPACE/26msit001_T_A/IFIF/if-rlvr/.cache
RUN_TMPDIR=${RUN_TMPDIR:-/var/tmp}
mkdir -p "$CACHE_ROOT"/{torchinductor,triton} "$RUN_TMPDIR"

export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$RUN_TMPDIR"

########################### NLTK data ###########################
NLTK_DATA_DIR=${NLTK_DATA_DIR:-${CACHE_ROOT}/nltk_data}
mkdir -p "${NLTK_DATA_DIR}"
export NLTK_DATA="${NLTK_DATA_DIR}"

python3 - <<'PY'
import os
import nltk

data_dir = os.environ["NLTK_DATA"]
os.makedirs(data_dir, exist_ok=True)

if data_dir not in nltk.data.path:
    nltk.data.path.insert(0, data_dir)

required = {
    "punkt_tab": "tokenizers/punkt_tab/english/",
    "punkt": "tokenizers/punkt/english.pickle",
}

for pkg, resource in required.items():
    try:
        nltk.data.find(resource)
        print(f"[NLTK] found {resource}")
    except LookupError:
        print(f"[NLTK] downloading {pkg} to {data_dir}")
        nltk.download(pkg, download_dir=data_dir, quiet=False, raise_on_error=True)
        nltk.data.find(resource)
PY
########################### end NLTK data ###########################

########################### runtime selection (non-invasive) ###########################
VERL_DIR=${VERL_DIR:-/NHNHOME/26msit001_A/IFIF/if-rlvr/}
export PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
IF_RLVR_DIR=${IF_RLVR_DIR:-${VERL_DIR}/if_rlvr}
REWARD_FN_PATH="${IF_RLVR_DIR}/if_reward_fn.py"
DATASET_CLS_PATH="${IF_RLVR_DIR}/if_dataset.py"

########################### user-adjustable ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MACHINE=${MACHINE:-}

ENABLE_THINKING=${ENABLE_THINKING:-true}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-${ENABLE_THINKING}}
IF_VAL_SIZE=${IF_VAL_SIZE:-512}
IF_DATA_SEED=${IF_DATA_SEED:-1}
DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-64}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
NPUS_PER_NODE=${NPUS_PER_NODE:-}

# Speed-only changes versus the original script:
# original: train_batch_size=256, ppo_mini_batch_size=256, token cap=65536.
train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-${train_batch_size}}
max_prompt_length=${MAX_PROMPT_LENGTH:-16384}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-98304}
log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-${ppo_max_token_len_per_gpu}}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}
py_given_x_reward_coeff=${PY_GIVEN_X_REWARD_COEFF:-${PPL_REWARD_COEFF:-0.1}}  # ##6/3 ppl## p(y|x)
if_ppl_reward_strategy=${IF_PPL_REWARD_STRATEGY:-anchor}  # ##6/3 ppl## rank|anchor
if_ppl_anchor_reward_mode=${IF_PPL_ANCHOR_REWARD_MODE:-both}  # ##6/3 ppl## both|no_lower_zero|lower_zero_only
if_ref_ppl_gate=${IF_REF_PPL_GATE:-false}  # unused in anchor mode; lower gate is handled by anchor reward
if_ref_ppl_gate_margin=${IF_REF_PPL_GATE_MARGIN:-0.0}
if_ref_anchor_precompute=${IF_REF_ANCHOR_PRECOMPUTE:-true}
if_ref_policy_anchor_ppl=${IF_REF_POLICY_ANCHOR_PPL:-true}
if_ref_anchor_precompute_batch_size=${IF_REF_ANCHOR_PRECOMPUTE_BATCH_SIZE:-4096}
agent_num_workers=${AGENT_NUM_WORKERS:-512}
if_ref_anchor_cache_path=${IF_REF_ANCHOR_CACHE_PATH:-${CACHE_ROOT}/if_ref_anchor_qwen3_4b_const1_train_seed${IF_DATA_SEED}_val${IF_VAL_SIZE}_thinkfalse.json}
if_ref_anchor_cache_metadata_strict=${IF_REF_ANCHOR_CACHE_METADATA_STRICT:-false}
if_ref_anchor_skip_missing_precompute=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-false}
if_ref_anchor_train_cached_only=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-${if_ref_anchor_skip_missing_precompute}}
export IF_REF_ANCHOR_CACHE_PATH="${if_ref_anchor_cache_path}"
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY="${if_ref_anchor_train_cached_only}"
export IF_REF_PPL_BASELINE=${IF_REF_PPL_BASELINE:-0}
export IF_REF_PPL_ANCHOR=${IF_REF_PPL_ANCHOR:-0}
export IF_REF_VLLM_BASE_URL=${IF_REF_VLLM_BASE_URL:-}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-Qwen/Qwen3-4B}
px_given_y_reward_coeff=${PX_GIVEN_Y_REWARD_COEFF:-0.0}  # ##6/3 ppl## p(x|y)
clipped_rollout_mode=${CLIPPED_ROLLOUT_MODE:-use}  # ##6/3 ppl## use|zero|drop

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.9}
rollout_n=${ROLLOUT_N:-8}
sp_size=${SP_SIZE:-1}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-25}
test_freq=${TEST_FREQ:-1000}

PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_4b_if_grpo_${INFER_BACKEND}_fsdp_think_${ENABLE_THINKING}_pyx_anchor_${py_given_x_reward_coeff}_pxy_${px_given_y_reward_coeff}_clip_${clipped_rollout_mode}_bsz_${train_batch_size}_const1only_refanchor_precompute_refpolicy_qwen3_4b}
WANDB_ENTITY=${WANDB_ENTITY:-ifif}
export WANDB_ENTITY
########################### end user-adjustable ###########################

########################### concurrent local run isolation ###########################
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_concurrent_run_env.sh"
########################### end concurrent local run isolation ###########################

########################### derived defaults ###########################
case "${DEVICE}" in
    gpu | npu) ;;
    *) echo "DEVICE must be gpu or npu, got: ${DEVICE}" >&2; exit 1 ;;
esac
if [ "${DEVICE}" = npu ] && [ "${INFER_BACKEND}" = trtllm ]; then
    echo "INFER_BACKEND=trtllm is only supported with DEVICE=gpu" >&2; exit 1
fi

EXTRA=()
case "${DEVICE}" in
    gpu)
        actor_param_offload=False
        actor_optimizer_offload=False
        rollout_tp=${rollout_tp:-1}
        case "${MACHINE}" in
            gb200)
                NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
                EXTRA+=(
                    actor_rollout_ref.rollout.enforce_eager=True
                    actor_rollout_ref.rollout.free_cache_engine=True
                    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
                    "+ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}"
                )
                if [ "${INFER_BACKEND}" = sglang ]; then
                    EXTRA+=(+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer)
                fi
                ;;
            *) NGPUS_PER_NODE=${NGPUS_PER_NODE:-4} ;;
        esac
        n_trainer_devices=${NGPUS_PER_NODE}
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        NPUS_PER_NODE=4
        n_trainer_devices=${NPUS_PER_NODE}
        actor_param_offload=True
        actor_optimizer_offload=True
        rollout_tp=${rollout_tp:-4}
        sp_size=4
        train_batch_size=16
        max_prompt_length=${MAX_PROMPT_LENGTH:-32768}
        max_response_length=${MAX_RESPONSE_LENGTH:-32768}
        ppo_mini_batch_size=16
        rollout_gpu_mem_util=0.3
        EXTRA+=(
            actor_rollout_ref.actor.use_torch_compile=False
            actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
            actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${sp_size}
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
            actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
        )
        if [ "${INFER_BACKEND}" = sglang ]; then
            EXTRA+=(+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=ascend)
        fi
        ;;
esac

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.custom_cls.path="${DATASET_CLS_PATH}"
    data.custom_cls.name=IFMultiConstraintsDataset
    data.train_files="['if_multi_train']"
    data.val_files="['if_multi_val']"
    "+data.if_dataset_val_size=${IF_VAL_SIZE}"
    "+data.if_dataset_seed=${IF_DATA_SEED}"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=${DATA_PROCESSOR_CPU_COUNT}
    data.truncation='error'
    data.prompt_key=prompt
    data.reward_fn_key=data_source
)

case "${IF_APPLY_ENABLE_THINKING_KWARG}" in
    1 | true | TRUE | yes | YES | on | ON)
        DATA+=("+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}")
        ;;
    0 | false | FALSE | no | NO | off | OFF)
        ;;
    *)
        echo "IF_APPLY_ENABLE_THINKING_KWARG must be true/false, got: ${IF_APPLY_ENABLE_THINKING_KWARG}" >&2
        exit 1
        ;;
esac

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu}
    actor_rollout_ref.rollout.agent.num_workers=${agent_num_workers}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    custom_reward_function.path="${REWARD_FN_PATH}"
    custom_reward_function.name=compute_score
    "+if_py_given_x_reward_coeff=${py_given_x_reward_coeff}"  # ##6/3 ppl## p(y|x)
    "+if_px_given_y_reward_coeff=${px_given_y_reward_coeff}"  # ##6/3 ppl## p(x|y)
    "+if_ppl_reward_strategy=${if_ppl_reward_strategy}"
    "+if_ppl_anchor_reward_mode=${if_ppl_anchor_reward_mode}"
    "+if_ref_anchor_precompute=${if_ref_anchor_precompute}"
    "+if_ref_policy_anchor_ppl=${if_ref_policy_anchor_ppl}"
    "+if_ref_anchor_precompute_batch_size=${if_ref_anchor_precompute_batch_size}"
    "+if_ref_anchor_cache_path=${if_ref_anchor_cache_path}"
    "+if_ref_anchor_cache_metadata_strict=${if_ref_anchor_cache_metadata_strict}"
    "+if_ref_anchor_skip_missing_precompute=${if_ref_anchor_skip_missing_precompute}"
    "+if_ref_ppl_gate=${if_ref_ppl_gate}"
    "+if_ref_ppl_gate_margin=${if_ref_ppl_gate_margin}"
    "+clipped_rollout_mode=${clipped_rollout_mode}"  # ##6/3 ppl##
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${n_trainer_devices}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.val_before_train=False
    trainer.resume_mode=${RESUME_MODE:-auto}
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${IF_RLVR_RAY_INIT_OVERRIDES[@]}" \
    "$@"
