#!/usr/bin/env bash
# GRPO | Qwen3-8B | FSDP | Instruction-Following RLVR (IFEval)
#
# verl's examples/grpo_trainer/run_qwen3_8b_fsdp.sh with the SAME training configuration,
# but with two changes only:
#   (1) DATA   -> the open-instruct IF dataset allenai/IF_multi_constraints_upto5, loaded straight
#                 from the HF Hub (like open-instruct) via the if_rlvr/if_dataset.py custom dataset
#                 class -- NO local parquet conversion step. Same data as open-instruct's
#                 scripts/train/rlvr/valpy_if_grpo_fast.sh.
#   (2) REWARD -> the instruction-following verifier migrated from open-instruct, plugged in via
#                 verl's STABLE, public custom_reward_function hook (if_rlvr/if_reward_fn.py).
#                 This is orthogonal to verl's training internals: it works with any reward manager
#                 (naive/dapo/prime/...), any RL algorithm, and across verl versions. Reward =
#                 fraction of IFEval constraints satisfied, thinking section stripped before scoring.
# Plus an ENABLE_THINKING toggle for Qwen3.
#
# The reward code (if_rlvr/) is self-contained and has NO verl imports in its core; it does
# not modify any existing verl file. Training stays 100% verl-native.
#
# Prereqs:
#   conda activate verl
#   # verifier runtime deps on EVERY ray node (or shared NLTK_DATA): ~40% of constraints use nltk
#   # punkt_tab, ~15% langdetect. (The optional reward manager fails fast if missing; the reward
#   # function will raise on a node that lacks them.)
#   pip install langdetect immutabledict nltk datasets  # if_rlvr/requirements.txt
#   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
#   # The IF dataset is pulled from the HF Hub automatically on first run (cached under HF_HOME);
#   # no conversion step. For an offline/local-parquet alternative instead, see if_rlvr/convert_data.py.
#
# Run:
#   ENABLE_THINKING=true  bash if_rlvr/run_qwen3_8b_if_rlvr.sh
#   ENABLE_THINKING=false bash if_rlvr/run_qwen3_8b_if_rlvr.sh

set -xeuo pipefail

########################### runtime selection (non-invasive) ###########################
VERL_DIR=${VERL_DIR:-/lustre/justinseo/if-verl/verl}
export PYTHONPATH="${VERL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"     # use the if-verl/verl checkout
IF_RLVR_DIR=${IF_RLVR_DIR:-${VERL_DIR}/if_rlvr}          # self-contained; can live anywhere
REWARD_FN_PATH="${IF_RLVR_DIR}/if_reward_fn.py"
DATASET_CLS_PATH="${IF_RLVR_DIR}/if_dataset.py"          # loads the IF data straight from the HF Hub

########################### user-adjustable ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
MACHINE=${MACHINE:-}

ENABLE_THINKING=${ENABLE_THINKING:-true}                        # IF: Qwen3 thinking mode true|false
IF_VAL_SIZE=${IF_VAL_SIZE:-512}                                 # held-out eval examples (open-instruct sampled 16)
IF_DATA_SEED=${IF_DATA_SEED:-1}                                 # HF split shuffle seed (open-instruct used --seed 1)

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-}
NPUS_PER_NODE=${NPUS_PER_NODE:-}

train_batch_size=${TRAIN_BATCH_SIZE:-1024}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-256}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}                    # base value; ~98% of IF prompts fit (set 2048 to keep ~99%)
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-}
rollout_n=${ROLLOUT_N:-5}
sp_size=${SP_SIZE:-1}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_if_grpo_${INFER_BACKEND}_fsdp_think_${ENABLE_THINKING}}
########################### end user-adjustable ###########################

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
        rollout_tp=${rollout_tp:-2}
        rollout_gpu_mem_util=${rollout_gpu_mem_util:-0.6}
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
            *) NGPUS_PER_NODE=${NGPUS_PER_NODE:-8} ;;
        esac
        n_trainer_devices=${NGPUS_PER_NODE}
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        NPUS_PER_NODE=8
        n_trainer_devices=${NPUS_PER_NODE}
        actor_param_offload=True
        actor_optimizer_offload=True
        rollout_tp=${rollout_tp:-4}
        sp_size=4
        train_batch_size=16
        max_prompt_length=$((1024 * 2))
        max_response_length=$((1024 * 32))
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
    # Load allenai/IF_multi_constraints_upto5 straight from the HF Hub via the custom dataset
    # class (like open-instruct) -- no parquet conversion. train_files/val_files are split
    # SELECTOR tokens (not paths); IFMultiConstraintsDataset reads them to pick train vs eval.
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
    data.truncation='error'
    data.prompt_key=prompt
    data.reward_fn_key=data_source
    "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}"
)

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
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# IF (ifeval) reward via verl's STABLE custom_reward_function hook (orthogonal; default reward
# manager = naive). Reward = fraction of constraints satisfied (thinking stripped). To scale like
# open-instruct (10.0; normalization-invariant under GRPO) add:
#   +custom_reward_function.reward_kwargs.verification_reward=10.0
# To instead reproduce the exact non-stop/truncation penalty, swap these two lines for the optional
# (verl-coupled) reward manager:
#   reward.reward_manager.source=importlib  reward.reward_manager.name=IFRewardManager
#   reward.reward_manager.module.path=${IF_RLVR_DIR}/if_reward_manager.py  reward.reward_model.enable=False
REWARD=(
    custom_reward_function.path="${REWARD_FN_PATH}"
    custom_reward_function.name=compute_score
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
    "$@"
