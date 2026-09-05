#!/usr/bin/env bash
# KL-penalty ablation -- Qwen3-1.7B, non-reasoning, constraint-only IF reward, GRPO.
#
# PURPOSE. Hold everything constant and sweep ONE knob: actor.kl_loss_coef, the coefficient on the
# low-var KL(policy || reference) term added to the policy loss. Four arms, run back to back on all
# 8 GPUs, so the training-dynamics curves (entropy, kl_loss, grad_norm, reward, response_length,
# clip fractions) are directly comparable across KL strengths.
#
#   arm   multiplier   kl_loss_coef        wandb experiment_name
#   ---------------------------------------------------------------------------------------------
#    1        2x          0.002            qwen3_17b_grpo_nonthink_constonly_klabl_x02_kl0p002
#    2        5x          0.005            qwen3_17b_grpo_nonthink_constonly_klabl_x05_kl0p005
#    3       10x          0.01             qwen3_17b_grpo_nonthink_constonly_klabl_x10_kl0p01
#    4       20x          0.02             qwen3_17b_grpo_nonthink_constonly_klabl_x20_kl0p02
#
# The 1x control is the existing run `qwen3_17b_grpo_nonthink_constraint_only_b1024_c1`
# (kl_loss_coef=0.001). It was trained on 4 GPUs for 5 epochs with a random dataloader order and no
# in-training validation; the optimization is identical but those three differences make it an
# imperfect control. To get a matched 1x arm, just add it to the sweep:
#
#   KL_MULTS="1 2 5 10 20" bash if_rlvr/exps/bidirectional/run_qwen3_17b_kl_ablation.sh
#
# ---------------------------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------------------------
#   bash if_rlvr/exps/bidirectional/run_qwen3_17b_kl_ablation.sh          # all 4 arms, in order
#   KL_MULTS="10 20" bash .../run_qwen3_17b_kl_ablation.sh                # a subset
#   KL_LOSS_COEFS="0.003 0.03" bash .../run_qwen3_17b_kl_ablation.sh      # explicit coefficients
#   DRY_RUN=1 bash .../run_qwen3_17b_kl_ablation.sh                       # print, do not launch
#   ON_FAILURE=continue bash .../run_qwen3_17b_kl_ablation.sh             # do not stop the sweep
#   SAVE_HF_MODEL=true bash .../run_qwen3_17b_kl_ablation.sh              # +8GB/ckpt, HF format
#
# WANDB_API_KEY must be set (trainer.logger includes wandb).
# Stop a running sweep with:  bash if_rlvr/exps/bidirectional/stop_qwen3_17b_kl_ablation.sh
#
# Every arm self-resumes (trainer.resume_mode=auto, IF_MAX_RETRIES=10), and an arm that already
# reached its last step exits immediately, so re-running this script after an interruption picks up
# where it stopped instead of redoing finished work.
#
# ---------------------------------------------------------------------------------------------
# WHY THESE NUMBERS
# ---------------------------------------------------------------------------------------------
# HELD FIXED (identical to qwen3_17b_constraint_only_nonreason.sh, the script being ablated):
#   Qwen/Qwen3-1.7B, enable_thinking=false, constraint-only reward (no anchor/PPL path),
#   train_batch_size=1024, ppo_mini_batch_size=1024 (=> ppo_epochs=1, one on-policy gradient step
#   per batch), actor lr 5e-7, prompt/response budget 2048/2048, rollout n=8, entropy_coeff=0,
#   kl_loss_type=low_var_kl, algorithm.use_kl_in_reward=False.
#
#   Sampling is left at the verl defaults -- temperature 1.0, top_p 1.0, top_k -1, i.e. UNTRUNCATED.
#   Do not "improve" this to top_p 0.95 / top_k 20. The Qwen3-4B reasoning run in this repo used
#   truncated sampling (forced on it by the anchor-cache metadata contract) and diverged at step
#   ~190 of 273; truncated sampling sustained over hundreds of steps is the leading suspect, since
#   vLLM samples from truncate(pi) while the actor differentiates the full 151,936-token softmax.
#   This ablation is precisely about the term that opposes that drift, so the sampler must stay
#   untruncated and identical to the run being ablated.
#
# CHANGED BY REQUEST:
#   TOTAL_EPOCHS=4 (was 5).  93,881 train rows / 1024 = 91 steps/epoch -> 364 steps per arm.
#   (95,373 HF rows - 512 val - 980 prompts over the 2048-token budget = 93,881; measured with the
#   Qwen3-1.7B chat template at enable_thinking=false, 98.97% retention.)
#   SAVE_FREQ=91 therefore lands exactly one checkpoint per epoch: 91 / 182 / 273 / 364.
#
# CHANGED FOR THIS HARDWARE (8x H100 80GB; all of these are throughput/memory knobs with NO effect
# on the optimization, so they do not compromise comparability with the 4-GPU reference run):
#   NGPUS_PER_NODE=8, GPU_SET=0..7      the reference used 4 GPUs; the global batch is unchanged
#                                       (1024 prompts x 8 samples), FSDP just shards it 8 ways and
#                                       `token-mean` loss aggregation normalizes by a globally
#                                       all-reduced token count, so the gradient is bit-for-bit the
#                                       same objective. Roughly halves wall clock.
#   ROLLOUT_TP=1                        a 1.7B model needs no tensor parallelism. TP=1 gives 8-way
#                                       rollout data parallelism, zero TP collectives, and a full
#                                       KV cache per replica.
#   rollout.max_model_len=4096          = 2048 + 2048 exactly. MUST be set: verl leaves it null, so
#                                       vLLM falls back to max_position_embeddings, which is 40960
#                                       for Qwen3-1.7B -- 10x the block-table and bookkeeping width
#                                       this run can ever use.
#   rollout.max_num_seqs=512            Qwen3-1.7B KV cache is 2*28 layers*8 kv heads*128 head_dim
#                                       *2 bytes = 112 KiB/token. At gpu_memory_utilization 0.85
#                                       (69.4GB of 81.6GB) minus ~4.1GB bf16 weights and ~4GB of
#                                       engine/activation overhead, that is ~61GB ~= 530k tokens of
#                                       KV per replica. Each replica sees 1024 requests (8192/8);
#                                       with prefix caching the 8 samples of a prompt share its
#                                       prompt blocks, so 512 concurrent sequences need only
#                                       ~512*(279/8 + 450) ~= 250k tokens. 512 keeps the decode
#                                       batch large enough to saturate a 1.7B model while leaving
#                                       enough KV that the long tail does not thrash on preemption;
#                                       admitting all 1024 at once (the verl default) would.
#   ROLLOUT_GPU_MEM_UTIL=0.85           the reference used 0.90. vLLM's budget is an absolute
#                                       fraction of TOTAL memory measured at engine init, so 0.90
#                                       leaves ~4.8GB for the training process's resident fp32 FSDP
#                                       shards + Adam state (~3.4GB/GPU here). We are nowhere near
#                                       KV-bound at 512 seqs, so the last 4GB buys nothing and 0.85
#                                       restores ~9GB of headroom instead. On the Qwen3-4B run 0.90
#                                       measured 81.0GB of 81.6GB during rollout.
#   PPO_MAX_TOKEN_LEN_PER_GPU=32768     the base launcher defaults to 98304. At the 151,936-token
#                                       vocab a microbatch costs ~0.283 MiB/token in bf16 logits
#                                       and the same again in the logits gradient, which is
#                                       model-size-independent: 98304 tokens is ~60GB of logits
#                                       alone. 32768 is the value measured safe at 4B on this box,
#                                       and 1.7B is strictly smaller. It costs nothing measurable:
#                                       ~717k tokens/GPU/step means 22 microbatches instead of 7,
#                                       and each extra FSDP all-gather + reduce-scatter for a 1.7B
#                                       model is ~15ms over NVLink -- ~0.3s against a step of
#                                       minutes. LOG_PROB_MAX_TOKEN_LEN_PER_GPU inherits it.
#   AGENT_NUM_WORKERS=256               the base default is 32. Each AgentLoopWorker is a Ray actor
#                                       that also runs the IFEval verifier (compute_score) for its
#                                       chunk, and that is CPU-bound Python. 1024 prompts / 256
#                                       workers = 4 prompts each on a 224-core box.
#   DATA_PROCESSOR_CPU_COUNT=32         one-off prompt-length filtering at startup.
#
# ADDED FOR ABLATION HYGIENE (again, no effect on the objective):
#   data.seed=1                         verl leaves data.seed null, which seeds the dataloader's
#                                       RandomSampler from entropy -- every arm would see a
#                                       DIFFERENT batch order, adding variance to the one thing the
#                                       sweep is trying to isolate. Pinning it makes all arms walk
#                                       the dataset identically. (vLLM sampling is still
#                                       nondeterministic under continuous batching; this removes
#                                       the largest controllable confound, not all of them.)
#   TEST_FREQ=91 + val_before_train     one validation pass per epoch plus a step-0 baseline, on the
#                                       512-row held-out split. val_kwargs is greedy (do_sample
#                                       false, temperature 0, n=1), computes no gradient and costs
#                                       ~a minute, so it is free relative to a 364-step run -- and
#                                       val_before_train catches a broken reward/verifier path
#                                       before an arm burns an epoch. The reference run had
#                                       test_freq=1000, i.e. never.
#   IF_THINK_END_TOKEN=none             disables the think/* reasoning-vs-answer split metrics. This
#                                       is a NON-reasoning run: `</think>` never appears in a
#                                       response, so the split would report every rollout as 100%
#                                       reasoning tokens and 0% answer tokens.
#
# DISK. Each checkpoint is fp32 FSDP model shards (~8.1GB) + Adam state (~16.2GB) + a little extra
# ~= 25GB, and there are 4 per arm. 4 arms x 4 epochs x 25GB ~= 400GB. SAVE_HF_MODEL=true adds an
# fp32 HF-format export (~8GB each, ~530GB total); it is off by default because the FSDP shards can
# be converted after the fact:
#   python3 -m verl.model_merger merge --backend fsdp --tie-word-embedding \
#       --local_dir checkpoints/verl_if_rlvr/<experiment>/global_step_364/actor \
#       --target_dir <output>
# ---------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_DIR=${VERL_DIR:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}
cd "${VERL_DIR}"

BASE_LAUNCHER="${SCRIPT_DIR}/qwen3_4b_01_00_const1_ref_anchor_reasoning.sh"
STOP_SCRIPT="${SCRIPT_DIR}/stop_qwen3_17b_kl_ablation.sh"

DRY_RUN=${DRY_RUN:-0}
ON_FAILURE=${ON_FAILURE:-stop}     # stop | continue
ALLOW_LOW_DISK=${ALLOW_LOW_DISK:-0}

########################### the swept knob ###########################
KL_BASE=${KL_BASE:-0.001}
KL_MULTS=${KL_MULTS:-"2 5 10 20"}
# KL_LOSS_COEFS, if set, wins over KL_MULTS and is used verbatim.
KL_LOSS_COEFS=${KL_LOSS_COEFS:-}

########################### held fixed ###########################
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B}
export IF_REF_VLLM_MODEL=${IF_REF_VLLM_MODEL:-${MODEL_PATH}}
export ENABLE_THINKING=${ENABLE_THINKING:-false}
export IF_APPLY_ENABLE_THINKING_KWARG=${IF_APPLY_ENABLE_THINKING_KWARG:-true}
export IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}

export IF_DATA_SEED=${IF_DATA_SEED:-1}
export IF_VAL_SIZE=${IF_VAL_SIZE:-512}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
export ACTOR_LR=${ACTOR_LR:-5e-7}
export ENTROPY_COEFF=${ENTROPY_COEFF:-0}
export ROLLOUT_N=${ROLLOUT_N:-8}
export PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}

export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
export SAVE_FREQ=${SAVE_FREQ:-91}
export TEST_FREQ=${TEST_FREQ:-91}

# Constraint-only reward: every anchor/PPL path off.
export PY_GIVEN_X_REWARD_COEFF=${PY_GIVEN_X_REWARD_COEFF:-0.0}
export PX_GIVEN_Y_REWARD_COEFF=${PX_GIVEN_Y_REWARD_COEFF:-0.0}
export IF_REF_ANCHOR_PRECOMPUTE=${IF_REF_ANCHOR_PRECOMPUTE:-false}
export IF_REF_POLICY_ANCHOR_PPL=${IF_REF_POLICY_ANCHOR_PPL:-false}
export IF_REF_PPL_GATE=${IF_REF_PPL_GATE:-false}
export IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE=${IF_REF_ANCHOR_SKIP_MISSING_PRECOMPUTE:-true}
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=${IF_REF_ANCHOR_TRAIN_CACHED_ONLY:-false}

########################### hardware / throughput ###########################
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
export GPU_SET=${GPU_SET:-0,1,2,3,4,5,6,7}
export RUN_SLOT=${RUN_SLOT:-0}
export IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-22000}
# MUST stay BELOW /proc/sys/net/ipv4/ip_local_port_range (32768-60999 on this box). The other
# launchers in this repo use 40000, which is INSIDE that range, and it fails: this run starts ~400
# Ray processes (8 workers + 256 AgentLoopWorkers + the rest), each binding a listening socket on a
# random ephemeral port, and some of them land on the ports vLLM then wants. _reserve_port_block()
# (vllm_async_server.py) needs VLLM_RESERVED_PORT_COUNT=16 CONSECUTIVE free ports inside each
# replica's VLLM_PORT_STRIDE=100 window, so ~6 scattered squatters in one window is fatal:
#     INFO  Shifted vLLM replica 1 port block from 40100-40115 to 40110-40125 ... was busy
#     OSError: No free vLLM port block of 16 ports in replica slot 40200-40299
# That is a dice roll at every engine start -- it would hit a later arm just as easily as the first.
# 8 replicas x stride 100 -> 25000-25799 per slot, clear of the ephemeral range and of the
# 22000-22100 Ray worker-group range.
export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-$((25000 + RUN_SLOT * 1000))}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

export ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
# 2x the actor cap, NOT the usual "same as the actor". old_log_prob and ref are forward-only under
# no_grad, and verl's logprobs_from_logits uses flash-attn's fused cross-entropy (available in this
# env), so neither materializes an fp32 full-vocab tensor and neither retains activations: measured
# at 32768 the whole step peaks at 44.0GB allocated / 47.5GB reserved, and that peak is
# update_actor's BACKWARD, not these passes. Doubling their microbatch halves the number of
# forward passes, and they are overhead-bound rather than FLOP-bound -- 786k tokens/GPU/step of
# forward at 25.9s is only ~123 TFLOPS on a card that does ~600 on this shape.
export LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((2 * PPO_MAX_TOKEN_LEN_PER_GPU))}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-256}
export DATA_PROCESSOR_CPU_COUNT=${DATA_PROCESSOR_CPU_COUNT:-32}

ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}

########################### logging / caches ###########################
export PROJECT_NAME=${PROJECT_NAME:-verl_if_rlvr}
# verl reads the wandb entity ONLY from this env var (verl/utils/tracking.py). This account's
# default entity is a personal one, so without it runs land in <user>/verl_if_rlvr instead of
# ifif/verl_if_rlvr -- same project name, different namespace, easy to miss.
export WANDB_ENTITY=${WANDB_ENTITY:-ifif}
# Non-reasoning run: turn the think/* reasoning-vs-answer split metrics off (see header).
export IF_THINK_END_TOKEN=${IF_THINK_END_TOKEN:-none}
# Pin the HF cache to the repo's own .cache. Without this an ambient HF_HOME wins, which on this
# box points at a filesystem holding none of the weights -- a re-download onto the wrong volume.
export HF_HOME=${KL_ABL_HF_HOME:-${VERL_DIR}/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}

SAVE_HF_MODEL=${SAVE_HF_MODEL:-false}
LOG_DIR=${LOG_DIR:-${VERL_DIR}/logs/qwen3_17b_kl_ablation}
NAME_PREFIX=${NAME_PREFIX:-qwen3_17b_grpo_nonthink_constonly_klabl}

########################### resolve the arm list ###########################
declare -a COEFS=()
declare -a TAGS=()

kl_tag() {   # 0.002 -> kl0p002
    local coef=$1
    echo "kl${coef//./p}"
}

mult_tag() { # 2 -> x02, 10 -> x10, 2.5 -> x2p5
    awk -v m="$1" 'BEGIN{
        if (m == int(m) && m < 10)      printf "x0%d", m;
        else if (m == int(m))           printf "x%d", m;
        else { s = sprintf("%g", m); gsub(/\./, "p", s); printf "x%s", s }
    }'
}

if [[ -n "${KL_LOSS_COEFS}" ]]; then
    for coef in ${KL_LOSS_COEFS}; do
        COEFS+=("${coef}")
        TAGS+=("$(kl_tag "${coef}")")
    done
else
    for mult in ${KL_MULTS}; do
        coef=$(awk -v b="${KL_BASE}" -v m="${mult}" 'BEGIN{printf "%.6g", b * m}')
        COEFS+=("${coef}")
        TAGS+=("$(mult_tag "${mult}")_$(kl_tag "${coef}")")
    done
fi

if (( ${#COEFS[@]} == 0 )); then
    echo "[klabl] no arms resolved; set KL_MULTS or KL_LOSS_COEFS" >&2
    exit 2
fi

########################### preflight ###########################
echo "=============================================================================="
echo "[klabl] Qwen3-1.7B non-reasoning KL-coefficient ablation"
echo "[klabl] model=${MODEL_PATH}  epochs=${TOTAL_EPOCHS}  save_freq=${SAVE_FREQ}  test_freq=${TEST_FREQ}"
echo "[klabl] batch=${TRAIN_BATCH_SIZE}x${ROLLOUT_N}  lr=${ACTOR_LR}  gpus=${GPU_SET}"
echo "[klabl] arms (${#COEFS[@]}), in order:"
for i in "${!COEFS[@]}"; do
    printf '[klabl]   %d. kl_loss_coef=%-8s -> %s_%s\n' \
        "$((i + 1))" "${COEFS[$i]}" "${NAME_PREFIX}" "${TAGS[$i]}"
done
echo "=============================================================================="

if [[ "${DRY_RUN}" != "1" ]]; then
    : "${WANDB_API_KEY:?WANDB_API_KEY is not set (trainer.logger includes wandb)}"
    export WANDB_API_KEY
fi

if [[ ! -x "${BASE_LAUNCHER}" && ! -f "${BASE_LAUNCHER}" ]]; then
    echo "[klabl] base launcher missing: ${BASE_LAUNCHER}" >&2
    exit 2
fi

# Disk: ~25GB per checkpoint (fp32 FSDP shards + Adam), +8GB with the HF export, 4 per arm.
per_ckpt_gb=25
case "${SAVE_HF_MODEL}" in 1 | true | TRUE | yes | YES | on | ON) per_ckpt_gb=33 ;; esac
steps_per_epoch=${STEPS_PER_EPOCH:-91}   # 93,881 rows / train_batch_size 1024
ckpts_per_arm=$(( (TOTAL_EPOCHS * steps_per_epoch + SAVE_FREQ - 1) / SAVE_FREQ ))
need_gb=$(( ${#COEFS[@]} * ckpts_per_arm * per_ckpt_gb * 11 / 10 ))
avail_gb=$(df -BG --output=avail "${VERL_DIR}" | tail -1 | tr -dc '0-9')
echo "[klabl] checkpoints: ${#COEFS[@]} arms x ${ckpts_per_arm} x ~${per_ckpt_gb}GB -> need ~${need_gb}GB, have ${avail_gb}GB"
if (( avail_gb < need_gb )) && [[ "${ALLOW_LOW_DISK}" != "1" ]]; then
    echo "[klabl] not enough free disk. Free some space, lower TOTAL_EPOCHS/arms, or set ALLOW_LOW_DISK=1." >&2
    exit 2
fi

# Ports: the whole vLLM window must sit below the kernel's ephemeral range, or Ray's own workers
# squat on it (see the VLLM_MASTER_PORT_BASE comment). Fail here rather than 3 minutes into an
# engine start with "No free vLLM port block".
vllm_port_hi=$(( VLLM_MASTER_PORT_BASE + NGPUS_PER_NODE * VLLM_PORT_STRIDE - 1 ))
eph_lo=$(awk '{print $1}' /proc/sys/net/ipv4/ip_local_port_range 2>/dev/null || echo 32768)
echo "[klabl] vllm ports ${VLLM_MASTER_PORT_BASE}-${vllm_port_hi} (ephemeral range starts at ${eph_lo})"
if (( vllm_port_hi >= eph_lo )); then
    echo "[klabl] WARNING: the vLLM port window overlaps the ephemeral range (>= ${eph_lo}); Ray workers" >&2
    echo "[klabl]          will randomly occupy it and engine startup will fail intermittently." >&2
    echo "[klabl]          Set VLLM_MASTER_PORT_BASE so that base + ${NGPUS_PER_NODE}*${VLLM_PORT_STRIDE} < ${eph_lo}." >&2
fi
occupied=$(ss -tlnH 2>/dev/null | awk -v lo="${VLLM_MASTER_PORT_BASE}" -v hi="${vllm_port_hi}" \
    '{split($4,a,":"); p=a[length(a)]+0; if (p>=lo && p<=hi) c++} END{print c+0}')
if (( occupied > 0 )); then
    echo "[klabl] WARNING: ${occupied} listener(s) already inside ${VLLM_MASTER_PORT_BASE}-${vllm_port_hi}." >&2
fi

busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024' | wc -l)
if (( busy > 0 )); then
    echo "[klabl] WARNING: ${busy} GPU(s) already hold >1GB. Run stop_qwen3_17b_kl_ablation.sh first." >&2
    if [[ "${DRY_RUN}" != "1" ]]; then
        exit 2
    fi
fi

mkdir -p "${LOG_DIR}"

########################### hydra overrides shared by every arm ###########################
# Everything here is either not exposed as an env knob by the base launcher, or is set there to a
# value this sweep deliberately overrides. Passed last, so they win.
HYDRA=(
    "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.free_cache_engine=True"
    "data.seed=${IF_DATA_SEED}"
    "trainer.val_before_train=True"
)
case "${SAVE_HF_MODEL}" in
    1 | true | TRUE | yes | YES | on | ON)
        HYDRA+=("actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]")
        ;;
esac

########################### run the arms, one at a time ###########################
SWEEP_START=${SECONDS}
declare -a RESULTS=()
failed=0

for i in "${!COEFS[@]}"; do
    coef=${COEFS[$i]}
    tag=${TAGS[$i]}
    exp="${NAME_PREFIX}_${tag}"
    run_id="qwen3_17b_klabl_${tag}_slot${RUN_SLOT}"
    log="${LOG_DIR}/${tag}_$(date +%Y%m%d_%H%M%S).log"

    echo
    echo "=============================================================================="
    echo "[klabl] arm $((i + 1))/${#COEFS[@]}  kl_loss_coef=${coef}"
    echo "[klabl]   experiment : ${exp}"
    echo "[klabl]   run_id     : ${run_id}"
    echo "[klabl]   checkpoints: checkpoints/${PROJECT_NAME}/${exp}"
    echo "[klabl]   log        : ${log}"
    echo "=============================================================================="

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[klabl] DRY_RUN: would run"
        echo "  KL_LOSS_COEF=${coef} EXPERIMENT_NAME=${exp} IF_RLVR_RUN_ID=${run_id} \\"
        echo "    bash ${BASE_LAUNCHER} \\"
        printf '      %s \\\n' "${HYDRA[@]}"
        echo
        continue
    fi

    arm_start=${SECONDS}
    set +e
    env \
        KL_LOSS_COEF="${coef}" \
        EXPERIMENT_NAME="${exp}" \
        IF_RLVR_RUN_ID="${run_id}" \
        bash "${BASE_LAUNCHER}" "${HYDRA[@]}" 2>&1 | tee "${log}"
    rc=${PIPESTATUS[0]}
    set -e
    arm_mins=$(( (SECONDS - arm_start) / 60 ))

    if (( rc == 0 )); then
        echo "[klabl] arm $((i + 1)) (${tag}) finished cleanly in ${arm_mins} min."
        RESULTS+=("OK    ${tag}  kl=${coef}  ${arm_mins}min  ${log}")
    else
        echo "[klabl] arm $((i + 1)) (${tag}) FAILED rc=${rc} after ${arm_mins} min (see ${log})." >&2
        RESULTS+=("FAIL  ${tag}  kl=${coef}  ${arm_mins}min  rc=${rc}  ${log}")
        failed=$((failed + 1))
    fi

    # Reap stragglers before the next arm. The base launcher's retry loop is already gone by now
    # (the pipeline returned), but a killed/crashed arm can leave vLLM engine cores or Ray workers
    # holding GPU memory and the run's port range, which the next arm would then fail to bind.
    QUIET=1 CLEAR_RAY_TMPDIR=1 RUN_SLOT="${RUN_SLOT}" bash "${STOP_SCRIPT}" || true
    for _ in $(seq 1 30); do
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1024' | wc -l)
        (( busy == 0 )) && break
        sleep 2
    done
    if (( busy > 0 )); then
        echo "[klabl] WARNING: ${busy} GPU(s) still hold >1GB after teardown." >&2
    fi

    if (( rc != 0 )) && [[ "${ON_FAILURE}" != "continue" ]]; then
        echo "[klabl] ON_FAILURE=stop: not starting the remaining arms." >&2
        break
    fi
done

########################### summary ###########################
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[klabl] DRY_RUN complete; nothing was launched."
    exit 0
fi

echo
echo "=============================================================================="
echo "[klabl] sweep finished in $(( (SECONDS - SWEEP_START) / 60 )) min"
for line in "${RESULTS[@]}"; do
    echo "[klabl]   ${line}"
done
echo "=============================================================================="
exit $(( failed > 0 ? 1 : 0 ))
