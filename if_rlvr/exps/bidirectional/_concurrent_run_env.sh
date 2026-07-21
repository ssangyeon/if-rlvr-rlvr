#!/usr/bin/env bash
# Shared launch isolation for running multiple local IF-RLVR jobs on one host.
#
# Preferred usage on an 8-GPU host:
#   GPU_SLOT=0 bash qwen3_4b_01_01_const1.sh   # uses GPUs 0,1,2,3
#   GPU_SLOT=1 bash qwen3_4b_01_00_const1.sh   # uses GPUs 4,5,6,7
#
# Advanced overrides:
#   GPU_SET=2,3,6,7         explicit CUDA_VISIBLE_DEVICES value
#   IF_RLVR_PORT_BASE=24000 explicit base for all per-run control ports

if [[ -n "${IF_RLVR_CONCURRENT_ENV_SH:-}" ]]; then
    return 0
fi
IF_RLVR_CONCURRENT_ENV_SH=1

# --- Ray node-death tolerance -------------------------------------------------
# Root cause of the intermittent crashes (early AND late steps): this Docker container
# has a 1600 GiB cgroup memory limit, but Ray misreads it as the host's 2.2 TB and sets
# its OOM threshold ABOVE the cgroup cap -- so it never protects. Aggregate pressure from
# the concurrently-running jobs drives the container to its ceiling (memory.peak ==
# memory.max, hit 7792x). At the ceiling the kernel thrashes on reclaim (no OOM-kill:
# oom_kill=0), and the stalls starve the raylet's heartbeat -> GCS marks the node dead ->
# raylet fate-shares with its agents (clean exit 0) -> the whole job dies. Widen the
# health-check window from ~seconds to ~minutes so a reclaim stall is ridden out instead
# of being fatal. Override any of these per-run if needed.
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-20}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-60000}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-10000}
export RAY_health_check_initial_delay_ms=${RAY_health_check_initial_delay_ms:-30000}
# -----------------------------------------------------------------------------

_if_rlvr_count_csv_items() {
    local csv=${1:-}
    if [[ -z "${csv}" ]]; then
        echo 0
        return
    fi
    local compact=${csv//[[:space:]]/}
    awk -F, '{print NF}' <<<"${compact}"
}

_if_rlvr_build_gpu_set() {
    local slot=$1
    local count=$2
    local start=$((slot * count))
    local end=$((start + count - 1))
    local gpu_set=""
    local gpu

    for gpu in $(seq "${start}" "${end}"); do
        if [[ -z "${gpu_set}" ]]; then
            gpu_set=${gpu}
        else
            gpu_set="${gpu_set},${gpu}"
        fi
    done
    echo "${gpu_set}"
}

_if_rlvr_sanitize_id() {
    tr -cs 'A-Za-z0-9_.-' '_' <<<"$1" | sed 's/^_*//; s/_*$//'
}

_if_rlvr_first_visible_gpu() {
    local csv=${1:-}
    local first=${csv%%,*}
    first=${first//[[:space:]]/}
    if [[ "${first}" =~ ^[0-9]+$ ]]; then
        echo "${first}"
    else
        echo 0
    fi
}

SCRIPT_STEM=$(basename "${BASH_SOURCE[1]:-${0}}" .sh)

if [[ -n "${GPU_SLOT:-}" && ! "${GPU_SLOT}" =~ ^[0-9]+$ ]]; then
    echo "GPU_SLOT must be a non-negative integer, got: ${GPU_SLOT}" >&2
    exit 1
fi

if [[ -n "${GPU_SLOT:-}" && -z "${GPU_SET:-}" ]]; then
    GPU_SET=$(_if_rlvr_build_gpu_set "${GPU_SLOT}" "${NGPUS_PER_NODE}")
fi

if [[ -n "${GPU_SET:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_SET}"
fi

if [[ -z "${RUN_SLOT:-}" ]]; then
    if [[ -n "${GPU_SLOT:-}" ]]; then
        RUN_SLOT=${GPU_SLOT}
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        first_gpu=$(_if_rlvr_first_visible_gpu "${CUDA_VISIBLE_DEVICES}")
        RUN_SLOT=$((first_gpu / NGPUS_PER_NODE))
    else
        RUN_SLOT=0
    fi
fi

if [[ ! "${RUN_SLOT}" =~ ^[0-9]+$ ]]; then
    echo "RUN_SLOT must be a non-negative integer, got: ${RUN_SLOT}" >&2
    exit 1
fi

if [[ "${DEVICE}" == "gpu" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_gpu_count=$(_if_rlvr_count_csv_items "${CUDA_VISIBLE_DEVICES}")
    if [[ "${visible_gpu_count}" -ne "${NGPUS_PER_NODE}" ]]; then
        echo "CUDA_VISIBLE_DEVICES has ${visible_gpu_count} entries, but NGPUS_PER_NODE=${NGPUS_PER_NODE}." >&2
        echo "Set NGPUS_PER_NODE to match the visible GPU count, or fix GPU_SLOT/GPU_SET." >&2
        exit 1
    fi
elif [[ "${DEVICE}" == "gpu" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "WARNING: CUDA_VISIBLE_DEVICES is unset; concurrent runs may compete for the same physical GPUs." >&2
    echo "         Prefer GPU_SLOT=0 and GPU_SLOT=1 for two 4-GPU runs on an 8-GPU host." >&2
fi

IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID:-$(_if_rlvr_sanitize_id "${SCRIPT_STEM}_slot${RUN_SLOT}")}
IF_RLVR_PORT_BASE=${IF_RLVR_PORT_BASE:-$((20000 + RUN_SLOT * 2000))}

if [[ ! "${IF_RLVR_PORT_BASE}" =~ ^[0-9]+$ ]]; then
    echo "IF_RLVR_PORT_BASE must be an integer, got: ${IF_RLVR_PORT_BASE}" >&2
    exit 1
fi

export VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE:-$((IF_RLVR_PORT_BASE + 200))}
export VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE:-100}
export VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT:-16}

IF_RLVR_RAY_MASTER_PORT_RANGE_START=${IF_RLVR_RAY_MASTER_PORT_RANGE_START:-${IF_RLVR_PORT_BASE}}
IF_RLVR_RAY_MASTER_PORT_RANGE_END=${IF_RLVR_RAY_MASTER_PORT_RANGE_END:-$((IF_RLVR_RAY_MASTER_PORT_RANGE_START + 100))}

# Ray appends long session and socket names under _temp_dir. Keep this path
# deliberately short to stay below the 107-byte AF_UNIX socket path limit.
RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ifrlvr_r${RUN_SLOT}}
mkdir -p "${RAY_TMPDIR}"

if [[ "${IF_RLVR_ISOLATE_RAY:-1}" == "1" ]]; then
    unset RAY_ADDRESS
fi

unset MASTER_ADDR MASTER_PORT DIST_INIT_METHOD RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE

IF_RLVR_RAY_INIT_OVERRIDES=(
    "+ray_kwargs.ray_init._temp_dir=${RAY_TMPDIR}"
    "+ray_kwargs.ray_init.include_dashboard=False"
    # Cap the Plasma object store. Ray defaults it to 200 GB here because it misreads the
    # container memory (host 2.2 TB, not the 1600 GiB cgroup cap); rollout DataProto
    # batches need only a few GB, so reserving 200 GB just adds to the pressure that
    # stalls the raylet. Override via IF_RLVR_OBJECT_STORE_BYTES (default 64 GiB).
    "+ray_kwargs.ray_init.object_store_memory=${IF_RLVR_OBJECT_STORE_BYTES:-68719476736}"
    "+ray_kwargs.ray_init.namespace=${IF_RLVR_RUN_ID}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_MASTER_PORT_BASE=${VLLM_MASTER_PORT_BASE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_PORT_STRIDE=${VLLM_PORT_STRIDE}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RESERVED_PORT_COUNT=${VLLM_RESERVED_PORT_COUNT}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_RLVR_RUN_ID=${IF_RLVR_RUN_ID}"
    "+ray_kwargs.ray_init.runtime_env.env_vars.IF_REQUIRE_THINK_END_FOR_REWARD=${IF_REQUIRE_THINK_END_FOR_REWARD:-false}"
    "+trainer.ray_worker_group_master_port_range=[${IF_RLVR_RAY_MASTER_PORT_RANGE_START},${IF_RLVR_RAY_MASTER_PORT_RANGE_END}]"
)

echo "[IF-RLVR] run_id=${IF_RLVR_RUN_ID} run_slot=${RUN_SLOT} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset}" >&2
echo "[IF-RLVR] ray_master_ports=${IF_RLVR_RAY_MASTER_PORT_RANGE_START}-${IF_RLVR_RAY_MASTER_PORT_RANGE_END} vllm_base=${VLLM_MASTER_PORT_BASE} stride=${VLLM_PORT_STRIDE}" >&2
