# shellcheck shell=bash
# Throughput profile for run_constraint_only_kl_sweep_h200.sh (8x H200 141GB, 2 runs x 4 GPUs).
#
# Sourced by the sweep runner at the START OF EVERY JOB, not once at launch, so the knobs for later
# phases can be retuned while earlier phases are still training -- edit this file, and the next job
# that starts picks the new values up. A job that is already running is unaffected.
#
# Every value here is a throughput/memory knob with no effect on the optimization objective:
# batch 1024 x n=8, ppo_mini_batch 1024 (one on-policy step per batch), lr, KL type, sampling and
# token budgets all come from the leaf scripts unchanged. Micro-batch packing only changes the
# float summation order of the gradient, and gradient checkpointing only decides whether
# activations are stored or recomputed.
#
# Memory model used to pick these (per GPU, FSDP over 4 GPUs; H200 = 140.4 GiB):
#   resident between steps: fp32 params + Adam m/v = 12 B/param / 4 GPUs
#       (grads are freed: EngineTrainModeCtx.__exit__ calls zero_grad(set_to_none=True))
#       Llama-3.1-8B ~24 GB, Qwen3-4B ~12 GB, Qwen3-1.7B ~5 GB.
#   rollout:  vLLM takes gpu_memory_utilization x 140.4 GiB on top of that resident state.
#   training: vLLM sleeps (free_cache_engine=True), so update_actor / log-prob passes get the rest.
#
# Per-family variables (prefix TULU_, Q4BNT_, Q17BT_, Q4BT_):
#   *_GPU_UTIL          vLLM gpu_memory_utilization
#   *_MAX_NUM_SEQS      vLLM max_num_seqs (concurrent sequences per replica)
#   *_BATCHED_TOKENS    vLLM max_num_batched_tokens
#   *_GRAD_CKPT         actor gradient checkpointing (True/False)
#   *_PPO_TOKENS        actor ppo_max_token_len_per_gpu (update micro-batch token cap);
#                       must be >= prompt+response length of the longest single sequence
#   *_LOGPROB_TOKENS    old_log_prob / ref forward micro-batch token cap

# ---- Llama-3.1-Tulu-3-8B-DPO, non-reasoning, 2048/2048 -------------------------------------------
# Rollout is tail-bound here (mean response ~120 tokens, slowest 2048): 2.85M tokens/step, of which
# the original 8xH100 run spent 30s in gen and 114s in old_log_prob+ref+update. KV need is small.
TULU_GPU_UTIL=0.68
TULU_MAX_NUM_SEQS=1024
TULU_BATCHED_TOKENS=16384
TULU_GRAD_CKPT=False
TULU_PPO_TOKENS=10240
TULU_LOGPROB_TOKENS=65536

# ---- Qwen3-4B, non-reasoning, 2048/2048 ----------------------------------------------------------
Q4BNT_GPU_UTIL=0.80
Q4BNT_MAX_NUM_SEQS=1024
Q4BNT_BATCHED_TOKENS=16384
Q4BNT_GRAD_CKPT=False
Q4BNT_PPO_TOKENS=12288
Q4BNT_LOGPROB_TOKENS=65536

# ---- Qwen3-1.7B, reasoning, 2048/8192 ------------------------------------------------------------
# Rollout is KV-read-bound (mean response ~2.6k tokens): KV is 112 KiB/token.
Q17BT_GPU_UTIL=0.85
Q17BT_MAX_NUM_SEQS=1024
Q17BT_BATCHED_TOKENS=16384
Q17BT_GRAD_CKPT=False
Q17BT_PPO_TOKENS=16384
Q17BT_LOGPROB_TOKENS=65536

# ---- Qwen3-4B, reasoning, 2048/8192 --------------------------------------------------------------
# KV is 144 KiB/token. PPO cap must stay >= 10240 (2048 + 8192) for a single max-length sequence.
Q4BT_GPU_UTIL=0.82
Q4BT_MAX_NUM_SEQS=1024
Q4BT_BATCHED_TOKENS=16384
Q4BT_GRAD_CKPT=False
Q4BT_PPO_TOKENS=12288
Q4BT_LOGPROB_TOKENS=65536

# Per-job escape hatch: KLSWEEP_EXTRA_HYDRA_<job> (a bash array name is not portable through env,
# so use a single space-separated string; values must not contain spaces). Example:
#   KLSWEEP_EXTRA_HYDRA_tulu3_8b_kl0p05="actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=49152"

# Phase-1 A/B result (2026-11-07, Tulu3-8B step 1, identical workloads, 5.65M tokens/step):
#   GC off, 8192-token micro-batches : update_actor 174.2s, peak alloc 76.2GB / reserved 89.2GB
#   GC on, 49152-token micro-batches : update_actor 175.1s, peak alloc 79.0GB / reserved 92.2GB
# No measurable difference (the recompute saved by GC-off is paid back in 6x more FSDP
# all-gather/reduce-scatter rounds), so every family uses the conventional GC-on profile.

# Phase-1 A/B #2: spend the ~45GB of idle headroom. GC off removes the recompute; larger
# micro-batches cut the FSDP gather/scatter rounds that cancelled that gain at 8192.
#   slot 0 (kl0p02): GC off, ppo 12288, logprob 65536   (profile above)
#   slot 1 (kl0p05): GC off, ppo 16384, logprob 98304   (near the 140GB ceiling)
# Result (step 1): slot 0 = update_actor 162.0s (-7% vs 175s), peak alloc 104.3 / reserved 117.4GB,
# step 394s, actor/ppo_kl = pg_clipfrac = 0.0. Slot 1 (16384) pushed GPU4 to 142.9GB of 143.8GB
# during the step -> no headroom; rejected. ADOPTED for all families: GC off + 12288-token update
# micro-batches (1.7B: 16384, its activations are ~half per token) + 65536-token log-prob passes.

# 2026-11-07 14:21, Tulu step 2 at 12288: peak alloc 119.4 / reserved 132.5GB (step 1 was 104/117) --
# packing variance alone moved the peak +15GB, leaving ~8GB of 140GB. A multi-day run must not ride
# that edge (an OOM costs up to 91 steps), so Tulu drops to 10240 (~-20GB). Qwen families keep
# 12288: their resident fp32 params+Adam is 12GB vs Tulu's 24GB, and they have no 8GB-larger grads.
