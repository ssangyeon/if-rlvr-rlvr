# Qwen3-1.7B KL-coefficient ablation (non-reasoning, constraint-only)

Sweeps **one** knob — `actor_rollout_ref.actor.kl_loss_coef`, the coefficient on the
`low_var_kl(policy ‖ reference)` term added to the GRPO policy loss — and holds everything else
fixed, so the training-dynamics curves are directly comparable across KL strengths.

Base recipe: [`qwen3_17b_constraint_only_nonreason.sh`](qwen3_17b_constraint_only_nonreason.sh)
(Qwen3-1.7B, `enable_thinking=false`, constraint-only IFEval reward, GRPO, no anchor/PPL path).

## Arms

| # | multiplier | `kl_loss_coef` | wandb `experiment_name` |
|---|-----------|----------------|--------------------------|
| — | 1× (control, pre-existing) | 0.001 | `qwen3_17b_grpo_nonthink_constraint_only_b1024_c1` |
| 1 | 2×  | 0.002 | `qwen3_17b_grpo_nonthink_constonly_klabl_x02_kl0p002` |
| 2 | 5×  | 0.005 | `qwen3_17b_grpo_nonthink_constonly_klabl_x05_kl0p005` |
| 3 | 10× | 0.01  | `qwen3_17b_grpo_nonthink_constonly_klabl_x10_kl0p01`  |
| 4 | 20× | 0.02  | `qwen3_17b_grpo_nonthink_constonly_klabl_x20_kl0p02`  |

All four share the `klabl` tag, so `klabl` in the wandb project search returns exactly the sweep.
Project `verl_if_rlvr`, entity `ifif`.

**The 1× control is imperfect.** The pre-existing run used 4 GPUs, 5 epochs, a random dataloader
order and no in-training validation. The optimization is identical, but for a fully matched control
just add it to the sweep:

```bash
KL_MULTS="1 2 5 10 20" bash if_rlvr/exps/bidirectional/run_qwen3_17b_kl_ablation.sh
```

## Run it

```bash
# all four arms, back to back, all 8 GPUs each
bash if_rlvr/exps/bidirectional/run_qwen3_17b_kl_ablation.sh

# variations
DRY_RUN=1        bash .../run_qwen3_17b_kl_ablation.sh   # print the resolved commands, launch nothing
KL_MULTS="10 20" bash .../run_qwen3_17b_kl_ablation.sh   # a subset
KL_LOSS_COEFS="0.003 0.03" bash .../run_qwen3_17b_kl_ablation.sh   # explicit coefficients
ON_FAILURE=continue bash .../run_qwen3_17b_kl_ablation.sh          # don't stop the sweep on a failed arm
SAVE_HF_MODEL=true  bash .../run_qwen3_17b_kl_ablation.sh          # also write HF-format checkpoints

# stop everything (ordered teardown; see below)
bash if_rlvr/exps/bidirectional/stop_qwen3_17b_kl_ablation.sh
```

`WANDB_API_KEY` must be set. Detached:

```bash
setsid nohup bash if_rlvr/exps/bidirectional/run_qwen3_17b_kl_ablation.sh \
  > logs/qwen3_17b_kl_ablation/sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Every arm self-resumes (`trainer.resume_mode=auto`, `IF_MAX_RETRIES=10`), and an arm that already
reached its last step exits immediately — so re-running the sweep script after an interruption picks
up where it stopped instead of redoing finished work.

## Steps and checkpoints

95,373 HF rows − 512 val − 980 prompts over the 2048-token budget = **93,881 train rows**
(measured with the Qwen3-1.7B chat template at `enable_thinking=false`; 98.97% retention).

```
93,881 / 1024 = 91 steps/epoch   ×  4 epochs  =  364 steps per arm
SAVE_FREQ=91  →  checkpoints at 91 / 182 / 273 / 364 (one per epoch)
```

Each checkpoint is fp32 FSDP model shards (~8.1 GB) + Adam state (~16.2 GB) ≈ **25 GB**;
4 arms × 4 checkpoints ≈ **400 GB**. The runner refuses to start if the volume can't hold that
(`ALLOW_LOW_DISK=1` overrides).

`SAVE_HF_MODEL=true` adds an fp32 HF-format export (+8 GB each, ~530 GB total). It is **off** by
default because the shards convert after the fact:

```bash
python3 -m verl.model_merger merge --backend fsdp --tie-word-embedding \
  --local_dir checkpoints/verl_if_rlvr/<experiment>/global_step_364/actor \
  --target_dir <output>
```

(`--tie-word-embedding` because Qwen3-1.7B ties its input and output embeddings.)

## What is held fixed, and why it matters

Identical to the script being ablated:

- `train_batch_size=1024`, `ppo_mini_batch_size=1024` → `ppo_epochs=1`, i.e. **one on-policy
  gradient step per batch**. The importance ratio is identically 1, so `ppo_kl` and `pg_clipfrac`
  read 0.000 at every step and PPO clipping never fires. The KL term is the *only* guardrail in
  this recipe — which is exactly what makes this sweep worth running.
- actor lr `5e-7`, prompt/response budget `2048/2048`, rollout `n=8`, `entropy_coeff=0`,
  `kl_loss_type=low_var_kl`, `algorithm.use_kl_in_reward=False`.
- **Sampling stays at the verl defaults: temperature 1.0, top_p 1.0, top_k −1 (untruncated).**
  Do not "improve" this. The Qwen3-4B reasoning run in this repo used truncated sampling (forced on
  it by the anchor-cache metadata contract) and diverged at step ~190 of 273; truncated sampling
  sustained over hundreds of steps is the leading suspect, because vLLM samples from `truncate(π)`
  while the actor differentiates the full 151,936-token softmax. This ablation is about the term
  that opposes exactly that drift, so the sampler must stay untruncated and identical to the
  reference.

Changed by request: **`TOTAL_EPOCHS=4`** (was 5).

## Throughput settings (8× H100 80GB)

Every one of these is a throughput/memory knob with **no effect on the optimization**, so moving
from the reference's 4 GPUs does not compromise comparability.

| knob | value | why |
|---|---|---|
| `NGPUS_PER_NODE` | 8 | Global batch is unchanged (1024 prompts × 8 samples); FSDP just shards it 8 ways, and `token-mean` aggregation normalizes by a globally all-reduced token count, so the objective is identical. Roughly halves wall clock. |
| `ROLLOUT_TP` | 1 | A 1.7B model needs no tensor parallelism. TP=1 → 8-way rollout data parallelism, zero TP collectives, full KV cache per replica. |
| `rollout.max_model_len` | 4096 | = 2048 + 2048 exactly. **Must** be set: verl leaves it `null`, so vLLM falls back to `max_position_embeddings` = 40960 for Qwen3-1.7B — 10× the block-table width this run can use. |
| `rollout.max_num_seqs` | 512 | KV is 2 × 28 layers × 8 kv-heads × 128 head-dim × 2 B = **112 KiB/token**. At util 0.85 (69.4 GB of 81.6) minus ~4.1 GB bf16 weights and ~4 GB engine overhead ≈ 61 GB ≈ **530k tokens** of KV per replica. Each replica sees 1024 requests (8192/8); with prefix caching the 8 samples of a prompt share its prompt blocks, so 512 concurrent sequences need only ~512 × (279/8 + 450) ≈ 250k tokens. Large enough to saturate decode on a 1.7B model, small enough that the long tail doesn't thrash on preemption — admitting all 1024 (the verl default) would. |
| `ROLLOUT_GPU_MEM_UTIL` | 0.85 | The reference used 0.90. vLLM's budget is an absolute fraction of **total** memory measured at engine init, so 0.90 leaves ~4.8 GB for the training process's resident fp32 FSDP shards + Adam (~3.4 GB/GPU here). We're nowhere near KV-bound at 512 seqs, so the last 4 GB buys nothing; 0.85 restores ~9 GB of headroom. On the Qwen3-4B run, 0.90 measured 81.0 GB of 81.6 GB during rollout. |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | 32768 | The base launcher defaults to 98304. At the 151,936-token vocab a microbatch costs ~0.283 MiB/token in bf16 logits **and the same again** in the logits gradient — a model-size-independent term, so 98304 tokens is ~60 GB of logits alone. 32768 is the value measured safe at 4B on this box. It costs nothing measurable: ~717k tokens/GPU/step means 22 microbatches instead of 7, and each extra FSDP all-gather + reduce-scatter for 1.7B is ~15 ms over NVLink — ~0.3 s against a multi-minute step. |
| `LOG_PROB_MAX_TOKEN_LEN_PER_GPU` | 65536 | **2× the actor cap**, not the usual "same as the actor". `old_log_prob` and `ref` are forward-only under `no_grad`, and `logprobs_from_logits` uses flash-attn's fused cross-entropy (available here), so neither materializes an fp32 full-vocab tensor nor retains activations — the measured 44.0 GB allocated / 47.5 GB reserved step peak comes from `update_actor`'s **backward**, not these passes. They are overhead-bound, not FLOP-bound: 786k tokens/GPU of forward in 25.9 s is ~123 TFLOPS on a card that does ~600 on this shape. Halving the pass count cuts ~12% off the step. |
| `AGENT_NUM_WORKERS` | 256 | Base default is 32. Each `AgentLoopWorker` is a Ray actor that also runs the IFEval verifier (`compute_score`) for its chunk, and that is CPU-bound Python. 1024 prompts / 256 workers = 4 each on a 224-core box. |
| `DATA_PROCESSOR_CPU_COUNT` | 32 | One-off prompt-length filtering at startup. |

## Ablation hygiene (added; also no effect on the objective)

- **`data.seed=1`** — verl leaves `data.seed` null, which seeds the dataloader's `RandomSampler`
  from entropy, so every arm would see a *different* batch order: pure variance on the one thing
  the sweep is isolating. Pinning it makes all arms walk the dataset identically. (vLLM sampling is
  still nondeterministic under continuous batching; this removes the largest controllable confound,
  not all of them.)
- **`TEST_FREQ=91` + `trainer.val_before_train=True`** — one validation pass per epoch plus a step-0
  baseline on the 512-row held-out split. `val_kwargs` is greedy (`do_sample=false`,
  `temperature=0`, `n=1`), computes no gradient and costs about a minute, so it is free against a
  364-step run — and `val_before_train` catches a broken reward/verifier path before an arm burns
  an epoch. The reference run had `test_freq=1000`, i.e. never.
- **`IF_THINK_END_TOKEN=none`** — disables the `think/*` reasoning-vs-answer split metrics. This is a
  non-reasoning run: `</think>` never appears in a response, so the split would report every rollout
  as 100% reasoning tokens and 0% answer tokens. (`resolve_think_end_token_id` in
  `verl/trainer/ppo/metric_utils.py` accepts `none`/`off`/`disabled`/`0`/empty as an explicit
  opt-out.)

## What to watch

The point of the sweep is the shape of these curves as `kl_loss_coef` rises:

| metric | what a rising KL coefficient should do |
|---|---|
| `actor/kl_loss` | the directly penalized quantity — should be pinned progressively closer to 0 |
| `actor/entropy` | the failure mode being guarded against. In the 4B run entropy sat flat through epoch 1, began climbing the moment reward plateaued, and then ran away to 11.74 (98.4% of `ln 151936` = 11.931). Higher KL should flatten or reverse that climb. |
| `critic/rewards/mean`, `critic/score/mean` | the cost side: how much task reward the extra anchoring gives up |
| `actor/grad_norm` | the 4B divergence showed 0.25 → 6.62 → 21.2 before collapse |
| `response_length/mean` | length drift; collapse in the 4B run came with runaway length |
| `val-core/.../mean@1` | held-out constraint satisfaction, once per epoch |
| `actor/ppo_kl`, `actor/pg_clipfrac` | both **exactly 0.000** every step (observed through step 3). The differing microbatch caps for the log-prob passes (65536) and the update (32768) do not perturb this: with `use_remove_padding=True` and block-diagonal flash attention, a packed sequence's per-token logits depend only on that sequence, so grouping sequences differently is bitwise-neutral. A non-zero value here means `ppo_mini_batch_size` or `ppo_epochs` changed and the arms are no longer comparable. |

## Stopping

```bash
bash if_rlvr/exps/bidirectional/stop_qwen3_17b_kl_ablation.sh
CLEAR_RAY_TMPDIR=1 bash .../stop_qwen3_17b_kl_ablation.sh   # also drop /tmp/ifrlvr_r0
```

Killing the trainer process alone is **not enough and makes things worse**: the base launcher wraps
`python3 -m verl.trainer.main_ppo` in an auto-resume loop (`IF_MAX_RETRIES`, default 10), so killing
just the driver makes the loop relaunch it. If a second arm has meanwhile started, two trainers
share the same GPUs, Ray temp dir, namespace and port range, which surfaces as a Gloo error that
names none of the real cause:

```
RuntimeError: [.../gloo/transport/tcp/pair.cc:544] Connection closed by peer
... in _build_model_optimizer -> torch.distributed.barrier()
```

The stop script therefore goes wrapper → tee → driver → vLLM → Ray, matches on
`/proc/<pid>/cmdline` (so it can't `pkill` itself the way `pkill -f` does), skips its **whole
ancestor chain** — which is what lets the sweep runner call it between arms — and re-scans up to
four times because Ray respawns workers while its raylet is dying.

The runner already calls it between arms with `QUIET=1 CLEAR_RAY_TMPDIR=1` and then waits for every
GPU to drop below 1 GB, so a crashed arm can't leave engine cores holding memory or ports that the
next arm needs.

Caveat: the `verl.trainer.main_ppo` and `ray::` patterns are host-wide. An unrelated verl job on the
same box would be taken down too.

## Port range gotcha (this one bit us)

`VLLM_MASTER_PORT_BASE` is **25000** here, not the 40000 the other launchers in this repo use.
40000 sits inside the kernel's ephemeral range (`/proc/sys/net/ipv4/ip_local_port_range` =
`32768 60999` on this box). This run starts ~400 Ray processes — 8 rollout workers, 256
`AgentLoopWorker` actors, plus Ray's own — and each binds a listening socket on a random ephemeral
port. Some land exactly where vLLM wants to go. `_reserve_port_block()` in
`verl/workers/rollout/vllm_rollout/vllm_async_server.py` needs `VLLM_RESERVED_PORT_COUNT=16`
**consecutive** free ports inside each replica's `VLLM_PORT_STRIDE=100` window, so roughly six
scattered squatters in one window is fatal:

```
INFO   Shifted vLLM replica 1 port block from 40100-40115 to 40110-40125 ... was busy
INFO   Shifted vLLM replica 7 port block from 40700-40715 to 40704-40719 ... was busy
OSError: [Errno 98] Address already in use
OSError: No free vLLM port block of 16 ports in replica slot 40200-40299
```

It is a dice roll at every engine start, so it would hit arm 3 as readily as arm 1 — and each hit
costs an `IF_MAX_RETRIES` cycle. 25000 + 8 replicas × stride 100 = **25000-25799**, clear of both
the ephemeral range and the 22000-22100 Ray worker-group range, with `+1000` per `RUN_SLOT` so
concurrent slots stay isolated. The runner preflights this and warns if the window ever overlaps
the ephemeral range or already has listeners.

The other launchers in this directory (`qwen3_17b_*`, `qwen3_4b_thinking_*`, …) still use 40000 and
carry the same latent hazard.

## Environment gotcha

vLLM 0.11.0 pins `numba==0.61.2`, which requires `numpy < 2.3`. If the env drifts you get:

```
ImportError: Numba needs NumPy 2.2 or less. Got NumPy 2.5
```

Fix: `pip install "numpy==2.2.6"`. (Currently pinned correctly.)

An ambient `HF_HOME=/workspace/.cache/huggingface` on this box holds none of the weights; the runner
pins `HF_HOME` to `<repo>/.cache/huggingface` so the arms don't re-download onto the wrong volume.
Override deliberately with `KL_ABL_HF_HOME`.
