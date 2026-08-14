# Qwen3-4B 4k-Subset Reward Ablations (Stage 1)

Six runs on the same 4,096 prompts, each changing exactly one component of the anchor
reward rule. Design + evidence: `if_rlvr/docs/anchor_subset_curation_design_2026-08-13.md`
and `if_rlvr/data_subsets/qwen3_4b_reasoning_anchor4k/v2_subset_final_verification.md`.

## Arms (run in this priority order)

| # | script | rule (rows with IF > 0) | wandb / checkpoint name |
|---|---|---|---|
| 1 | `a0_constraint_only.sh` | reward = IF, no anchors | `qwen3_4b_sub4k_a0_constraint_only_b256_ep4` |
| 2 | `a3_anchor_strict.sh` | P<A → 0 (flips zeroed too — July semantics) · band → IF+0.1 | `..._a3_baseline_floorzero_flipszeroed_...` |
| 3 | `a2_anchor_softfloor.sh` | P<A → IF (no zero, forfeits bonus) · band → IF+0.1 | `..._a2_softfloor_keepif_...` |
| 4 | `a4_anchor_penalty01.sh` | P<A → IF−0.1 · band → IF+0.1 | `..._a4_floorpenalty01_...` |
| 5 | `a5_anchor_nofloor.sh` | no floor: P≤B → IF+0.1 | `..._a5_nofloor_upperonly_...` |
| 6 | `a1_anchor_flipabstain.sh` | flips → IF untouched; valid P<A → 0 (= shipped fixed rule) | `..._a1_flipabstain_floorzero_...` |

Every arm shares: the same 4,096 rows, b=256 (16 steps/epoch), 4 epochs = 64 steps,
n=8 rollouts, lr 1e-6, budgets 2048/8192, rollout sampling t1.0/p0.95/k20 (matched to
the v2 anchors), per-epoch checkpoints with `hf_model` export, no in-training validation.
Reward knobs are implemented in `verl/trainer/ppo/ray_trainer.py::apply_if_ppl_anchor_reward`
(`IF_PPL_ANCHOR_FLIP_HANDLING`, `IF_PPL_ANCHOR_FLOOR_ACTION`, `IF_PPL_ANCHOR_FLOOR_PENALTY`;
defaults preserve current behavior).

## Prerequisites

1. This repo (the reward knobs live in `verl/trainer/ppo/ray_trainer.py`), a working verl
   conda/venv, 8×80GB GPUs.
2. Network access to the HF Hub on first run — everything data-side is fetched
   automatically: the subset cache (public, from `sangyon/anchor_cache`; 3,620 r8192
   rows + 476 targeted r32768 fills, provenance in
   `if_rlvr/data_subsets/qwen3_4b_reasoning_anchor4k/v2_subset_provenance.json`),
   `Qwen/Qwen3-4B`, `allenai/IF_multi_constraints_upto5`, and the NLTK punkt data.
   `HF_HOME` defaults to `<repo>/.cache/huggingface`; override if your cache lives
   elsewhere. Air-gapped boxes: pre-place the subset JSON at
   `<repo>/.cache/<same filename>` and warm the HF caches.
3. `WANDB_API_KEY` in the environment.

## Run

All six sequentially (verify → reap FSDP shards keeping hf_model → GPU teardown between arms;
fail-stop on an incomplete arm; safely re-runnable, completed arms are skipped):
```
WANDB_API_KEY=... nohup bash if_rlvr/exps/subset4k/run_all_arms.sh > run_all_arms.log 2>&1 &
```
One arm:
```
WANDB_API_KEY=... bash if_rlvr/exps/subset4k/a3_anchor_strict.sh
```

## Notes

- **Memory**: tuned for 8×80GB — engine sleep on, util 0.9, and `PPO_MAX_TOKEN_LEN_PER_GPU=65536`.
  The 98304 default OOMs at b=256 (a maximally-packed microbatch's 151k-vocab logits are ~26GB;
  observed 2026-08-14). On the first run anywhere, watch step 1–2 for memory before walking away.
- **Strict metadata is off by design**: the cache records the build box's `tokenizer_class` and
  `train_sample_count=93882`, which can never equal a subset run's values. The launcher preflight
  asserts all *semantic* fields (sampling, budgets, prefix mode, scope, model) instead.
- Completion = `latest_checkpointed_iteration.txt == 64` in
  `checkpoints/verl_if_rlvr/<experiment>/`; epoch checkpoints at steps 16/32/48/64.
- Compare arms at matched epochs only; deltas read against A3 (the strict baseline).
