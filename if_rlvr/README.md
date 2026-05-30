# Instruction-Following RLVR for verl (`if_rlvr`)

Train (e.g. Qwen3-8B) in verl with AllenAI **open-instruct**'s instruction-following verifiable reward
— the `ifeval` verifier used by `open-instruct/scripts/train/rlvr/valpy_if_grpo_fast.sh` on
`allenai/IF_multi_constraints_upto5`.

## Design: orthogonal to verl

This recipe is **decoupled from verl's training internals** so it survives algorithm changes and verl
version updates:

- The reward **logic** (`ifeval_oi/`) is pure Python with **zero `verl` imports** — it can't break from any
  verl change. (`instructions*.py` are byte-identical to open-instruct except their intra-package imports.)
- The reward **plugs into verl via the stable, public `custom_reward_function` hook** (`if_reward_fn.py`),
  which works with **any** reward manager (`naive`/`dapo`/`prime`/…), **any** RL algorithm
  (GRPO/PPO/RLOO/DAPO/…), and across verl versions. Training stays 100% verl-native.
- Nothing here **modifies any existing verl file**. The whole directory is self-contained and can be moved
  anywhere (inside or outside the verl tree) and referenced by absolute path.

```
if_rlvr/
  ifeval_oi/                 # pure, verl-INDEPENDENT verifier (vendored IFEvalG + score_ifeval)
  if_reward_fn.py            # >> RECOMMENDED << verl custom_reward_function entry (no verl imports)
  if_reward_manager.py       # OPTIONAL: custom reward manager that adds the non-stop penalty (couples to verl)
  convert_data.py            # allenai/IF_multi_constraints_upto5 -> verl parquet (standalone, only needs `datasets`)
  run_qwen3_8b_if_rlvr.sh    # run_qwen3_8b_fsdp.sh config + IF data + IF reward + enable_thinking toggle
  requirements.txt           # langdetect, immutabledict, nltk (+ nltk punkt/punkt_tab data)
  tests/                     # reproduce_ifeval.py (vs the real open-instruct oracle), integration_smoke.py
```

## Two integration paths

| | Hook | Coupling | Non-stop penalty |
|---|---|---|---|
| **Orthogonal (default)** | `custom_reward_function.path=if_reward_fn.py` | none (stable public hook) | no |
| **Optional** | `reward.reward_manager.source=importlib … if_reward_manager.py` | verl reward-manager API | yes (truncated → `non_stop_penalty_value`) |

The orthogonal reward = **fraction of IFEval constraints satisfied** (thinking section stripped before
verification). open-instruct also multiplied by `verification_reward=10.0`, but under GRPO advantage
std-normalization any positive scale is equivalent, so the default returns the raw `[0,1]` fraction
(set `+custom_reward_function.reward_kwargs.verification_reward=10.0` to match the magnitude). The only
piece of open-instruct's reward the orthogonal hook can't reproduce is the **non-stop penalty** (it needs
response token ids, which `custom_reward_function` doesn't receive) — use the optional manager for that.

## Setup & run

```bash
conda activate verl
pip install langdetect immutabledict nltk          # on EVERY ray node (or shared NLTK_DATA)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt')"
python if_rlvr/convert_data.py \
    --output-dir /lustre/justinseo/if-verl/data/ifeval_multi --val-size 512 --seed 1

ENABLE_THINKING=true  bash if_rlvr/run_qwen3_8b_if_rlvr.sh
ENABLE_THINKING=false bash if_rlvr/run_qwen3_8b_if_rlvr.sh
```

The run script is verl's `examples/grpo_trainer/run_qwen3_8b_fsdp.sh` training config (lr 1e-6, kl 0.001
`low_var_kl`, batch 1024, `rollout.n=5`, std-normalized GRPO advantage, 15 epochs) with only the **data**
and **reward** swapped in. It pins the if-verl/verl checkout via `PYTHONPATH`.

## Verification

```bash
PYTHONPATH=/lustre/justinseo/if-verl/verl python if_rlvr/tests/reproduce_ifeval.py --per-type 120
PYTHONPATH=/lustre/justinseo/if-verl/verl python if_rlvr/tests/integration_smoke.py
```

`reproduce_ifeval.py` compares `score_ifeval` against the **actual** `open_instruct.IFEvalVerifier` (imported
with a tiny `beaker` stub) → **0 mismatches over ~52k (response, constraint) pairs, all 54 instruction types**.
`integration_smoke.py` validates both reward paths end-to-end through verl `load_reward_manager` + `RLHFDataset`.

## "Unseeded" — identical to open-instruct (verified)

Three checkers (`language:response_language`, `change_case:english_*`, ~15% of constraints) call
`langdetect.detect`, which is non-deterministic unless `langdetect.DetectorFactory.seed` is set.
**Neither open-instruct nor this recipe seeds it** (grep-confirmed in both), so the reward is mildly
non-deterministic on borderline inputs — this is *exactly* open-instruct's original behavior, faithfully
reproduced (not introduced here). `build_description`'s `random` calls only fire when a kwarg is absent;
the dataset always supplies kwargs, so on real data the only non-determinism source is langdetect. The
reproduction test seeds both only to compare code paths. (To make it deterministic in production — a
*deviation* from open-instruct — set `langdetect.DetectorFactory.seed=0` in `ifeval_oi/instructions.py`.)

## Runtime deps note

The reward runs on round-robin-scheduled reward-loop actors, so `langdetect`/`immutabledict`/`nltk` +
`punkt`/`punkt_tab` must be on **every** ray node (or via shared `NLTK_DATA`). The optional
`IFRewardManager` fails fast at startup if they're missing; the orthogonal `compute_score` raises on the
first sample that needs a missing resource.
