# Direct likelihood baselines

The raw-likelihood Qwen3-1.7B non-reasoning baseline uses

```text
y = extracted final answer (after </think>, when present)
NLL = -sum_t log p_ref(y_t | x, y_<t)
likelihood = exp(-NLL) = product_t p_ref(y_t | x, y_<t)
reward = constraint_reward + p_ref(final_answer | x)
```

Run from the checkout in the training environment:

```bash
bash if_rlvr/exps/bidirectional/qwen3_17b_raw_likelihood_nonreason.sh
```

The existing inverse-perplexity baseline is also available:

```bash
bash if_rlvr/exps/bidirectional/qwen3_17b_inverse_ppl_nonreason.sh
```

Both launchers default to `MODEL_PATH=Qwen/Qwen3-1.7B` and
`ENABLE_THINKING=false`. Use an environment with this checkout's GPU training
dependencies (including verl, Ray, and vLLM) and install the IF verifier
dependencies with `uv pip install -r if_rlvr/requirements.txt`. The dataset is
loaded by the existing IF dataset class; no anchor precomputation is required.

The launchers default to four GPUs numbered `4,5,6,7` and the shared launcher's
W&B entity `ifif`. Set `GPU_SET` to available devices and `WANDB_ENTITY` to your
own entity, or use console logging. For example, on GPUs 0 through 3:

```bash
GPU_SET=0,1,2,3 NLTK_DATA_DIR="${PWD}/.cache/nltk_data" \
  bash if_rlvr/exps/bidirectional/qwen3_17b_raw_likelihood_nonreason.sh \
  'trainer.logger=[console]'
```

Use the inverse-PPL script in the same command to run that baseline. The
launchers locate the checkout relative to themselves; `VERL_DIR` and
`CACHE_ROOT` remain available as overrides.

| Strategy | Added term (default coefficient 1) | Tokens scored |
| --- | --- | --- |
| `raw_likelihood` | `exp(-NLL)` | Extracted final-answer tokens, as in anchor scoring |
| `inverse_ppl` | `exp(-NLL / N)` | Extracted final-answer tokens, as in anchor scoring |

Both terms are in [0, 1]. NLL itself is nonnegative and unbounded; adding
positive NLL would reward less likely responses. Inverse perplexity is the
geometric mean token probability. Raw likelihood is the probability of the
specified token sequence. Neither measures the probability of answer correctness.

The rollout still conditions on x+c. Scoring uses the existing frozen reference,
x-only non-reasoning chat prefix, and temperature 1. Both baselines use the same
`ppl_y_final_answer_ids`, produced by the existing final-answer extraction: take
the text after `</think>` and remove leading whitespace. For these non-reasoning
Qwen3 runs, the chat prefix already closes the empty thinking block, so the
response itself supplies the final answer. Both baselines share the extractor's
tokenization and special-token handling; no additional termination token is
appended. Generated reasoning is excluded from both the scored continuation and
the scoring context. Only the final answer is appended to the x-only prefix.

If extraction finds no final answer, either likelihood baseline contributes
zero instead of falling back to scoring the reasoning. If an answer is truncated
at the token cap, its raw likelihood is the probability of the extracted token
sequence, without assuming termination afterward. The two baselines differ
only in whether the same final-answer NLL is divided by its token count before
exponentiation.

Every valid response contributes its likelihood term, including a response
with zero constraint reward. There are no anchors, rank comparisons, thresholds,
or constraint-success gates. Empty/invalid scored responses contribute zero;
the existing clipped-rollout `use` (default), `zero`, and `drop` settings are
respected. With `zero`, both the constraint and likelihood terms are zeroed.

The default likelihood coefficient is 1.0, making the reward an equal-weight
sum in [0, 2]. Both launchers accept the existing coefficient variable, for
example to compare at the anchor run's 0.1 coefficient:

```bash
PY_GIVEN_X_REWARD_COEFF=0.1 bash if_rlvr/exps/bidirectional/qwen3_17b_raw_likelihood_nonreason.sh
```

The generated experiment names include the strategy and coefficient. The
launchers forward Hydra arguments to the shared launcher and otherwise match
the two original Qwen3-1.7B launchers' training defaults. No anchor cache is
required. The existing `IF_REF_POLICY_ANCHOR_PPL=true` flag selects the
frozen-reference scorer; its name does not imply anchors are used in these
strategies. KL loss remains enabled as in the comparison runs, keeping the
reference worker available. The combined reward feeds the existing GRPO
advantage calculation.

For a controlled comparison, use identical training indices in all arms. The
existing anchor launcher filters to complete cached-anchor rows, while these
baselines and constraint-only default to the full training split. If the cache
is incomplete, supply the same explicit `IF_TRAIN_INDEX_FILTER` JSON list of
complete cached-anchor indices to every run. This fixes the examples
independently of the reward calculation.

Metrics under `if_p_y_given_x_raw_likelihood/` and `if_p_y_given_x_likelihood/`
respectively report likelihood, its weighted reward contribution, constraint
reward, composed reward, invalid scores, and excluded clipped responses.
Constraint accuracy remains logged separately.

Raw sequence probabilities can become extremely small as responses grow longer.
The raw baseline logs `zero_likelihood_count` for exponentiation underflow,
`term_underflow_count` for positive probabilities lost when converting/scaling
to the reward dtype, and `rounded_away_count` for nonzero contributions that
leave the summed reward unchanged at its numerical precision. No probability
floor, rescaling, or length normalization is applied. Tiny values can therefore
make this run behave like constraint-only training; inspect these metrics before
attributing differences solely to the choice of reward. GRPO also normalizes
combined rewards within each prompt group. Compare held-out answer quality,
constraint accuracy, length, and repetition alongside the numerical diagnostics.

Focused CPU tests (no model loading or Ray startup):

```bash
python -m pytest -q tests/trainer/ppo/test_if_likelihood_rewards_on_cpu.py
```
