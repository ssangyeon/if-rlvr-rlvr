# Tulu self-IntentCheck bonus

This experiment trains `allenai/Llama-3.1-Tulu-3-8B-DPO` using the same initial,
fixed model as an IntentCheck judge. The actor responds to the original task
plus constraints (`x+c`); the judge evaluates its final answer against `x`
from `ppl_prompt`. Both models use non-thinking chat templates.

Only the additional-bonus evaluator changes. Existing constraint reward `C`
is preserved:

| Condition | Judge call | Total reward |
|---|---|---|
| `C = 0` | Skipped | `0` |
| `C > 0`, IntentCheck YES | Called | `C + 0.1` |
| `C > 0`, IntentCheck NO | Called | `C` |
| `C > 0`, API or parsing error | Error recorded | `C` |

The prompt follows [IFDecorator](https://arxiv.org/abs/2508.04632v2), printed
pages 27–28. Attribution and the upstream prompt license are in
`if_rlvr/intentcheck.py`. This implementation does **not** adopt the paper's
complete reward combination: IntentCheck only determines the extra bonus.

## Launch configuration

```bash
bash if_rlvr/exps/bidirectional/llama31_tulu3_8b_dpo_intentcheck_llama31_tulu3_8b_dpo_bonus01_nonreason.sh --dry-run
```

Remove `--dry-run` only when ready to train. Use the existing VERL/Tulu runtime
and configure credentials through the environment; no credentials are included
in these files. The launcher defaults to four B200-class GPUs, batch size 1024
with eight rollouts per prompt, four total epochs, and saving every 91 steps.
The inherited self-judge launcher checks for idle GPUs with at least 170,000 MiB
each (180 GiB-class hardware) and adequate cgroup memory; adapt its memory
profile before using smaller GPUs. The policy output limit is 2048 tokens; the
judge limit is 1024 tokens, with temperature 0 and top-p 1.
Judge output is an ordered checklist followed by a labeled
`Final Verification: YES` or `NO`, not a numerical G-Eval score.

Anchor/PPL rewards and anchor-only fallback are disabled. The reference/judge
remains pinned to the original Tulu checkpoint, not the updated actor.

Fresh runs disable resume and automatic retry. To explicitly restore a saved
checkpoint, append these final Hydra overrides (and point `CKPT_DIR` to the
intended experiment directory):

```bash
trainer.resume_mode=resume_path \
trainer.resume_from_path=/absolute/path/to/global_step_91
```

Use a fresh W&B run ID when replaying steps older than the original W&B log;
model/optimizer restoration is controlled separately by the checkpoint settings.

## Integration and metrics

`IFLLMVerifierRewardManager` remains in `geval` mode unless explicitly configured
with `if_llm_verifier_mode=intentcheck` (or `IF_LLM_VERIFIER_MODE=intentcheck`).
The numeric threshold is unused in IntentCheck mode. The launcher explicitly
passes the Hydra string `'false'` for judge `enable_thinking`, because the legacy
manager treats the boolean `False` as an unset value.

The parser requires one labeled final verdict. It accepts ordinary explanations
after that verdict but rejects unlabeled votes, ambiguous alternatives, and
multiple final-verification labels. It never uses checklist numbers as scores.

- `if_llm_verifier/score_mean`: YES fraction among valid judgments, including NO=0.
- `if_llm_verifier/pass_rate`: YES fraction among all judge calls, including errors.
- `if_llm_verifier/bonus_mean`: mean bonus over all rollouts, including skipped rows.
- `if_llm_verifier/error_rate`: API/parsing errors among called judgments.

## Tests (no GPU or model/API calls)

```bash
python -m unittest discover -s tests/if_rlvr -p 'test_intentcheck*.py' -v
```

Tests cover the prompt/parser, request format and x-only provenance, complete
reward cases, error handling, default G-Eval compatibility, and binary metrics.
