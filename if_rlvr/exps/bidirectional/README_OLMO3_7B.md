# OLMo3 7B IF-RLVR

These launchers adapt the existing IF-RLVR anchor-precompute and GRPO flow to
`allenai/Olmo-3-7B-Instruct-DPO` on GPUs `0,1,2,3`.

Do not start these launchers while another job is using GPUs `0,1,2,3`. The
OLMo wrappers call the existing model-generic anchor launcher (whose historical
filename still begins with `qwen3_`) and give their Ray/vLLM run a distinct ID.

The model needs Transformers 4.57 or newer. The shared verl environment has
Transformers 4.56.1, so `setup_olmo3_runtime.sh` creates an isolated overlay
venv. It reuses PyTorch, vLLM, Ray, and the remaining packages from the existing
verl environment without changing that environment. The setup is idempotent and
skips pip when the validated 4.57.1 overlay is already present.

```bash
cd /NHNHOME/NHNHOME/WORKSPACE/26msit001_T_A/IFIF/if-rlvr

bash if_rlvr/exps/bidirectional/setup_olmo3_runtime.sh

bash if_rlvr/exps/bidirectional/precompute_teacher_olmo3_7b_instruct_dpo_anchor_scored_by_olmo3_7b.sh \
  2>&1 | tee precompute_teacher_olmo3_7b.log

bash if_rlvr/exps/bidirectional/olmo3_7b_t7b_anchor_grpo_nonreason.sh \
  2>&1 | tee olmo3_7b_grpo.log
```

For a constraint-only control run that does not read or generate an anchor
cache:

```bash
bash if_rlvr/exps/bidirectional/olmo3_7b_constraint_only_nonreason.sh \
  2>&1 | tee olmo3_7b_constraint_only.log
```

The default anchor reward coefficient is `0.1`. Override it without editing the
script, for example:

```bash
PY_GIVEN_X_REWARD_COEFF=0.05 \
bash if_rlvr/exps/bidirectional/olmo3_7b_t7b_anchor_grpo_nonreason.sh \
  2>&1 | tee olmo3_7b_grpo_pyx005.log
```

The default cache is:

```text
.cache/if_ref_anchor_teacher_olmo3_7b_instruct_dpo_nonreason_train_seed1_val512_scored_by_olmo3_7b_instruct_dpo.json
```

Model-specific choices:

- no Qwen `enable_thinking` chat-template kwarg;
- the whole response is treated as the final answer because OLMo Instruct has no
  `</think>` marker;
- BF16 actor/reference weights match the published checkpoint dtype;
- vLLM tensor parallelism is 1, giving one replica per B200;
- rollout settings match the latest Qwen3-8B profile in this workspace:
  `max_model_len=40960`, `max_num_batched_tokens=8192`,
  `max_num_seqs=1024`, and `gpu_memory_utilization=0.8`.
