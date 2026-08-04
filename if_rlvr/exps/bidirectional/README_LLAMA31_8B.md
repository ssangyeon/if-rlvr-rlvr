# Llama 3.1 8B IF-RLVR

These launchers run the existing IF-RLVR precompute and GRPO pipelines with
`meta-llama/Llama-3.1-8B-Instruct` on GPUs `0,1,2,3`.

The model is gated. Before the first run, the Hugging Face account must have
accepted the model license and the container must receive `HF_TOKEN` or already
be authenticated with `hf auth login`.

At launch, the selected script downloads the model once with
`huggingface_hub.snapshot_download`, validates every safetensors shard, and then
passes the resolved local snapshot to Ray/vLLM. This avoids concurrent Hub
downloads from the four GPU workers. Meta's duplicate `original/*.pth` files
are intentionally excluded.

```bash
cd /NHNHOME/WORKSPACE/26msit001_A/IFIF/if-rlvr

# 1. Generate and save the anchor cache.
bash if_rlvr/exps/bidirectional/precompute_teacher_llama31_8b_instruct_anchor_scored_by_llama31_8b.sh

# 2a. Constraint-only control (independent of the cache).
bash if_rlvr/exps/bidirectional/llama31_8b_constraint_only_nonreason.sh

# 2b. Anchor p(y|x)=0.1 run (requires step 1).
bash if_rlvr/exps/bidirectional/llama31_8b_t8b_anchor_pyx01_nonreason.sh
```

The default anchor cache is:

```text
.cache/if_ref_anchor_teacher_llama31_8b_instruct_nonreason_train_seed1_val512_scored_by_llama31_8b_instruct.json
```

Llama-specific behavior:

- no Qwen `enable_thinking` chat-template argument;
- no `</think>` requirement; the complete response is the final answer;
- BF16 actor/reference weights;
- rollout TP=1 on four GPUs;
- batch size 1024, response length 2048, rollout `n=8` for GRPO;
- temperature 1.0, top-p 1.0, and no top-k filtering.
