# Qwen3-4B 20,480-row N=3 anchor package

The public dataset repository is `sangyon/anchor_cache`. This package is stored under `qwen3_4b_reasoning_anchor20k_n3/` so it does not overwrite the three full-population source caches already at the repository root.

## What an anchor draw means

For an instruction `x` and its constraint text `c`, one draw contains two independently sampled responses from the same frozen Qwen3-4B reference policy:

- `y0 ~ p(. | x)`, the unconstrained response used for the lower anchor;
- `y1 ~ p(. | x+c)`, the constrained response used for the upper anchor.

Both token sequences are then scored under `Qwen/Qwen3-4B` with standard-prefix perplexity. NLL and token counts cover final-answer tokens only. A usable cache item therefore contains non-empty `y0` and `y1`, finite `ref0_nll/ref1_nll` and `ref0_ppl/ref1_ppl`, positive token counts exactly matching the token-list lengths, and neither response exceeds 32,768 tokens.

The fixed sampling configuration for missing draws is:

| setting | value |
|---|---:|
| model and scorer | `Qwen/Qwen3-4B` |
| thinking | enabled |
| temperature | 1.0 |
| top-p | 0.95 |
| top-k | 20 |
| presence penalty | 0.0 |
| maximum prompt tokens | 2,048 |
| maximum completion tokens | 32,768 |
| model-level empty-response retries | 0 |

The original full-population caches were generated with an 8,192-token limit plus selected 32,768-token fallback generations. Only the still-missing draws use the fixed 32,768-token protocol above.

## Package layout

`artifacts/` contains the data, not just index lists:

- `selected_train.parquet`: all 20,480 selected raw training examples with stable `if_train_index` and panel labels;
- `selected_audit.parquet`: all selected distribution/audit features;
- `audit_frame.parquet`: the full 93,882-row eligible-population audit frame;
- `anchor_views/run{1,2,3}.SUBSET20480.AVAILABLE.json`: every currently available source anchor for the fixed panel;
- `anchor_views/run1.SUBSET20480.best_available.json`: the 20,449-row valid cache used by the interim run-1 baseline;
- `generated_runs/run1.missing.cache.json`: all 2,664 recorded run-1 one-shot attempts, including the 31 explicitly identified invalid rows;
- `generation_plan.json` and `audit_provenance.json`: workload and source provenance.

`manifests/` contains the fixed selection, panels, exact missing-index sets, cost-balanced audit shards, coverage report, and verification report. `code/` contains the exact preparation, validation, production launch, and token-safe publication scripts. `current_generation_status.json` is the authoritative current-state summary.

An `AVAILABLE` cache is an audit/completion input, not a strict 20,480-row training cache. Never silently treat it as complete.

## Current remaining workload

Run 1 submitted all 2,664 planned missing draws. Of those, 2,633 are train-ready and 31 contain an empty response, so the artifact is preserved but not called complete.

- run 2: 1,059 missing anchor draws = 2,118 responses;
- run 3: 1,066 missing anchor draws = 2,132 responses;
- total currently unattempted: 2,125 draws = 4,250 responses.

Strict N=3 coverage of every one of the 20,480 selected inputs additionally requires an explicit policy decision to regenerate the 31 invalid run-1 attempts. The supplied no-retry launcher intentionally does not make that decision implicitly.

## Reproducing or continuing generation

The authoritative generation utilities are included in the Hub package's `code/` directory and versioned in [the GitHub repository](https://github.com/ssangyeon/if-rlvr-rlvr/tree/main/if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k). They were validated against base Git revision `af34fac4b2dae646d0ddf5afa6dcfad7b4cc0745`; the GitHub publication is a direct descendant of remote-main revision `4c605121dcef612ce3437cae0d02dc0b47a86ec1`. Place the Hub package files back into their corresponding workspace locations:

```text
artifacts/selected_train.parquet
  -> .agent_runtime/subset20k/selected_train.parquet
artifacts/selected_audit.parquet
  -> .agent_runtime/subset20k/selected_audit.parquet
artifacts/audit_frame.parquet
  -> .agent_runtime/subset20k/audit_frame.parquet
artifacts/anchor_views/*
  -> .agent_runtime/subset20k/anchor_views/
artifacts/generated_runs/*
  -> .agent_runtime/subset20k/generated_runs/
manifests/generation_manifests/*
  -> if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k/generation_manifests/
code/*
  -> if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k/
```

The model `Qwen/Qwen3-4B` and dataset `allenai/IF_multi_constraints_upto5` must be available locally. Run the non-GPU preflight first:

```bash
if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k/run_anchor_generation_20k.sh \
  --preflight-only
```

Then launch the production sequence:

```bash
if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k/run_anchor_generation_20k.sh
```

The launcher waits until physical GPUs 4,5,6,7 are free. It uses four complete Qwen3-4B replicas (`tensor_model_parallel_size=1`) and dynamically routes the input batch to the least-loaded replica. It recognizes the existing exact-key run-1 artifact and skips it without retry, then executes run 2 followed by run 3. It fails closed on partial key sets or metadata mismatches.

The four `shard*.indices.json` files are retained for audit and alternate data-parallel implementations. The production launcher does not shard the model and does not start four competing Ray control planes.

## Integrity checks

Core SHA-256 values at publication time:

| file | SHA-256 |
|---|---|
| `selected_train.parquet` | `370c2bec02efd6eb50524a045e210ec60ddb9974bd2befb8ff9e468fcfc580d9` |
| `selected_audit.parquet` | `7fd04533eed1e43f3e2bc07dde47e5be54f9240f01a0c608f60fd14461db820d` |
| `audit_frame.parquet` | `ebef8e36cf8a9c1b61751b50820c89041ae159514ac5182ef1025605df42f909` |
| `run1 AVAILABLE` | `ee3a3863abdb03c51e0a68ee244a884d26109570c11b52e5b5eff010befb0649` |
| `run2 AVAILABLE` | `5b7556066243f42292b036be0e8e791535d74172524b5c7a9202c0da29192f97` |
| `run3 AVAILABLE` | `fd237d72e56a9e7923b01d87d508d285a2520407531a0d0d9cd53a4361686fdd` |
| run-1 one-shot artifact | `ea26cfefed13a19b470ccac1f4a3de2aee35b96d5b7601936ef0bb8ccfc480a0` |
| run-1 best-valid cache | `652b955fefd3df8c9248cac38803f32994868b73ab6ec02c986e278267f7f25a` |
| `subset_indices.json` | `7952fdf77cbf0ca8a09a275a5c3e9cae7198b9e33eb2fa224ac593abb0aecbcb` |

`prepare_anchor_views.py` independently validates selected key sets, cache metadata, token counts, finite NLL/PPL values, exact output lengths, and final N=3 completeness.
