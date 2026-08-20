# Representative Qwen3-4B 20,480-row anchor panel

This directory defines the fixed training population for the ~20K scale experiments. The primary artifact is `subset_indices.json`; it contains exactly 20,480 unique post-split training indices and includes every row in the previously validated 4,096-row panel.

## Population being represented

The pinned `allenai/IF_multi_constraints_upto5` snapshot has 95,373 rows. Applying the established seed-1 shuffle and 512-row validation holdout leaves 94,861 nominal training rows. Under the exact Qwen3 thinking chat template, 979 of those prompts exceed the configured 2,048-token prompt limit. Those 979 indices exactly equal the indices absent from the full run-1 cache, so the effective population seen by the configured trainer is 93,882 rows.

The 20,480-row target is 21.81% of that effective population. It was chosen instead of a rounded 20,000 because it is five panels of 4,096 and is divisible by the established 256/512/1,024 batch units.

## Selection design

Panel A is the immutable historical 4,096-row subset. The selector adds 16,384 rows without replacing any panel-A row.

Selection uses all three pinned anchor draws. Exact or hierarchical stratification covers:

- run-1/run-2/run-3 availability bit pattern;
- canonical anchor hygiene class (missing, short, suspicious loop, duplicate, CoT marker, or clean);
- number of flipped draws among the available runs;
- number of constraints;
- quintile of constraint-ID-pooled IFEval difficulty;
- whether no, some, or all available constrained anchors fully satisfy their constraints.

Availability pattern is never merged away. Sparse cells are hierarchically collapsed only until their expected subset count reaches five. Integer stratum allocation minimizes squared error from the proportional full-population allocation while respecting mandatory inclusion of panel A.

Within those allocations, 500 seeded rerandomized candidates are compared, followed by 30,000 fixed-preserving swaps. The objective matches 24 continuous distributions (prompt and constraint length, script ratios, multi-run IFEval difficulty, NLL geometry, output length, cross-run variance, and the N=3 CI95 endpoints) plus constraint IDs, families, the top 100 ID pairs, 55 source/domain buckets, and the top 200 exact constraint signatures.

This design intentionally preserves missing/failed anchor rows instead of selecting only convenient complete anchors. Generation failure is correlated with hard, long, or pathological examples; conditioning on complete N=3 data would make the 20K set artificially easy.

## Acceptance result

All whole-subset thresholds were fixed in the selector before the production search. The K-S target of 0.006 is stricter than the approximate unadjusted 5% two-sample critical value of 0.0107 for 20,480 versus 93,882 rows. K-S is used here as a distribution-distance criterion, not as a post-selection hypothesis-test p-value.

| check | result | threshold |
|---|---:|---:|
| maximum continuous K-S distance | 0.004107 | 0.006000 |
| constraint-ID total variation | 0.000203 | 0.001500 |
| constraint-family total variation | 0.000090 | 0.001000 |
| top-100 constraint-pair TV | 0.002275 | 0.005000 |
| source/domain TV | 0.001211 | 0.005000 |
| top-200 signature TV | 0.008118 | 0.010000 |
| anchor-availability-pattern TV | 0.003320 | 0.005000 |

The residual availability-pattern distance is irreducible while retaining panel A: panel A already contains 168 run-1-only rows, whereas a perfectly proportional 20,480-row sample would contain about 100. The final set contains exactly those 168 and does not add more.

The complete comparison, including every continuous variable, per-run flips/IFEval/all-satisfied rates, hygiene classes, constraint IDs, and correlation replay, is in `verification_report.md`. The primary whole-set correlations are also retained: for example, constrained-anchor NLL versus output length is -0.6240 in the subset versus -0.6217 in the full population.

Panels B-E are storage/paired-analysis partitions, not four independently optimized 4K substitutes. The 20,480-row union is the statistically accepted object; panel A remains available only for exact continuity with prior experiments.

## Anchor coverage and future generation

The selected inputs are fixed independently of completion work:

| draw | currently complete | missing |
|---|---:|---:|
| run 1 (with verified panel-A override) | 17,816 | 2,664 |
| run 2 | 19,421 | 1,059 |
| run 3 | 19,414 | 1,066 |

There are 17,446 rows with all three draws, 19,749 with at least one draw, and 731 with no complete draw. In total, 3,034 unique selected inputs need at least one additional draw before a strict N=3 experiment; there are 4,789 run-specific missing draws.

`prepare_anchor_views.py prepare` writes:

- `.agent_runtime/subset20k/selected_train.parquet`, the raw selected examples with stable `if_train_index` and `subset20k_panel` columns;
- three `*.SUBSET20480.AVAILABLE.json` caches, which are audit/carving artifacts and are deliberately labeled **not train-ready for 20,480 rows**;
- `anchor_coverage.json` and `generation_manifests/run{1,2,3}/`, the exact missing sets and four disjoint cost-balanced data shards per run.

Rows with no observed draw receive the full two-response 32K budget as their static balancing cost. The four cost-balanced shard manifests remain useful audit artifacts, but they are not the production launch topology.

The exact missing workload is 4,789 **anchor draws**. Every anchor draw produces one unconstrained response `y0 ~ p(.|x)` and one constrained response `y1 ~ p(.|x+c)`, so completing the panel requires 9,578 full vLLM completions plus 9,578 prompt-logprob scoring calls. The three jobs contain 2,664, 1,059, and 1,066 draws respectively.

Production completion uses one isolated Ray job per run. Each job exposes physical GPUs 4,5,6,7, sets tensor parallel size to one, and therefore creates four complete Qwen3-4B vLLM replicas. The full run-specific input batch is dynamically routed by verl's global least-inflight load balancer. This provides data parallelism without starting four competing Ray control planes.

Run the read-only preflight first:

```bash
if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k/run_anchor_generation_20k.sh \
  --preflight-only
```

After intentionally allocating GPUs 4-7, the same entry point runs all three jobs and finalizers sequentially:

```bash
if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k/run_anchor_generation_20k.sh
```

The fixed generation configuration is Qwen3-4B thinking mode, temperature 1.0, top-p 0.95, top-k 20, presence penalty 0, a 2,048-token prompt limit, and a 32,768-token completion limit. vLLM uses `gpu_memory_utilization=0.55`, `max_num_seqs=128`, and `max_num_batched_tokens=65536`. The explicit KV-cache limit leaves enough transient VRAM for vLLM's float32 prompt-logprob tensor when scoring a full-length Qwen3 continuation; BF16 FSDP initialization retains additional NCCL headroom. Precompute batches of 288, 384, and 328 rows persist completed work during run 1, run 2, and run 3 without sleeping or reloading the four replicas between batches. Every full and final batch is divisible by its run's worker count, preventing verl's divisibility padding from submitting hidden duplicate requests.

This campaign has zero semantic or process retries: empty-response retries are zero and both launcher and campaign attempt each absent run only once. An existing output with every declared key is idempotently skipped, even if some rows are not train-ready. A structurally invalid or partial-key output causes an immediate fail-closed stop. A newly completed exact-key run with unusable rows is preserved while the other run-specific attempts continue, ensuring all 4,789 planned draws are submitted before strict readiness is assessed. Logs, one-minute GPU telemetry, status, and the machine-validated workload plan are written under `.agent_runtime/subset20k/`. Ray temporary state uses the short `.rt20k/` path on the spacious `/data` filesystem rather than the nearly full root `/tmp`; preflight requires at least 100 GiB free there.

`finalize-run` refuses to emit a final cache unless the generated keys exactly equal the predeclared missing set and every item is train-ready. If all three runs pass, `validate-complete` requires all three final caches to contain the exact fixed 20,480-row selection. If any one-shot row is unusable, the campaign finishes the remaining draws, records `generation_complete_incomplete`, exits 5, and submits no retry. The legacy `generate_missing_shard.sh` wrapper is retained only for provenance and must not be used for this campaign because it contains retry and hygiene loops.

Do not train with an `AVAILABLE` cache while expecting 20,480 rows. Use only a finalized cache and set `IF_EXPECT_TRAIN_ROWS=20480` plus `IF_REF_ANCHOR_TRAIN_CACHED_ONLY=true`.

### Recorded state on 2026-08-20

Run 1 has now submitted every one of its 2,664 originally missing anchor draws. The one-shot artifact contains the exact declared key set; 2,633 rows are train-ready and 31 rows have an empty `y0` and/or `y1`, so it is deliberately retained as an audited attempt rather than mislabeled as a complete 20,480-row cache. The best valid run-1 training view contains 20,449 rows.

Run 2 and run 3 have not started their missing-row generation. They still require 1,059 and 1,066 anchor draws respectively: 2,125 draws, or 4,250 model responses, in total. The production launcher detects the recorded run-1 artifact, does not resubmit it, and proceeds with run 2 and then run 3 after GPUs 4-7 become free.

The public Hub package, file mapping, exact remaining workload, and restart procedure are documented in `HUGGINGFACE_GENERATION_GUIDE.md`. Machine-readable status is in `current_generation_status.json`.

## Reproduction and integrity

The production selection is deterministic with seed 20260819:

```bash
/data/IFIF/.miniforge3/envs/verl/bin/python \
  if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k/curate_anchor_subset.py \
  --audit .agent_runtime/subset20k/audit_frame.parquet \
  --fixed-manifest if_rlvr/data_subsets/qwen3_4b_reasoning_anchor4k/subset_indices.json \
  --output-dir if_rlvr/data_subsets/qwen3_4b_reasoning_anchor20k \
  --selected-audit .agent_runtime/subset20k/selected_audit.parquet \
  --draws 500 --swaps 30000
```

A clean replay produced byte-identical artifacts:

- `subset_indices.json`: `7952fdf77cbf0ca8a09a275a5c3e9cae7198b9e33eb2fa224ac593abb0aecbcb`
- `train_indices.json`: `a634e56fde31fbfb92eb374b60dff41dd51399a05779b234d6d498baa37a8a33`
- `selected_audit.parquet`: `7fd04533eed1e43f3e2bc07dde47e5be54f9240f01a0c608f60fd14461db820d`

The audit frame is pinned by dataset, tokenizer, source revisions, metadata, and SHA256 checks. N=3 still gives noisy per-input estimates; this curation therefore claims close empirical distribution matching, not perfect recovery of latent per-example difficulty. Important multi-run joints are protected by hard strata, while the report makes the remaining marginal and correlation distances explicit.
