# Review of the Partial v2 Anchor Cache (Qwen3-4B reasoning, B200 build)

*2026-08-13. File reviewed: `sangyon/anchor_cache/if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.json` (557.6 MB snapshot, mid-regeneration). Companions: `if_rlvr_anchor_reward_review_2026-08-11.md` (v1 failure analysis), `anchor_subset_curation_design_2026-08-13.md` (4,096-row subset).*

## Verdict in one paragraph

The v2 build is **fundamentally sound** — correct configuration, correct scoring semantics, genuinely fresh generations — but this file is a **mid-process snapshot**: the hygiene loop has not run, 13.5% of rows are empty for a *systematic budget* reason that plain re-running only partially fixes, and two expectations from the v2 recipe are now measured false on qwen3-4b (loops were not eliminated by the sampling change; inversions are unchanged — the latter exactly as the redesign analysis predicted: they are rule geometry, not data hygiene). The 4,096-row subset is 86.9% covered; the 538 holes are systematically long-form prompts, so paired ablations should wait for those to fill (priority index list provided).

## 1. What is verified GOOD

| check | result |
|---|---|
| metadata | exact v2 targets: `temp 1.0, top_p 0.95, top_k 20`, budget 8192, `ppl_prefix_mode=standard`, `ppl_nll_scope=final_answer_tokens_only`, v3, `Qwen/Qwen3-4B` |
| budget-match premise | **confirmed against wandb configs**: every qwen3-4b thinking training run used `data.max_response_length = 8192` |
| content freshness | 99.34% of complete rows differ from v1 byte-wise (537 "identical" rows are 1–5-token answers regenerated to the same ids; 93% of those have different NLLs → independently rescored) |
| internal consistency | `ppl == exp(nll/count)` on 100.000%; `token_count == len(ids)` on 100.000% |
| think-strip / CoT leakage | 0.027% (v1: 0.44%) — `FINAL_ANSWER_ONLY` worked |
| scoring semantics | m0 shifted −0.08 nats *uniformly across all length sextiles* vs v1 — the expected effect of truncated sampling producing more-typical text; no level jump → same prefix/scope as v1 |
| length regime | y1 p50 201 (v1: 202), y0 p50 474 (v1: 484); the v1 >8k tail is gone (5 residual rows >8192 are decode/re-encode token-count inflation, not real generations) |

## 2. Issue — the 12,838 empties are systematic (budget), not sampling noise

- File universe = **exactly v1's 93,882 indices** (the 979 v1-missing rows were never attempted). Of these: 81,044 complete, **12,239 empty y0 vs 599 empty y1** — a 20:1 asymmetry.
- Mechanism: y0 answers x *alone*; on open-ended prompts qwen3-4b thinks longest (often drafting the full answer inside `<think>`), exhausts the 8,192 budget before `</think>`, and `FINAL_ANSWER_ONLY` correctly refuses the raw CoT. Three attempts (1 + 2 retries) all failed.
- Systematicity [measured]: missing rows' v1 final answers are ~2× longer (y0 p50 903 vs 484; y1 433 vs 202) and less compliant (v1 allsat 22.9% vs 30.4%). These are the long-form prompts.
- **This boundary is shared with the policy**: training also runs at 8,192 with `IF_REQUIRE_THINK_END_FOR_REWARD=true`, so on these prompts most policy rollouts are reward-ineligible anyway. Excluding a residual unfillable tail via `TRAIN_CACHED_ONLY` is principled. **Do not raise the anchor budget** — that breaks the anchor/policy regime match, which is the core of the v2 fix.
- Regeneration guidance: raise `IF_ANCHOR_PRECOMPUTE_EMPTY_RESPONSE_RETRIES` to ~4–6 for refill passes; track marginal fill per pass; stop when a pass fills <1% and record the residual as the "8k-infeasible set."

## 3. Issue — sampling did NOT eliminate loops on qwen3-4b (correction to the v2 recipe's expectation)

- Fresh loop-flagged y1: 4.20%. Adjudicated with the repo's own IFEval checkers: **32.5% are legitimate repetition** (copy/combination/new constraint families, or the answer passes all its constraints), **67.5% true degeneration = 2.83% of all fresh rows**.
- v1's analogous true-degeneration is ≈2.9–3.6% (its 5.4% headline was also ~46% legitimate-ish repetition under a text-based criterion). Net: `top_p 0.95 / top_k 20` — which collapsed Llama degeneration to ≈0 — **transferred weakly at best to qwen3-4b thinking**.
- Consequences:
  1. **The hygiene detector must be fixed before running the hygiene loop**: as scripted it deletes any loop-flagged y1 and regenerates — but for copy/repeat-family prompts the *correct* answer is repetitive, so those rows would churn through all passes and be dropped. Fix: for y1, delete only `loop ∧ fails-own-constraints`; keep y0 loop deletion unconditional (y0 has no constraints; fresh y0 loops 0.85%, all degenerate).
  2. For the typicality-z redesign: the "source control" mitigation layer for the self-reinforcing-loop blind spot is **weaker than designed** on this model. The optional absolute backstop (open question 3 of the redesign doc) gains importance; hybrids remain covered by the judge.

## 4. Non-issue — inversions unchanged (12.84% vs v1 12.54%): expected, and newly informative

- Regeneration cannot remove flips; they are mean-NLL length geometry, exactly as the failure analysis said. Anyone reading flipped pairs as a generation defect should stop worrying: the fixed rule ignores invalid bands (keeps IF), and the z-redesign has no bands to invert.
- New measurement from the v1↔v2 pairing: **P(v2 inverted | v1 inverted) = 49.8%** vs P(v2 inverted | v1 valid) = 7.5% — inversion is about half row-intrinsic (prompt geometry), half draw-luck. Keep flipped rows in the cache and in the subset (they are preserved proportionally by design).

## 5. Housekeeping — this snapshot has not been through hygiene

Present among complete rows: short y0 2.43%, short y1 4.02%, dup y0==y1 0.155%, loops per §3. Run the (fixed) hygiene pass after refills complete; it deletes true-degen only and the next build pass refills.

## 6. Strict-metadata landmines for future training runs

The trainer's strict check compares the **full metadata dict** (`_if_ref_anchor_cache_metadata`, `ray_trainer.py:2174`). Two guaranteed mismatches on this box:

1. `tokenizer_class`: file says `Qwen2Tokenizer` (the B200 env used the slow tokenizer); this box computes `Qwen2TokenizerFast`. Cosmetic (identical vocab; scoring-side tokenization was done by the vLLM server), but full-dict equality fails.
2. `train_sample_count`: file says 93,882; a subset training run computes `len(train_dataset)` = 4,096 after filtering. The field encodes the *build* universe, not the *load* universe — structurally incompatible with subset workflows.

Options before the first v2 training run: (a) launcher-side field-by-field assertion that ignores `{tokenizer_class, train_sample_count}` with `IF_REF_ANCHOR_CACHE_METADATA_STRICT=false`; or (b) a small loader patch whitelisting informational fields. Either preserves the protection that matters (sampling, budgets, prefix mode, scope, model).

## 7. Subset fit (the operative question)

- Coverage today: **3,558 / 4,096 (86.9%)**; 538 missing (13.13%), roughly uniform across strata (clean 13.8%, degen_y1 13.0%, nc=5 18.4%; halves 282/256).
- The covered part is still representative of the v2-complete population (KS ≤ 0.012 on m0/m1/width/lengths; inversions 13.15% vs 12.81%; loops 3.8% vs 4.2%).
- **But** the 538 holes are the systematically long-form prompts (§2) — running the paired v1-vs-v2 ablation on covered rows only would under-sample exactly the regime where the v1 floor misbehaved most (long answers). **Fill the holes first.**
- Priority list shipped: `if_rlvr/data_subsets/qwen3_4b_reasoning_anchor4k/v2_missing_subset_indices.json` (538 indices). Filling these first unblocks the paired subset in hours instead of waiting for the full 12,838.
- The curation itself needs no rework: same indices, same strata, same halves; when coverage reaches 4,096, carve the v2-subset by filtering items — identical procedure to the v1-subset file.

## 8. Recommended order of operations

1. Point the ongoing regeneration at the 538 subset indices first, then the remaining ~12.3k (optionally also the 979 never-attempted v1-missing rows).
2. Refill passes with retries raised to 4–6; measure marginal fill; accept the infeasible residual. Budget stays 8,192.
3. Fix the hygiene detector (compliance-aware y1 loop deletion) before any hygiene pass runs.
4. Keep flipped pairs — rule-level concern, already handled.
5. Decide strict-metadata handling (§6) before the first v2 training run.
6. When refills + hygiene are done: re-run this audit as the acceptance gate, carve the v2-subset, and start the paired ablations.

## Appendix: artifacts

- Audit scripts + raw reports: `scratchpad/v2review/{audit_v2_cache.py, audit_report.txt, mixture_report.txt, loop_adjudication.txt}` on the training box; per-row v2 frame: `v2review/v2_frame.parquet` (81,044 rows with lengths, NLLs, defect flags, carryover flag).
- Loop adjudication table: `v2review/loop_adjudication.parquet` (3,378 rows: rep-family flag, ifscore, allsat).
- Subset priority list: `if_rlvr/data_subsets/qwen3_4b_reasoning_anchor4k/v2_missing_subset_indices.json`.
