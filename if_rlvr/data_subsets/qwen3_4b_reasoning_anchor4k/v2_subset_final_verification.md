# Final v2 Subset (4,096 rows) — Verification Record

*2026-08-14. Final file: `.cache/if_ref_anchor_teacher4b_reasoning_train_seed1_val512_t1_p095_k20_pp0_r8192_scored_by_qwen3_4b.SUBSET4096.json` (27.5 MB). Provenance sidecar: `v2_subset_provenance.json`.*

## Composition

| source | rows | generation budget |
|---|---|---|
| regen2 (main v2 build) | 3,620 | 8,192 (matched to training) |
| targeted fill `...v2_subset476_r32768...` | 476 | **32,768** (the think-budget-infeasible tail; retries=8) |

Zero conflicts between sources (the 476 were exactly the rows absent/empty in regen2). The final file carries regen2's metadata block verbatim; the mixed-budget provenance is recorded per-row in the sidecar.

## 476-fill audit (all pass)

- Indices match the delivered work-order exactly (476/476); zero incomplete.
- Internal consistency 100%: `token_count == len(ids)`, `ppl == exp(nll/count)`.
- Config verified from metadata: t1.0 / top_p 0.95 / top_k 20, `ppl_prefix_mode=standard`, `ppl_nll_scope=final_answer_tokens_only`, seed-1/val-512, `Qwen/Qwen3-4B` — everything except the budget matches the v2 recipe.
- Hygiene: 0 dups, 1 CoT-leak (0.2%), 7 short y1, 21 loop-flagged y1 of which 5 legitimate repetition and **16 true degeneration (3.4%** vs 2.8% in the main population — expectedly slightly higher on hard rows).
- **The 32k budget bought thinking space, not runaway answers**: every y0 final answer ≤ 2,350 tokens (all within the policy's 8k regime); y1 p50 = 450, with a tail of **20 rows > 8,192 tokens** (max 32,366) — 0.49% of the subset carries an anchor answer longer than the policy can produce.
- Scoring-semantics continuity: m0/m1 levels sit on the population length-NLL curve (m0 mean 0.458 at y0 p50 957 — exactly where the length law puts long rows); inversion rate 13.4% ≈ population.

## Final battery

- **Trainer loader accepts all 4,096 rows** (`_complete_anchor_cache_indices`); launcher preflight fields verified.
- **Regime-matched representativeness** (subset minus the 476 fills vs the full v2-complete population): KS ≤ 0.0117 on m0, m1, width, log-lengths — statistically indistinguishable, unchanged from the pre-fill check.
- Whole subset (incl. the 32k tail) vs full-v2-complete: KS 0.029–0.041 on m0/m1/width. **This gap is coverage, not bias**: the subset contains the long-form tail that the *full* v2 file still lacks (its counterpart rows remain unfilled there). Relative to the true 94k prompt distribution, the subset is now more complete than the full v2 cache itself.
- Inversions: subset 11.8% vs full-v2 11.2%. Band geometry of the fills: narrower bands (width p50 0.47 vs 0.84) with a low floor — on these rows the anchor mostly abstains by geometry (policy rollouts sit above the low floor, bonuses rare), i.e., the off-regime rows degrade gracefully toward IF-passthrough rather than injecting wrong punishments.
- The prompt-level curation guarantees (KS ≤ 0.0053, TV(54-type) = 0.0007, exact defect/flip strata, functional replay) are untouched — indices never changed.

## Known caveats

1. 476 rows (11.6%) carry anchors generated at 4× the training budget; documented per-row in the sidecar. Any full-scale v2 run must either fill those rows the same way or accept their absence — do not mix silently.
2. 20 rows have y1 longer than the policy's total budget (unreachable-length ceiling; behaves as no-bonus).
3. The 16 true-degeneration loop rows were retained deliberately (defect strata are part of the design; hygiene ablations need them).
