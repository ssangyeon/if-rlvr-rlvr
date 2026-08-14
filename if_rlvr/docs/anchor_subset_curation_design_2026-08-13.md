# Representative 4k Subset of the Qwen3-4B Reasoning Anchor Cache — Curation Design

*2026-08-13. Companion to `if_rlvr_anchor_reward_review_2026-08-11.md`. All pilot numbers below are measured on the real frame (not simulated).*

## 1. Goal

A ~4k-prompt subset of the existing anchor cache (`sangyon/anchor_cache/if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.json`, 93,882 rows) that is **distributionally indistinguishable from the full training set** on every axis the anchor experiments care about, so that reward-rule ablations, the typicality-z pilot, hygiene ablations, and the v2 anchor rebuild can iterate ~23× faster with paired, comparable results.

Requirements set by the team:

1. same **difficulty** profile (the IF-reward accuracy eval);
2. same **anchor NLL/PPL geometry**, *including* the problematic mass (inverted pairs, degenerate anchors, near-empty rows) at identical proportions — hygiene ablations must be runnable *on the subset*;
3. same **anchor output-length** distributions;
4. same **input composition** (constraint types/counts, instruction lengths, co-occurrence structure).

## 2. Sampling frame and drop-in mechanics

- **Frame** = the 93,882 cache rows (train split of seed-1 shuffle, val 512 held out). The 979 train rows missing from the cache are excluded (no anchors); their indices are spread uniformly over the index range (deciles 21 / 23,444 / 46,941 / 70,661 / 94,849), so no systematic hole is inherited.
- **Join key**: the stable per-row `index` (position in the seed-1-shuffled train split; `if_dataset.py:_to_verl_row`). Cache items are keyed by this index.
- **Drop-in**: `IF_REF_ANCHOR_TRAIN_CACHED_ONLY=true` filters the training set to exactly the indices present in the cache JSON at `IF_REF_ANCHOR_CACHE_PATH` (`if_dataset.py:127`). **The subset therefore ships as one filtered cache file — zero trainer/code changes.** The subset JSON copies the full cache's `metadata` block verbatim so strict metadata checks and PPL-semantics checks pass unchanged.
- The same 4,096 prompts serve all ablation types: trainer-rule ablations (old cache), the v2 anchor rebuild (regenerate anchors for these prompts only — ~23× cheaper), and ΣH/ΣV z-score validation. Because every arm sees literally identical prompts, arm-vs-arm deltas are **paired**, not resampled — much tighter than comparing two independent draws.

## 3. Size: n = 4,096

- Divides batch 256 / 512 / 1024 exactly → 16 / 8 / 4 steps per epoch (recommended ablation batch: 256 or 512 for step granularity).
- Sampling rate 4.36%; the rarest of the 54 instruction IDs (`copy:copying_simple`, 0.70% of constraint slots) still receives ~74 expected slots — no type starves.
- 3,072 (= 3 × 1024) is the in-range alternative if a strictly ≤ 4k count is preferred; all guarantees below hold at either size.

## 4. What is preserved exactly (stratification key)

Proportional stratified allocation (largest-remainder) over the cross of:

| variable | levels | why exact |
|---|---|---|
| **defect class** (mutually exclusive, priority order) | near-empty y0 (<10 tok, 2.55%) → near-empty y1 (3.98%) → y1 loop (`y1_degen`, 5.43%) → y0 loop (0.43%) → y0==y1 dup (0.39%) → CoT leak (≈1.1%) → clean | hygiene ablations need identical defect mass (team requirement) |
| **inverted** (m1 < m0) | 12.67% | the flipped-pair pathology, preserved to the row |
| **n_constraints** | 1–5 (24.1 / 25.1 / 24.5 / 18.9 / 7.4%) | difficulty scaling |
| **difficulty quintile** | type-pooled expected compliance (below) | difficulty profile |
| **y1_allsat** | 29.36% | bonus-semantics split (compliant vs non-compliant reference) |

Cells with expected subset count < 6 merge hierarchically (drop `y1_allsat`, then difficulty, then n_constraints) but **never across defect × inverted boundaries**. Result on the real frame: 164 strata, 89% of rows in unmerged full-key cells.

**Difficulty is init-policy difficulty, not a proxy.** The anchor teacher *is* the policy initialization (Qwen3-4B, thinking mode), so y1 = the untrained policy's own single-draw answer to x+c, and per-row `y1_ifscore`/`y1_allsat` = the policy's pass@1 on the IF reward at step 0. The stratification score pools this per instruction ID over the full frame (per-ID mean compliance, range 0.000 `new:copy_span_idx` → 0.913 `detectable_content`) and averages over each row's constraints; the raw per-row draw is preserved via the `y1_allsat` stratum and the `y1_ifscore` distribution check.

## 5. What is matched statistically (two-stage selection)

Within strata, selection is **rerandomized then polished**:

1. **Stage 1 — rerandomization**: M = 20,000 candidate within-stratum draws, each scored on a pre-registered composite (tie-exact KS on 11 continuous variables + total-variation distance on the 54-ID marginal, 16-family marginal, top-50 ID-pair co-occurrence + a hard worst-ID coverage floor of 0.5× expected). Keep the argmin.
2. **Stage 2 — greedy swap polish**: 60,000 proposed single-row swaps, each **within one stratum** (all exact-match guarantees from §4 are preserved by construction); accept iff the composite improves. Deterministic under the fixed seed.

Balanced variables: `m0`, `m1`, band width, log y0-length, log y1-length, log length-ratio, `y1_ifscore`, `p_zero`, `p_bonus` (the audit's modeled per-row floor/bonus probabilities under the current rule), x length (constraint-free instruction chars), difficulty score; plus the three categorical marginals.

## 6. Measured pilot results (real frame, seed 20260813)

Composite: random median 0.517 → rerandomized 0.380 → polished **0.0635**.

| check | full set | subset | tolerance context |
|---|---|---|---|
| max KS D over all 11 variables | — | **0.0053** | detectability threshold at n=4096: 0.0212 |
| TV, 54-instruction-ID marginal | — | **0.0007** | ≈ exact composition |
| TV, 16-family marginal | — | 0.0003 | |
| TV, top-50 ID pairs | — | 0.0066 | |
| worst-ID coverage | — | 0.99× expected | hard floor 0.5× |
| inverted rows | 12.67% | 12.70% | exact by construction (rounding) |
| false-wipe of compliant y1 (P(m1<m0 \| allsat)) | 11.51% | 11.35% | the doc's 11.5% headline |
| p_zero mean | 0.1870 | 0.1870 | |
| p_bonus mean | 0.3293 | 0.3293 | |
| y1_allsat | 29.36% | 29.25% | |
| y1_ifscore mean | 0.5724 | 0.5722 | |
| Spearman(m1, log y1-len) | −0.806 | −0.805 | the length-proxy signature |
| Spearman(width, log ratio) | −0.602 | −0.589 | |
| inversion by length-ratio octile | 1.2→47.8% | 1.9→46.1% | monotone gradient reproduced |

Interpretation: with 4,096 rows one cannot statistically distinguish the subset from the full set on any measured axis, and the reward rule's aggregate behavior (zeroing mass, bonus mass, inversion structure, the length-proxy correlation) replays to within a fraction of its own sampling noise.

## 7. Verification battery shipped with the artifact

The final curation run auto-generates a report with: all §6 checks; per-defect-class counts vs expected; per-ID counts (54 rows); n_constraints × difficulty cross-table; QQ-plots data for m0/m1/width/lengths; and the functional replay of both the July (buggy) and fixed reward rules. Acceptance thresholds are pre-registered in the script header (KS < 0.01 per variable, TV(ID) < 0.005, every §6 functional stat within its full-set 95% CI).

## 8. Artifacts

```
if_rlvr/data_subsets/qwen3_4b_reasoning_anchor4k/
  curate_anchor_subset.py      # deterministic (fixed seed), reads audit parquet + HF dataset + full cache
  subset_indices.json          # 4,096 indices + stratum labels + per-row key features
  verification_report.md       # auto-generated battery of §7
.cache/if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.SUBSET4096.json
                               # filtered items + verbatim metadata  → the drop-in file
```

Usage in any existing launcher:

```bash
export IF_REF_ANCHOR_CACHE_PATH=.../...SUBSET4096.json
export IF_REF_ANCHOR_TRAIN_CACHED_ONLY=true
# optional: train_batch_size 256  → 16 steps/epoch
```

Optionally uploaded to `sangyon/anchor_cache` next to the full file so the B200 box can use it directly. Optionally split into two balanced 2,048 halves (within-stratum pairing on m1) for smoke tests.

## 9. Honest limitations

1. **Distribution ≠ dynamics.** The subset guarantees the *data distribution* matches; it cannot make a 16-step epoch behave like a 91-step epoch (each prompt is revisited ~23× more often per wall-clock; multi-epoch memorization appears earlier; per-step batch composition is noisier). Read subset ablations as **paired arm-vs-arm comparisons under identical data**, not as absolute forecasts of full-run curves; graduation criterion for any winning variant remains a full-set run.
2. `p_zero`/`p_bonus` are audit-modeled probabilities under the current rule at initialization — excellent matching targets, not live-run ground truth.
3. Difficulty is init-policy difficulty (see §4); difficulty relative to the *evolving* policy shifts during training in ways no static subset can pin.
4. The 979 cache-missing rows are excluded by necessity (1.03%, uniformly spread; measured, not assumed).
5. The `cot_leak` and `dup` classes are small (≈1.1% / 0.39%); at n=4,096 they carry ~45 / ~16 rows — proportions right, per-class statistical power limited (fine: they exist to keep hygiene ablations honest, not to be analyzed alone).

## 10. Open decisions

1. n = 4,096 (recommended) vs 3,072.
2. Produce the balanced 2,048 halves? (free)
3. Upload the subset cache JSON to `sangyon/anchor_cache`? (needed for the B200 box)
4. Immediately build the **v2 anchors for these 4,096 prompts** once GPUs free (~hours instead of days) so old-cache and v2-cache ablations run paired on identical prompts?
