# IF-RLVR Anchor Rewards: Method, Failure Analysis, and Proposed Redesign

*Working document for internal review — 2026-08-11.*
*Assumes no prior knowledge of the project. All numbers are measured on our own runs/data unless labeled otherwise; every claim is tagged **[proof]** (mathematical), **[measured]** (verified on our data), or **[pilot]** (requires the proposed validation run).*

---

## Table of contents

1. [Problem setup and intuition](#1-problem-setup-and-intuition)
2. [The three reward arms](#2-the-three-reward-arms)
3. [The anchor method in full detail](#3-the-anchor-method-in-full-detail)
4. [The observed problem](#4-the-observed-problem)
5. [Investigation and findings](#5-investigation-and-findings)
6. [Root-cause synthesis](#6-root-cause-synthesis)
7. [Proposed redesign: typicality-standardized anchors](#7-proposed-redesign-typicality-standardized-anchors)
8. [End-to-end pipeline specification](#8-end-to-end-pipeline-specification)
9. [Validation plan](#9-validation-plan)
10. [Open questions for discussion](#10-open-questions-for-discussion)
11. [Appendix: runs, artifacts, glossary](#11-appendix)

---

## 1. Problem setup and intuition

We train language models with RLVR (RL with verifiable rewards) on instruction-following data. Each training prompt is built as **x + c**:

- **x** — an open-ended instruction ("Explain photosynthesis to a child"), *not* verifiable by a program;
- **c** — one to five appended verifiable constraints (IFEval-style: "wrap the answer in double angular brackets", "use the word *gas* as the last word", "at least 300 words", ...), checkable by exact functions.

Data: `allenai/IF_multi_constraints_upto5`, 94,861 train rows (val split 512, seed 1), 54 distinct constraint types, 1–5 constraints per row.

**The core tension:** a reward built only from the constraint checker optimizes *c* and ignores *x*. The model learns to satisfy the checkable part while the actual response quality (relevance, completeness, naturalness with respect to x) degrades — classic reward hacking. We observed exactly this (Section 4).

**Our idea (the anchor method):** the unverifiable part of the prompt can still be supervised *implicitly* through the reference model's probability. Before training, we record how "natural" answers to x look to the frozen reference model, and how much more "surprising" a constraint-obeying answer looks. During training, each rollout's answer is required to stay inside that plausibility band: not suspiciously more predictable than a natural answer (degeneration guard), not far more surprising than a constrained answer needs to be. This costs **one generation per side per prompt, once, before training** — no reward model, no extra inference at training time beyond one scoring pass that the RL setup already performs.

---

## 2. The three reward arms

All arms share the same base: GRPO (group size n=8), batch 1024, lr 1e-6 (constant), KL-loss coef 0.001, entropy coef 0, 4 epochs = 364 steps, and the same constraint reward `IF ∈ [0,1]` from the checker.

| arm | reward | extra cost |
|---|---|---|
| **constraint-only** | `IF` | none |
| **anchor (ours)** | `IF` shaped by the perplexity band (below) | one-time cache build |
| **LLM judge** | `IF + 0.1` if gpt-oss-120b rates the answer ≥ 5/10 against x | judge inference every step |
| **hybrid (anchor + judge)** | disjunction: bonus if *either* the anchor band *or* the judge approves; judge is only called where the anchor didn't already grant the bonus | both |

The hybrid also implements a *restore* path: if the anchor floor zeroed a row's reward and the judge approves the answer, the constraint reward is restored. (This exists in the current code and is exercised in production: e.g., step 91 of the ongoing Tulu-3 hybrid run reviewed 3,717 rows, granted 1,491 bonuses, restored 170 zeroed rows.)

---

## 3. The anchor method in full detail

### 3.1 Cache build (offline, once per reference model)

For every training row, using the frozen reference model (for the failing case: `Qwen/Qwen3-4B`, a hybrid think/no-think model):

1. Sample **y0** = the model's answer to **x alone** (thinking mode on for reasoning runs).
2. Sample **y1** = the model's answer to **x + c**.
3. For reasoning models, strip the `<think>…</think>` block; only the final answer is kept.
4. Score both answers' token-level log-probs **conditioned on x only**, using the *non-thinking* chat template (so the scoring prefix ends in Qwen3's built-in empty scaffold `<think>\n\n</think>\n\n`).
5. Store per row: `y0, y1` (token ids), `ref0_nll, ref1_nll` (summed NLL), `ref0/1_token_count`, `ref0/1_ppl`.

Verified mechanics **[measured]**: `ppl == exp(nll / token_count)` exactly on 100% of rows; token counts exact; rows join to the dataset by shuffled index (join verified two independent ways); scoring prefix identical between cache build and training (verified via 465 single-token y0 rows with NLL ≈ 0, impossible under a thinking prefix).

The failing cache: `if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.json` — 93,882 rows (979 = 1.03% missing vs the train set), metadata v2, sampling `temp 1.0, top_p 1.0, top_k −1`, generation budget 16–32k tokens.

### 3.2 Training-time scoring

Every rollout's final answer (think-stripped, identical extraction) is scored the same way — summed NLL under the frozen reference given x — by the FSDP reference worker (`_score_continuations_with_ref_policy`, `ray_trainer.py:1986`). The cache was scored via vLLM prompt-logprobs; engine difference ≈ 0.001–0.01 nats, negligible **[measured]**. Notation:

```
A = ref0_nll / ref0_count      mean-NLL of y0 given x   ("natural" anchor)
B = ref1_nll / ref1_count      mean-NLL of y1 given x   ("constrained" anchor)
P = rollout answer mean-NLL given x
IF ∈ [0,1] = constraint checker score (base reward)
```

### 3.3 The reward rule (`apply_if_ppl_anchor_reward`, `ray_trainer.py:421`)

Applied **only to rollouts with IF > 0** — the anchor never touches an already-failed rollout; every row it modifies had correct constraints. Mode `both` (used by all main runs):

```
As run in July 2026 (BUGGY):            Current code (fixed 2026-08-09):
if P < A:            reward = 0         if B < A:   keep IF   (invalid band → ignore)
elif A ≤ B, P ≤ B:   reward = IF+0.1    elif P < A: reward = 0
else:                reward = IF        elif P ≤ B: reward = IF+0.1
                                        else:       reward = IF
```

Under GRPO, rewards are group-normalized: zeroing one correct rollout inside a group flips its advantage to ≈ −2.47 — the policy gradient actively pushes *away* from that (correct) behavior.

---

## 4. The observed problem

On **Qwen3-4B in thinking mode**, the anchor arm underperforms constraint-only, unlike our other model families where the anchor arm wins offline:

**Offline (IFBench + LLM-judged quality, 6 judges, 3 repeats; noise floor ±0.063 G-Eval):**

| arm | IFBench acc | G-Eval quality | median answer length |
|---|---|---|---|
| base model | – | 5.02 | 1067 |
| constraint-only | **0.5400** | 3.87 (collapsed) | 302 (collapsed) |
| anchor | 0.4567 | **5.53** | healthy |
| judge | 0.4533 | 5.45 | healthy |

Constraint-only wins accuracy by ~9 points but its quality collapses *below the untrained base* with 3.5× answer shrinkage — the reward hacking the anchor was designed to prevent. The anchor prevents the collapse but pays the accuracy price, and adds nothing over the plain judge.

**Train-time constraint accuracy at matched steps** (wandb, all thinking-mode arms):

| arm | const-acc @ ~362 |
|---|---|
| constraint-only | 0.8144 |
| anchor without the floor (`no_lower_zero` ablation) | 0.7980 |
| hybrid (30B judge) | 0.7904 |
| judge only | 0.7807 |
| **anchor (full rule)** | **0.7680** |

Note: an earlier internal comparison quoted "0.740 vs 0.814"; that mixed step 272 with step 362. The matched-step gap is **+0.046**.

For calibration, the same train-time gap (const-only − anchor) exists in *every* family — llama-8B +0.009, qwen-think +0.057, qwen-nonthink +0.112, tulu +0.125, qwen-1.7B +0.137 **[measured]** — because constraint-only optimizes exactly the train metric, and on non-reasoning models it does so by degenerating (length −68…−77%, KL 0.44–0.70) while the anchor arm demonstrably prevents the degeneration (length flat, KL 5–6× smaller). The anchor's *offline* superiority on those families is the team's motivating result. The qwen-thinking case is the anomaly this document explains: there the anchor paid the accuracy tax **without** buying visible quality protection at train time.

---

## 5. Investigation and findings

Three parallel lanes: (a) line-level code audit incl. git archaeology; (b) forensic audit of all 93,882 cache rows; (c) full per-step wandb histories of ~20 runs across 5 model families. Full reports and data artifacts listed in the Appendix.

### Finding 1 — A branch-order bug destroyed rewards on invalid anchors **[measured]**

In the July code, the floor (`P < A → 0`) was evaluated **before** checking that the band is valid (`A ≤ B`). On *inverted* rows (`B < A`, 12.99% of the cache), A is the corrupted/high side, so most decent answers had `P < A` — their **entire constraint reward was zeroed against a meaningless threshold**.

- Metric fingerprint proving it fired: at matched step 5, on the identical cache, the anchor run logs `invalid_anchor` = 4.4% while the `no_lower_zero` ablation (which structurally skips the buggy branch) logs 10.8%. The missing 6.4pp of inverted rows were being reclassified into `lower_zero` and zeroed.
- Total zeroing measured: ~10.35% of eligible (= constraint-correct) rollouts per step ≈ 740 correct answers zeroed *every step*; the bug accounts for ~62% of that mass, the legitimate floor ~38%.
- Fixed in commit `9c1fee6b` (2026-08-09). **Every anchor/hybrid run before that date carries the bug** (all qwen July runs, the llama/tulu Aug 6–7 runs, all hybrids). Results from those runs are lower bounds on the method.

### Finding 2 — Mean-NLL is length-normalized but not length-invariant **[measured + proof]**

Perplexity `= exp((1/N)·Σ NLL_i)` divides by N, which removes the growth of the *sum* but not the position-dependence of the *terms*: the first tokens of an answer are expensive (many valid answers; probability spread), later tokens are cheap (self-conditioning). The average of a declining profile depends on its length. From our own cache, median perplexity of *healthy* reference answers by answer length:

| answer length (tokens) | 4–8 | 16–32 | 64–128 | 256–512 | 512–1024 | 2048–4096 |
|---|---|---|---|---|---|---|
| median PPL | 11.30 | 3.07 | 3.58 | 1.92 | 1.83 | 1.52 |

A ~6× swing with nothing but length. A crude linear summary: total NLL ≈ C + r·n with C ≈ 165 nats (y0) / 347 nats (y1), r ≈ 0.34 / 0.10 (OLS, n ∈ [5, 8192]); i.e. mean-NLL ≈ C/n + r. Consequences for the rule, which compares P (rollout's length) against A and B (two *other* lengths):

- **Inversions are mostly length artifacts:** 67.6% of inverted rows have y1 longer than y0; inversion rate rises monotonically from 1.2% (y1 ≪ y0) to 47.7% (y1 ≫ y0) across length-ratio octiles. Even among perfectly clean pairs, y1 > 2×y0 in length ⇒ 35.2% inverted.
- **The floor mostly tests length:** a good long rollout mechanically sits below A (wiped); a good short one mechanically sits above B (can never earn the bonus). Spearman(B, y1-length) = **−0.81** vs corr(B, y1 constraint-compliance) = **+0.05**.
- The comparison noise is 20× heteroscedastic (mean-NLL std 8.5 at <30 tokens vs 0.37 at >1000).

### Finding 3 — The floor was economically net-negative on thinking **[measured]**

- The anchor reward term of the thinking run was **negative at every one of 272 logged steps** (mean −0.022, max −0.0027). It operated as a pure tax.
- Band geometry explains why thinking is hit hardest: the thinking band is the narrowest (0.165 nats) and the policy *starts* at normalized position 0.26 — on top of the floor — while every healthy family starts at ≈ 1.0. Thinking `ref0` is dragged up +0.45 nats because y0 is a think-stripped answer: the reasoning that made it predictable was deleted before scoring.
- Behavioral response: answer length 2632 → 1703 (−35%) while policy corpus-PPL rose 2.57 → 2.81 — the policy learned to *dodge the floor*, not to write better answers.
- Direct falsification test of the floor: if the reference's own *compliant* y1 were the rollout, the rule would zero it **11.5%** of the time.
- The `no_lower_zero` ablation closed 56% of the final const-acc gap and flipped the anchor term positive (+0.080) — but drifted below `ref0` with the lowest entropy of any arm (0.291), confirming that *some* floor is needed (the user-observed degeneration risk is real).

### Finding 4 — The upper anchor (per-row) carries almost no per-row signal, but a real pooled signal exists **[measured]**

- Only **29.4%** of y1 satisfy their own constraints; 72.3% of all bonus mass was paid on rows whose reference violates its own constraint. The +0.1 fired on 60.1% of eligible rollouts — a near-constant offset.
- Raw corr(B, compliance) = +0.05 — but this is **Simpson's paradox**: easy constraint types (title: 99.2% compliant) have low surprise; hard ones (`keyword_specific_position`: 1.1%) have high surprise. **Within a constraint type, compliant y1 sit +0.59σ higher in length-adjusted surprise than non-compliant y1** (effect > 0.5σ in 64% of types). The hypothesized signal exists; a single draw (within-type σ = 1.37) cannot measure it, but pooled estimates can (SE 0.05–0.18σ per type).

### Finding 5 — Cache hygiene **[measured]**

Sampling at `temp 1.0 / top_p 1.0 / top_k −1` with no degeneracy protection left: 5.4% of y1 as repetition loops (a 29,842-token `apple apple apple…` is one of our ref1 values; loops have near-zero NLL and cause 27.8% of inversions), 1.7% near-empty y1 (a single Devanagari character `क` grants an unconditional bonus), 1.3% near-empty y0 (a 1-token `"yes"` sets a 40-nats/token floor that zeroes *every* rollout on that prompt forever), 4.7% unhittable zero-width bands, 3.8% poisoned floors (P(zero) = 78%). Union: **41.3% of rows carry ≥ 1 structural defect; only 24.7% are usable and semantically correct.** Ruled out: sampling mismatch vs training (identical params), prefix/template mismatch, think-strip failures (<0.03%), CoT leakage (<0.7%), join misalignment, missing values (0 rows).

Important attribution: **97.1% of expected floor-zeroing mass comes from perfectly clean rows.** The text defects are second-order; the destruction is definitional (Findings 2–3), so cache cleanup alone cannot fix the method: removing all degenerate/empty rows lowers inversions only 12.99% → 9.56%, and the Llama sibling cache built with clean sampling (`p0.95/k20`) still had 18.4% inversions.

### Finding 6 — Component roles invert across model families **[measured]**

A four-mode ablation family (qwen-1.7B) shows `lower_zero_only` costs nothing on non-reasoning models (the policy simply drifts out of the penalty zone: lz% 9.1 → 0.1) while the *interval bonus* carries the entire cost there; on thinking, the floor is permanently binding (lz ≈ 10% throughout) — the asymmetry that makes thinking the worst case for the floor and explains the anomaly.

---

## 6. Root-cause synthesis

The failure chain, in causal order:

1. **[bug]** Inverted anchors fed the floor → ~6.4% of correct rollouts/step zeroed for a fake reason (fixed 2026-08-09).
2. **[metric geometry]** Mean-NLL comparisons across different lengths are length comparisons first; the thinking cache's y0 (think-stripped, unusually "expensive" under no-think scoring) placed the floor directly on top of the policy from step 1.
3. **[estimator variance]** A and B are single draws with σ ≈ 1.4 in corrected units — even without bias, one-draw thresholds false-fire at double-digit rates.
4. **[semantic miss]** B doesn't anchor compliant answers (70.6% non-compliant), so the bonus is noise.
5. **[hygiene]** 41% of rows carry defects that amplify 1–4.

None of this indicts the core idea (reference-model plausibility bands supervising the unverifiable part of the prompt). Each failure is in the *estimator*, not the *estimand*.

---

## 7. Proposed redesign: typicality-standardized anchors

### 7.1 Design constraints (set by the team)

1. Perplexity stays the metric — no substitute statistics (windowed scans etc. rejected).
2. One generation per side per prompt — no k-sample anchors, no best-of-k selection (retries allowed only for *malformed* generations).
3. Keep a lower bound — the degeneration guard is a deliberate design feature (validated by Finding 3's ablation).
4. Per-prompt, per-rollout, instance-wise decisions.

An intermediate proposal (per-length percentile tables pooled over the cache) was reviewed and **rejected** by the team as too estimator-like (binning knobs, borrowed population). It survives in this document only as measured evidence of the bias magnitude (false-wipe 11.5% → 0.7% when length is controlled) and as fallback option B.

### 7.2 The key identity

For every scored position i we already have the reference model's full predictive distribution `p_i` (we read `log p_i(actual token)` from it). For text **sampled from the model itself** — which y0, y1, and (at initialization) every rollout are — the expected NLL at position i *is* the Shannon entropy of that distribution:

```
E[ NLL_i ] = H_i = H(p_i)        Var[ NLL_i ] = V_i   (varentropy of p_i)
```

The decaying profile of H_i (large at answer start, small once committed) **is** the length curve of Finding 2 — `C ≈ Σ (H_i − H_tail)`. So instead of comparing a rollout's mean-NLL against another text's mean-NLL (different Σ H_i ⇒ length bias), subtract the model's own expectation pointwise:

```
z(t) = [ Σ NLL_i  −  Σ H_i ]  /  sqrt( Σ V_i )
```

**Theorem [proof]:** for text sampled from the reference model at the scoring temperature, `E[z] = 0` and `Var[z] = 1` — at every length, every prompt, every position profile. No bins, no fitted curves, no population pooling, no knobs. (If rollout sampling truncates the distribution — top-p/top-k — compute H_i and V_i under the same truncated, renormalized distribution; then the identity is exact again.)

Interpretation: **z is "how many standard deviations more (or less) predictable is this answer than the reference model's own generation process would be, at these exact positions, given this exact prompt."** z ≪ 0 = canned/sharpened; z ≈ 0 = typical; z > 0 = surprising given x — which is where constraint-shaped answers live, i.e., the anchor signal itself. This is the standardized statistic underlying the information-theoretic notion of *typicality* (sequences sampled from a source concentrate around its entropy; deviation measures atypicality).

### 7.3 The rule — unchanged in structure, corrected in units

```
current (fixed) rule:                    proposed rule:
if B < A:    keep IF                     (invalid-band case no longer exists)
elif P < A:  reward = 0                  if z(r) < −τ:           reward = 0
elif P ≤ B:  reward = IF + 0.1           elif z(r) ≤ z(y1) + κ:  reward = IF + 0.1
else:        reward = IF                 else:                    reward = IF
```

- **The upper anchor stays per-prompt and single-draw**: `z(y1)` is this prompt's own constrained answer, now on a scale where its length contribution cancels exactly.
- **The floor becomes a fixed threshold −τ** (recommended τ ≈ 2): a single y0 draw contributes ±1σ of pure noise to a reward-destroying threshold; the fixed floor keeps the *semantics* ("atypically predictable") with exact calibration. Keeping `z(y0)` as the floor instead is possible (maximally faithful to the original design) at a measured noise cost.
- **Calibration [proof + pilot]:** at step 0 the policy *is* the reference, and z is approximately standard normal by CLT over hundreds of tokens ⇒ initial wipe rate ≈ Φ(−τ) (τ=2 ⇒ ~2%). Any later growth in the wipe rate is genuine drift toward canned text — exactly what the floor is for.
- Inversions cannot fire the floor: the floor no longer references any second text.

### 7.4 What z catches, and its one blind spot (stated honestly)

There are two distinct degeneration modes:

1. **Sharpening / canned drift** — the policy picks the predictable continuation at positions where the reference is *uncertain* (`NLL_i < H_i` at choice points, accumulating z ≪ 0). This is the classic RL failure and the drift our no-floor ablation exhibited. **z detects this exactly** — it is the definition of the statistic.
2. **Self-reinforcing repetition loops** — once a loop is established, the reference *itself* collapses (`H_i → 0` alongside `NLL_i → 0`), so the loop is "typical" of the model's conditional process and **z ≈ 0: a pure typicality floor is structurally blind to sampled loops.** (The old absolute floor caught 84% of them — at the cost of the 11.5% false wipes.)

Layered mitigation for mode 2:

- **Source control (primary):** rollout sampling `top_p 0.95 / top_k 20` empirically eliminates loop *generation* (our Llama cache: 12.5–16.4% degenerate → ≈0) **[measured]**.
- **Hybrid runs:** the judge trivially rejects word loops; the existing restore/review path covers the region.
- **Optional absolute backstop (anchor-only runs):** wipe if raw P is below the 1st percentile of the reference's own natural answers at that length — 0.24% false-trip, catches the most extreme loops **[measured]**. Team decision whether to include (it reintroduces one population table).

### 7.5 Fallback option B (if the team prefers zero new math)

Run only the already-merged fix (Rule 2) + hygiene (Stage A below) + matched sampling. This removes the bug and most degenerate anchors but leaves Findings 2–4 in place (length-tilted comparisons, one-draw noise, uninformative bonus). Expected to recover part of the gap (the no-floor ablation bounds it), not all.

---

## 8. End-to-end pipeline specification

### Stage A — cache build (GPU; one-time)

| item | value | why |
|---|---|---|
| sampling | `temp 1.0, top_p 0.95, top_k 20, presence_penalty 0.0` | kills loop generation [measured]; penalty matched to training |
| generation budget | = training budget (8192 incl. `<think>`) | anchors must be producible under the policy's budget (3.2% of current y1 are not) |
| retries | malformed only: empty/<10-token answer, missing `</think>`, repetition loop, `y0==y1` | measurement failures, not selection; single kept draw per side |
| rejection mechanism | build → offline validator deletes bad rows from JSON → rerun (the build loop refills only missing rows) | reject-and-resample with zero new trainer code |
| **new fields** | per-row `Σ H_i`, `Σ V_i` for y1 (and y0), recorded during the scoring pass | needed for z; two floats/row |
| metadata v4 | + `presence_penalty`, `top_k`, per-row retry counts, prompt hash, `ppl_prefix_mode`, `nll_scope`; loader hard-fails on missing keys | provenance; the strict-metadata check has already caught one real config bug |

The **existing cache** can be upgraded (add ΣH, ΣV) with one rescoring pass — a few GPU-hours, no regeneration — so the pilot does not wait for a rebuild.

### Stage B — trainer change (~60–100 lines, two functions)

- In the ref-scoring pass (`_score_continuations_with_ref_policy`): alongside the existing per-token log-probs, compute per-position entropy H_i and varentropy V_i from the same logits (verl already has the fused entropy kernel used for `actor/entropy`; varentropy is one more reduction). **No new forward passes.** If rollout sampling is truncated, compute H/V under the truncated renormalized distribution.
- In `apply_if_ppl_anchor_reward` (+ its mirror `get_if_ppl_anchor_interval_bonus_mask` so hybrid judge-gating stays consistent): replace the `(P, A, B)` comparisons with `(z(r), −τ, z(y1)+κ)`. Per-row `A, B` remain logged as diagnostics.
- Floor action: **hybrid** — provisional zero + judge review + restore-on-pass (existing code path); **anchor-only** — hard zero (default) or bounded penalty (open question 1).
- New logging: z histogram per step (mean/p10/p50/p90), wipe rate, bonus rate, restore rate, and a step-0 assertion `|wipe_rate − Φ(−τ)|` small.

### Stage C — training configuration

- Rollout sampling **explicitly** `temp 1.0, top_p 0.95, top_k 20, presence_penalty 0.0` (verl defaults are `p=1.0, k=−1`; the July runs actually trained at p=1.0 — nothing set it).
- Budgets matched (8192); `IF_REQUIRE_THINK_END_FOR_REWARD=true` unchanged; eligibility gate unchanged.
- No in-training validation (team decision: an iid 512-row val split can't detect the failures that matter). Replacement: **offline epoch-boundary evals** on the checkpoints every run already saves (`SAVE_FREQ=91`), compared at matched steps only.
- `IF_REF_ANCHOR_CACHE_METADATA_STRICT=true`; launch from a committed SHA; alarms: entropy < 0.30, anchor-reward mean ≤ 0 persisting, wipe-rate drift.

---

## 9. Validation plan

**Offline, before any training (GPU: a few hours after the current run frees the node):**

1. Rescore a stratified ~5k-row cache sample computing z. Pre-registered checks: (i) mean z ≈ 0 in *every* length bin for clean text — the falsifiable test of the theorem's applicability; (ii) z distribution of compliant y1 (drives κ); (iii) z of loops ≈ 0 (quantifies the blind spot so the backstop is sized from data).

**Pilot (one run, qwen3-4b thinking, existing upgraded cache):**

Anchor-only arm first (cleanest attribution against history), all Stage-C configs, τ = 2, κ = 0.5 initial. Compared at matched steps against `790faw9y` (const-only), `zrqp58e5` (no-floor), `mgijdvdj` (old anchor). Success criteria, pre-registered:

1. step-0 wipe rate ≈ Φ(−τ) (calibration);
2. anchor-reward mean > 0 from early steps (old run: negative at every step);
3. train const-acc within ~0.02 of const-only at matched steps;
4. entropy stays above the no-floor run's 0.291;
5. epoch-2 offline eval: quality (G-Eval) ≥ anchor-old, without the ~9-point IFBench deficit vs const-only.

If it passes → hybrid variant (judge screener + restore), then the Stage-A rebuilt cache when GPUs allow.

---

## 10. Open questions for discussion

1. **Below-floor action (anchor-only runs):** hard zero (design-faithful) vs bounded penalty (−0.1)? Hard zero flips a correct answer's GRPO advantage to ≈ −2.47; bounded keeps the direction without making it the group's worst. Proposed: hard zero for the pilot, ablate after.
2. **Floor form:** fixed −τ (recommended: calibrated, noise-free) vs per-prompt z(y0) (maximally faithful, +1σ threshold noise)?
3. **Loop backstop for anchor-only runs:** none (rely on sampling) vs 1st-percentile absolute backstop (reintroduces one population table at 0.24% false-trip)?
4. **κ and the edge:** is per-prompt z(y1) enough, or blend toward the constraint-type mean (the +0.59σ within-type compliance signal suggests type-level structure is real)? Blending reintroduces pooling — team's call on purity vs precision.
5. **τ selection:** 2.0 (≈2% initial wipe) vs 2.5/3.0 (more permissive)? The step-0 measurement makes this a one-batch decision.
6. **Cache rebuild timing:** before or after the pilot (pilot does not require it)?

---

## 11. Appendix

### 11.1 Key runs (wandb `ifif/verl_if_rlvr`)

| run(s) | arm | notes |
|---|---|---|
| `roixvo8n` → `790faw9y` | qwen3-4b think, constraint-only | final 0.8144 |
| `q25wnmtl` → `mgijdvdj` | qwen3-4b think, anchor pyx0.1 | **buggy code**; final 0.7680 |
| `rgn6fe88` → `hhf14ka6` | qwen3-4b think, judge gpt-oss-120b | final 0.7807 |
| `zrqp58e5` | qwen3-4b think, anchor **no_lower_zero** | escaped the bug structurally; 0.7980 |
| `21gsuhhe`, `4243en2i`→`ssickv7e`→`opqxrrbp`, `bdhi3n4p` | qwen3-4b think hybrids | all pre-fix |
| `kmqwwm9j`, `ddnpqut5` et al., `gama2wku` | qwen3-4b non-think arms | contrast family |
| `89gipi20`/`6fbo8yg9`, `myf68cle`/`1gyg97hj` | llama-8B / tulu-8B anchor & const | offline-winning families; pre-fix |
| (live) `5ziiw4bk` | tulu-3-8B hybrid anchor+judge | post-fix code; in flight |

Caveats found while stitching: the llama "anchor" run's scored policy-PPL starts at 46.6 (its cache deserves its own audit before being cited); the non-think anchor "chain" is two different cache experiments sharing one name; one hybrid has visible restart damage at steps 161/321.

### 11.2 Artifacts (on the training box)

- `investigation/cache_audit_report.md` — 570-line forensic audit, all tables and worst examples.
- `investigation/wandb_dynamics_report.md` — 678-line cross-run dynamics report; 33 full scan-history CSVs + 20 stitched per-arm CSVs.
- `investigation/cacheaudit_rows_final.parquet` — all 93,882 rows with decoded texts + derived metrics (the source of every cache number in this doc).
- Fix commit: `9c1fee6b` (2026-08-09); buggy era: `7f37be8e` (Jun 29) onward.
- Reward rule: `verl/trainer/ppo/ray_trainer.py:421` (`apply_if_ppl_anchor_reward`); scoring: `:1986`; prefix builder: `verl/experimental/agent_loop/single_turn_agent_loop.py:118`.

### 11.3 Glossary

- **x / c** — unverifiable instruction / verifiable constraints appended to it.
- **y0 / y1** — reference model's one-shot answers to x / x+c (final answer only, think block stripped).
- **ref0 (A) / ref1 (B)** — mean-NLL of y0 / y1 under the reference model conditioned on x only.
- **mean-NLL** — `(1/N)·Σ −log p(token_i)`; perplexity = `exp(mean-NLL)`. Same quantity, log scale.
- **inversion** — a row with B < A (band undefined). 12.99% of the cache.
- **lower_zero / floor** — rule branch that zeroes the total reward when the rollout is more predictable than the lower anchor.
- **eligible row** — a rollout with IF > 0; the anchor only ever modifies these.
- **H_i / V_i** — entropy / varentropy of the reference's predictive distribution at position i.
- **z(t)** — `(ΣNLL − ΣH)/√(ΣV)`: standardized (a)typicality of text t under the reference. 0-mean, unit-variance for reference-sampled text at any length **[proof]**.
- **GRPO** — group-relative PPO; rewards normalized within each prompt's 8 rollouts (advantages ≈ ±2.47 at the extremes).
