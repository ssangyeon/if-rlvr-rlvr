# Subset4k Ablation Program — Design Notes, Decisions, and Idea Tracker

*Living document. Full background: `if_rlvr/docs/if_rlvr_anchor_reward_review_2026-08-11.md`
(method + failure forensics), `if_rlvr/docs/anchor_subset_curation_design_2026-08-13.md`
(the 4,096-row subset), `if_rlvr/docs/anchor_cache_v2_partial_review_2026-08-13.md` and
`if_rlvr/data_subsets/qwen3_4b_reasoning_anchor4k/v2_subset_final_verification.md` (v2 data).*

## 1. Purpose and standing design constraints

Goal: identify the root causes of the qwen3-4b (thinking) anchor arm's underperformance
vs constraint-only, and search reward/anchor configurations that keep the anchor's
quality protection without the accuracy tax — fast, on a 4,096-row subset that is
verified-representative of the full training set.

Constraints set by the team (standing; do not re-litigate silently):

1. **Perplexity/mean-NLL stays the metric.** No substitute statistics (windowed scans
   etc. rejected).
2. **One generation per side per prompt** is a core claim of the method. Retries are
   allowed for *malformed* draws only. (k=3 draws are authorized specifically for the
   floor-robustness experiment, arm B2.)
3. **A degeneration guard must exist in some form.** The no-floor ablation (`zrqp58e5`)
   drifted to the lowest entropy of any arm (0.291) — some floor is needed.
4. **Per-prompt, per-rollout, instance-wise decisions.** Population/percentile tables
   were reviewed and rejected as too estimator-like.
5. **Anchor generation budget matches the training budget** (8,192 incl. think). The
   sole sanctioned exception: the 476 think-budget-infeasible subset rows were filled
   at r32768 (documented per-row in `v2_subset_provenance.json`).
6. **One component per arm**; compositions only after individual gains are shown;
   deltas read against the strict baseline (A3).
7. **No in-training validation** (uninformative 512-row iid split); offline evals on
   epoch checkpoints instead.

## 2. Evidence base (measured; drives the priorities)

| finding | number |
|---|---|
| July branch-order bug: flips fed the hard-zero floor | ~62% of ~10.35%/step zeroing of correct rollouts (fixed `9c1fee6b`) |
| mean-NLL is a length proxy | corr(m1, y1-length) = −0.81; corr(m1, compliance) = +0.05 |
| thinking-band geometry | width 0.165 nats; policy starts at position 0.26 (healthy families ≈ 1.0) |
| anchor reward term (thinking run) | negative at all 272 logged steps |
| floor falsification | rule zeroes the ref's own compliant y1 11.5% of the time |
| bonus selectivity | fires on 60.1% of eligible; 72.3% of mass where the ref violates its own constraint |
| within-type compliance signal (Simpson's) | +0.59σ length-adjusted, invisible to single draws (σ=1.37) |
| v2 anchors vs v1 | inversions unchanged (12.8% vs 12.5%); true-degen loops ~2.8% vs ~2.9%; geometry identical |
| flip identity across independent draws | P(flip₂ \| flip₁) = 49.8% vs 7.5% base → ~half intrinsic, half draw-luck |
| margin calibration (2026-08-14, this subset) | pseudo-wipe 11.82% at c=0 → 1.95% at **c=0.7 nats** (≈ 4× the historical band width) |

## 3. Stage 1 (priority-ordered; one component per arm; runs elsewhere)

| # | arm | change vs A3 | prediction (pre-registered) |
|---|---|---|---|
| 1 | a0 constraint-only | no anchors | acc ceiling; quality collapse signature |
| 2 | a3 strict baseline | — (July semantics) | floor binds from step 1; anchor term ≤ 0 |
| 3 | a2 soft floor | floor action → ignore | closes most of the acc gap; watch late drift |
| 4 | a4 penalty −0.1 | floor action → −0.1 | between a3 and a2; guard direction kept |
| 5 | a5 no floor | floor removed | most gap closed; entropy-collapse risk (0.291 precedent) |
| 6 | a1 flip-abstain | flips keep IF | recovers the majority of zeroing mass (bug attribution ~62%) |

Decision rule at matched epochs: const-acc gap to a0 ≤ 0.02 AND G-Eval ≥ a1 (noise
±0.063) AND entropy ≥ 0.35 AND no length collapse (<60% of a0). Scorecard per arm:
const-acc, IFBench, G-Eval, length, entropy, KL, anchor-term, branch rates.

## 4. Stage 2 (prepared now; run after Stage 1 picks the floor action)

**Ready (scripts in this folder):**

- **b1 — calibrated margin floor (k=1)**: fixed rule, zero only when P < A − c;
  c = 0.7 from `tools/calibrate_floor_margin.py` (pre-registered 2% pseudo-wipe
  target). The k=1 rival to b2: if b1 ≈ b2, the paper keeps single-generation.
- **b2 — k=3 floor** (the anchor-robustness experiment; needs the team's extra y0
  draws): `tools/make_k3_floor_cache.py` builds a derived cache (trainer untouched)
  and reports, per form, the per-row anchor std (first direct measurement), the
  loosening distribution, and the tighter-than-single-draw failure rate. Default
  form **min-of-3** (distribution-free; P(new draw < min) = 1/4); mean−z·std forms
  are computed for comparison but carry the df=2 noise caveat — choose from the
  report, not by taste.
- **b3 — flip-abstain + penalty**: the one non-degenerate Stage-1 composition;
  contingent on both components showing individual gains.
- **`tools/augment_subset_hv.py`**: ΣH/ΣV per anchor (full + sampling-truncated
  distributions), self-validating against cached NLLs — the z-rule's data
  prerequisite, runnable on any GPU box.

**Designed, pending (in order):**

- **z — typicality rule** (z(r) < −τ → floor action; z(r) ≤ z(y1)+κ → bonus; τ=2,
  κ=0.5): length-invariant by construction; floor position and flips dissolve rather
  than get patched. Needs (a) the H/V sidecar (tool ready), (b) a trainer change to
  emit per-rollout ΣH/ΣV from the ref-scoring pass (entropy exists in verl's kernels;
  varentropy is one more reduction) and the z-comparison in the reward fn, (c) a
  smoke run — do not ship this blind. Pre-registered: step-0 wipe ≈ Φ(−τ) ≈ 2%.
  Known blind spot: self-reinforcing loops look typical (H→0 with NLL→0); mitigations
  layered (sampling truncation, judge in hybrids, optional absolute backstop —
  measured 20.5% catch at 0.24% false-trip, team call).
- **hybrid — Stage-1/2 winner + gpt-oss-120b judge** (restore path included): the
  production configuration; adapt the existing qwen hybrid launcher to the subset
  frame once the winner's knobs are known.
- **graduation**: winner → 8-epoch subset extension (memorization watch) → one seed
  replicate of the decisive comparison → full 94k run. Full-scale requires either
  filling the remaining ~11.6k v2 rows (same r32768 fallback for the infeasible
  tail) or accepting their absence — do not mix silently.

## 5. Rejected ideas (with reasons — keep to avoid re-litigating)

| idea | verdict | why |
|---|---|---|
| per-length percentile floor tables | rejected | population/binning knobs; "too heuristic" (team) |
| windowed/sliding-NLL statistics | rejected | PPL must remain the metric (team) |
| best-of-k / averaged y1 upper anchor | rejected | violates the single-generation core claim; cost trap on low-compliance types |
| k=3 mean−z·std floor as default | cautioned | df=2 std noise; floors go *tighter* than single-draw when draws agree; min-of-3 preferred pending b2's report |
| in-training validation (TEST_FREQ) | rejected | iid 512-row split can't detect the failures that matter |
| raising the anchor budget globally | rejected | breaks the anchor/policy regime match (training runs at 8,192) — the 476-row r32768 fill is a documented, bounded exception |
| dropping flipped rows from training | deferred | flips are ~half draw-luck; reward-level abstention (A1) is the right first test; data surgery only if A1 leaves flips suspicious; systematically deletes long-form prompts |
| hygiene detector deleting all loop-flagged y1 | fixed instead | 32.5% of flagged rows are legitimate repetition (copy/repeat constraints); detector is now compliance-aware |

## 6. Idea backlog (unscheduled; promote deliberately)

- **Compliance-gated bonus**: +0.1 only where the ref's own y1 passes its constraints
  (29.4% of rows; flags exist). Attacks the 72.3% misdirected bonus mass.
- **Type-pooled bonus edge / κ blend**: use the +0.59σ within-type compliance signal
  to set per-type edges — reintroduces pooling; purity-vs-precision call.
- **Bonus magnitude sweep** (0.05/0.1/0.2) — only if the winner's bonus looks load-bearing.
- **Loop backstop for anchor-only z runs** (q1% absolute floor) — 20.5% catch @ 0.24%
  false-trip; decide with z pilot data.
- **Presence-penalty ablation** for rollouts (anchors used 0.0; training default 0.0 —
  confirm no drift if defaults change).
- **8-epoch memorization probe** on the subset (what a 4k set can't otherwise say).
- **Seed replicate** of the decisive pair to size run-to-run noise before any close call.
- **2×4-GPU concurrent arms** to halve calendar time (validate memory first).
- **Offline eval battery runner** (IFBench + single-judge G-Eval per epoch checkpoint)
  — still to build; required before any winner is declared.
- **Full-v2 completion**: extend the r32768 fill flow to the remaining ~11.6k rows for
  the graduation run.

## 7. Decision log

- **2026-08-11** — root-cause forensics accepted (bug + geometry + estimator variance +
  bonus miss + hygiene); redesign constraints 1–4 fixed by the team.
- **2026-08-12** — subset methodology approved; n=4096 (exact strata, rerandomize+polish).
- **2026-08-13** — two-stage ablation program locked; Stage-1 = one-component arms in
  the team's priority order (const-only → strict → soft → penalty → no-floor →
  flip-abstain); floor-only and compliance-gated-bonus arms cut from Stage 1.
- **2026-08-14** — v2 subset finalized (3,620 r8192 + 476 r32768); reward knobs +
  launchers pushed; runs to execute on another machine (auto-download bootstrap);
  memory profile fixed after observed OOM (65536 token cap, util 0.9, engine sleep);
  margin calibrated c=0.7; Stage-2 b1/b2/b3 + tools prepared; z-rule and hybrid
  deliberately gated on Stage-1 results + a smoke test.
