# Qwen3-4B representative 20,480-row subset — verification

Population: 93,882 trainable rows. Subset: 20,480 rows (21.81%). Nested fixed panel A: 4,096 rows.
Seed: 20260819. Composite: random median 0.496746 → best draw 0.418158 → polished 0.080561 (1911 swaps).

## Pre-registered acceptance checks

- PASS — max_continuous_KS: 0.004107 < 0.006000
- PASS — TV_constraint_ID: 0.000203 < 0.001500
- PASS — TV_constraint_family: 0.000090 < 0.001000
- PASS — TV_top100_ID_pairs: 0.002275 < 0.005000
- PASS — TV_source_domain: 0.001211 < 0.005000
- PASS — TV_top200_signature: 0.008118 < 0.010000
- PASS — TV_availability_pattern: 0.003320 < 0.005000

## Continuous distribution checks

| variable | KS D |
|---|---:|
| log_anchor_y0_tokens | 0.004107 |
| anchor_m0_std | 0.003443 |
| ci95_lower | 0.002933 |
| prompt_tokens | 0.002803 |
| x_prompt_tokens | 0.002697 |
| log_constraint_chars | 0.002628 |
| anchor_m1_std | 0.002607 |
| ci95_upper | 0.002426 |
| anchor_center_width | 0.002349 |
| anchor_m1_mean | 0.002212 |
| log_anchor_length_ratio | 0.002209 |
| constraint_tokens | 0.002187 |
| log_x_words | 0.002134 |
| log_anchor_y1_tokens | 0.002031 |
| difficulty | 0.001995 |
| anchor_m0_mean | 0.001984 |
| x_ascii_ratio | 0.001530 |
| log_x_chars | 0.001390 |
| observed_ifscore | 0.001351 |
| x_cyrillic_ratio | 0.001102 |
| x_devanagari_ratio | 0.000543 |
| x_cjk_ratio | 0.000264 |
| x_hangul_ratio | 0.000192 |
| x_arabic_ratio | 0.000163 |

## Anchor availability and geometry

| statistic | full | subset |
|---|---:|---:|
| run1 available | 0.868984 | 0.869922 |
| run1 flip | available | 0.127491 | 0.126572 |
| run1 IFEval score | available | 0.583547 | 0.584658 |
| run1 all-satisfied | available | 0.303915 | 0.304838 |
| run2 available | 0.951237 | 0.948291 |
| run2 flip | available | 0.124642 | 0.124195 |
| run2 IFEval score | available | 0.576885 | 0.577714 |
| run2 all-satisfied | available | 0.297512 | 0.297873 |
| run3 available | 0.951002 | 0.947949 |
| run3 flip | available | 0.127058 | 0.127331 |
| run3 IFEval score | available | 0.576547 | 0.575962 |
| run3 all-satisfied | available | 0.297776 | 0.295199 |
| N=3 center flipped | 0.098400 | 0.098533 |
| any per-run flip among N=3 | 0.228998 | 0.228304 |

### Availability pattern

| bit pattern (r3r2r1) | full n (%) | subset n (%) |
|---|---:|---:|
| 000 | 3371 (3.591%) | 731 (3.569%) |
| 001 | 460 (0.490%) | 168 (0.820%) |
| 010 | 303 (0.323%) | 65 (0.317%) |
| 011 | 466 (0.496%) | 102 (0.498%) |
| 100 | 284 (0.303%) | 60 (0.293%) |
| 101 | 463 (0.493%) | 100 (0.488%) |
| 110 | 8342 (8.886%) | 1808 (8.828%) |
| 111 | 80193 (85.419%) | 17446 (85.186%) |

### Cross-run flip count among rows with all three draws

| number of flipped draws | full n (%) | subset n (%) |
|---:|---:|---:|
| 0 | 61829 (77.100%) | 13463 (77.170%) |
| 1 | 10017 (12.491%) | 2173 (12.456%) |
| 2 | 4711 (5.875%) | 1020 (5.847%) |
| 3 | 3636 (4.534%) | 790 (4.528%) |

## Constraint load and pooled difficulty

Difficulty quintile 0 is the hardest and 4 is the easiest under the constraint-ID-pooled mean IFEval score across available draws.

| category | level | full n (%) | subset n (%) |
|---|---:|---:|---:|
| constraints per input | 1 | 22656 (24.132%) | 4963 (24.233%) |
| constraints per input | 2 | 23550 (25.085%) | 5116 (24.980%) |
| constraints per input | 3 | 22953 (24.449%) | 5008 (24.453%) |
| constraints per input | 4 | 17739 (18.895%) | 3878 (18.936%) |
| constraints per input | 5 | 6984 (7.439%) | 1515 (7.397%) |
| pooled difficulty quintile | 0 | 18828 (20.055%) | 4117 (20.103%) |
| pooled difficulty quintile | 1 | 18880 (20.110%) | 4112 (20.078%) |
| pooled difficulty quintile | 2 | 18621 (19.834%) | 4077 (19.907%) |
| pooled difficulty quintile | 3 | 18779 (20.003%) | 4082 (19.932%) |
| pooled difficulty quintile | 4 | 18774 (19.997%) | 4092 (19.980%) |

## Interpretable distribution quantiles

Values are q05 / q25 / q50 / q75 / q95. Anchor summaries condition on the stated availability so missing values are not converted into numeric sentinels.

| variable | scope | full quantiles | subset quantiles |
|---|---|---:|---:|
| prompt tokens | all | 53.0000 / 97.0000 / 156.0000 / 290.0000 / 520.0000 | 53.0000 / 97.0000 / 156.0000 / 290.0000 / 518.0000 |
| constraint tokens | all | 13.0000 / 29.0000 / 46.0000 / 67.0000 / 97.0000 | 13.0000 / 28.0000 / 46.0000 / 67.0000 / 97.0000 |
| pooled IFEval score | at least one draw | 0.0000 / 0.3333 / 0.5833 / 0.8667 / 1.0000 | 0.0000 / 0.3333 / 0.5833 / 0.8667 / 1.0000 |
| mean x-only NLL/token | at least one draw | 0.2419 / 0.4152 / 0.5846 / 0.7717 / 1.6902 | 0.2416 / 0.4152 / 0.5846 / 0.7707 / 1.6806 |
| mean x+c NLL/token | at least one draw | 0.3928 / 0.8959 / 1.4697 / 2.4955 / 6.4540 | 0.3927 / 0.8978 / 1.4727 / 2.4912 / 6.5258 |
| center width (m1-m0) | at least one draw | -0.2211 / 0.2973 / 0.8119 / 1.7570 / 5.5138 | -0.2199 / 0.2993 / 0.8173 / 1.7633 / 5.6218 |
| mean x-only output tokens | at least one draw | 36.0000 / 262.0000 / 552.0000 / 862.6667 / 1409.4167 | 36.6667 / 263.0000 / 554.0000 / 868.0000 / 1418.2000 |
| mean x+c output tokens | at least one draw | 17.6667 / 99.6667 / 236.6667 / 520.3333 / 2565.0000 | 17.0000 / 99.6667 / 236.3333 / 522.0000 / 2520.2667 |
| x-only cross-draw std | at least two draws | 0.0108 / 0.0389 / 0.0718 / 0.1350 / 0.4677 | 0.0108 / 0.0389 / 0.0726 / 0.1360 / 0.4636 |
| x+c cross-draw std | at least two draws | 0.0245 / 0.1159 / 0.2789 / 0.6544 / 2.3188 | 0.0257 / 0.1172 / 0.2791 / 0.6534 / 2.3290 |
| CI95 lower endpoint | all three draws | -0.2126 / 0.2025 / 0.3676 / 0.5383 / 0.9354 | -0.2079 / 0.2032 / 0.3670 / 0.5401 / 0.9403 |
| CI95 upper endpoint | all three draws | 0.6677 / 1.4309 / 2.4455 / 4.3600 / 12.4037 | 0.6711 / 1.4343 / 2.4505 / 4.3794 / 12.4841 |

## Largest source/domain groups

| source/domain | full n (%) | subset n (%) |
|---|---:|---:|
| personahub | 24858 (26.478%) | 5423 (26.479%) |
| ai2/evol_codealpaca_heval_decontaminated | 11044 (11.764%) | 2410 (11.768%) |
| ai2/tulu_v3.9_aya_100k | 10376 (11.052%) | 2264 (11.055%) |
| ai2/flan_v2_converted | 9577 (10.201%) | 2089 (10.200%) |
| ai2/numinamath_tir_math_decontaminated | 6876 (7.324%) | 1500 (7.324%) |
| ai2/tulu_v3.9_wildchat_100k | 5804 (6.182%) | 1266 (6.182%) |
| ai2/tulu_v3.9_wildjailbreak_decontaminated_50k | 5388 (5.739%) | 1176 (5.742%) |
| personas_math_easy | 5336 (5.684%) | 1164 (5.684%) |
| ai2/tulu_v3.9_open_math_2_gsm8k_50k | 5301 (5.646%) | 1156 (5.645%) |
| ai2/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k | 5294 (5.639%) | 1155 (5.640%) |
| opaque_short_id | 1174 (1.251%) | 256 (1.250%) |
| ai2/no_robots_converted | 985 (1.049%) | 215 (1.050%) |
| ai2/tulu_v3.9_table_gpt_5k | 513 (0.546%) | 119 (0.581%) |
| oasst1 | 466 (0.496%) | 102 (0.498%) |
| science.craftchem_ner | 40 (0.043%) | 10 (0.049%) |
| science.drug_combo_extraction_re | 40 (0.043%) | 6 (0.029%) |
| science.chemprot_ner | 37 (0.039%) | 9 (0.044%) |
| science.chemdner_ner | 37 (0.039%) | 8 (0.039%) |
| science.chia_ner | 34 (0.036%) | 5 (0.024%) |
| science.covidfact_entailment | 34 (0.036%) | 8 (0.039%) |

## Constraint composition

| instruction ID | full % | subset % |
|---|---:|---:|
| last_word:last_word_answer | 2.1424 | 2.1423 |
| detectable_format:bigram_wrapping | 2.1256 | 2.1255 |
| last_word:last_word_sent | 2.1248 | 2.1236 |
| punctuation:no_comma | 2.1076 | 2.1067 |
| keywords:start_end | 2.1015 | 2.1030 |
| keywords:word_once | 2.0921 | 2.0917 |
| detectable_format:sentence_hyphens | 2.0913 | 2.0917 |
| change_case:english_capital | 2.0860 | 2.0861 |
| length_constraints:number_sentences | 2.0856 | 2.0861 |
| keywords:exclude_word_harder | 2.0831 | 2.0842 |
| detectable_format:title | 2.0806 | 2.0804 |
| keywords:keyword_specific_position | 2.0806 | 2.0804 |
| keywords:no_adjacent_consecutive | 2.0778 | 2.0786 |
| letters:letter_counting2 | 2.0745 | 2.0748 |
| detectable_format:square_brackets | 2.0729 | 2.0729 |
| count:count_unique | 2.0688 | 2.0673 |
| punctuation:punctuation_exclamation | 2.0680 | 2.0673 |
| keywords:existence | 2.0631 | 2.0636 |
| length_constraints:nth_paragraph_first_word | 2.0618 | 2.0617 |
| keywords:forbidden_words | 2.0618 | 2.0598 |
| detectable_content:postscript | 2.0590 | 2.0561 |
| keywords:palindrome | 2.0582 | 2.0579 |
| keywords:letter_frequency | 2.0582 | 2.0598 |
| punctuation:punctuation_dot | 2.0557 | 2.0561 |
| first_word:first_word_answer | 2.0557 | 2.0561 |
| paragraphs:paragraphs2 | 2.0532 | 2.0523 |
| copy:repeat_phrase | 2.0528 | 2.0561 |
| detectable_content:number_placeholders | 2.0524 | 2.0523 |
| letters:letter_counting | 2.0520 | 2.0523 |
| detectable_format:number_highlighted_sections | 2.0508 | 2.0504 |
| detectable_format:number_bullet_lists | 2.0500 | 2.0485 |
| paragraphs:paragraphs | 2.0483 | 2.0485 |
| keywords:frequency | 2.0467 | 2.0467 |
| count:count_increment_word | 2.0459 | 2.0485 |
| startend:end_checker | 2.0406 | 2.0410 |
| keywords:word_count_different_numbers | 2.0373 | 2.0373 |
| change_case:capital_word_frequency | 2.0340 | 2.0335 |
| length_constraints:number_words | 2.0279 | 2.0279 |
| count:lowercase_counting | 2.0234 | 2.0242 |
| first_word:first_word_sent | 2.0177 | 2.0185 |
| startend:quotation | 2.0013 | 1.9998 |
| change_case:english_lowercase | 1.9809 | 1.9810 |
| length_constraints:number_paragraphs | 1.9759 | 1.9754 |
| detectable_format:multiple_sections | 1.9694 | 1.9698 |
| count:counting_composition | 1.8091 | 1.8084 |
| language:response_language | 1.7714 | 1.7709 |
| combination:two_responses | 0.7869 | 0.7860 |
| combination:repeat_prompt | 0.7632 | 0.7635 |
| copy:copying_multiple | 0.7276 | 0.7279 |
| new:copy_span_idx | 0.7260 | 0.7279 |
| copy:copy | 0.7137 | 0.7129 |
| detectable_format:json_format | 0.7064 | 0.7072 |
| detectable_format:constrained_response | 0.7031 | 0.7035 |
| copy:copying_simple | 0.6953 | 0.6941 |

## Canonical anchor defects

| class | full n (%) | subset n (%) |
|---|---:|---:|
| clean | 82019 (87.364%) | 17919 (87.495%) |
| cot_marker | 25 (0.027%) | 6 (0.029%) |
| duplicate | 121 (0.129%) | 26 (0.127%) |
| loop_y0 | 579 (0.617%) | 124 (0.605%) |
| no_anchor | 3371 (3.591%) | 731 (3.569%) |
| short_y0 | 2371 (2.526%) | 512 (2.500%) |
| short_y1 | 2956 (3.149%) | 636 (3.105%) |
| suspect_loop_y1 | 2440 (2.599%) | 526 (2.568%) |

## Panel-level representativeness

| panel | rows | max KS | TV ID | availability-pattern TV |
|---|---:|---:|---:|---:|
| A | 4096 | 0.092600 | 0.000690 | 0.131016 |
| B | 4096 | 0.031858 | 0.021843 | 0.031338 |
| C | 4096 | 0.029431 | 0.024631 | 0.033291 |
| D | 4096 | 0.030299 | 0.022222 | 0.031094 |
| E | 4096 | 0.027260 | 0.021841 | 0.030606 |

## Correlation replay

| pair | full Spearman | subset Spearman |
|---|---:|---:|
| anchor_m1_mean ~ log_anchor_y1_tokens | -0.62174 | -0.62401 |
| anchor_m0_mean ~ log_anchor_y0_tokens | -0.32689 | -0.32701 |
| anchor_center_width ~ log_anchor_length_ratio | -0.44818 | -0.44749 |
| difficulty ~ observed_ifscore | +0.65196 | +0.65376 |

## Generation work remaining

- Missing complete run1 anchors: 2,664
- Missing complete run2 anchors: 1,059
- Missing complete run3 anchors: 1,066
- Rows with no currently complete draw: 731

## Overall: ALL CHECKS PASS
