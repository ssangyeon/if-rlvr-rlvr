# Subset verification report (auto-generated)

n = 4096 of 93882 frame rows; seed 20260813; composite 0.0635 (random-draw median 0.5170).
Cache file: `if_ref_anchor_teacher4b_reasoning_train_seed1_scored_by_qwen3_4b.SUBSET4096.json`; halves A/B = 2048/2048.

## Acceptance checks
- PASS — KS (worst of 11 variables): 0.0053 < 0.01 (detectability threshold 0.0212)
- PASS — TV 54-instruction-ID marginal: 0.0007 < 0.005
- PASS — worst-ID coverage: 0.99x expected
- PASS — inverted %: full 0.1267 vs subset 0.1270 (tol 0.0102)
- PASS — y1_allsat %: full 0.2936 vs subset 0.2925 (tol 0.0139)
- PASS — p_zero mean: full 0.1870 vs subset 0.1870 (tol 0.0119)
- PASS — p_bonus mean: full 0.3293 vs subset 0.3293 (tol 0.0144)
- PASS — false-wipe compliant y1: full 0.1151 vs subset 0.1135 (tol 0.0098)

## Per-variable KS D

| variable | KS D |
|---|---|
| log_y0 | 0.0053 |
| diff_score | 0.0051 |
| x_len | 0.0051 |
| p_bonus | 0.0051 |
| log_ratio | 0.0046 |
| m1 | 0.0044 |
| log_y1 | 0.0043 |
| width | 0.0042 |
| p_zero | 0.0031 |
| m0 | 0.0031 |
| y1_ifscore | 0.0018 |

TV family = 0.0003; TV top-50 pairs = 0.0066

## Correlation structure (Spearman)

| pair | full | subset |
|---|---|---|
| m1 ~ log_y1 | -0.806 | -0.805 |
| m0 ~ log_y0 | -0.378 | -0.366 |
| m1 ~ y1_ifscore | +0.056 | +0.047 |
| width ~ log_ratio | -0.602 | -0.589 |

## Defect classes (count / % vs full %)

| class | subset n | subset % | full % |
|---|---|---|---|
| clean | 3590 | 87.65 | 87.61 |
| degen_y1 | 216 | 5.27 | 5.28 |
| ne_y1 | 131 | 3.20 | 3.19 |
| ne_y0 | 104 | 2.54 | 2.55 |
| cot_leak | 37 | 0.90 | 0.91 |
| degen_y0 | 14 | 0.34 | 0.34 |
| dup | 4 | 0.10 | 0.11 |

## Inversion by length-ratio octile (%)

| octile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| full | 1.2 | 2.0 | 3.1 | 4.7 | 8.2 | 14.6 | 19.7 | 47.8 |
| subset | 1.9 | 2.2 | 2.8 | 4.6 | 7.6 | 15.4 | 20.4 | 46.1 |

## Instruction-ID marginal (subset share vs full share, %)

| instruction_id | subset % | full % |
|---|---|---|
| last_word:last_word_answer | 2.136 | 2.142 |
| detectable_format:bigram_wrapping | 2.127 | 2.126 |
| last_word:last_word_sent | 2.127 | 2.125 |
| punctuation:no_comma | 2.108 | 2.108 |
| keywords:start_end | 2.099 | 2.102 |
| keywords:word_once | 2.089 | 2.092 |
| detectable_format:sentence_hyphens | 2.099 | 2.091 |
| change_case:english_capital | 2.089 | 2.086 |
| length_constraints:number_sentences | 2.089 | 2.086 |
| keywords:exclude_word_harder | 2.080 | 2.083 |
| detectable_format:title | 2.080 | 2.081 |
| keywords:keyword_specific_position | 2.080 | 2.081 |
| keywords:no_adjacent_consecutive | 2.080 | 2.078 |
| letters:letter_counting2 | 2.071 | 2.075 |
| detectable_format:square_brackets | 2.071 | 2.073 |
| count:count_unique | 2.071 | 2.069 |
| punctuation:punctuation_exclamation | 2.061 | 2.068 |
| keywords:existence | 2.061 | 2.063 |
| length_constraints:nth_paragraph_first_word | 2.061 | 2.062 |
| keywords:forbidden_words | 2.061 | 2.062 |
| detectable_content:postscript | 2.061 | 2.059 |
| keywords:palindrome | 2.061 | 2.058 |
| keywords:letter_frequency | 2.061 | 2.058 |
| punctuation:punctuation_dot | 2.052 | 2.056 |
| first_word:first_word_answer | 2.061 | 2.056 |
| paragraphs:paragraphs2 | 2.052 | 2.053 |
| copy:repeat_phrase | 2.052 | 2.053 |
| detectable_content:number_placeholders | 2.052 | 2.052 |
| letters:letter_counting | 2.052 | 2.052 |
| detectable_format:number_highlighted_sections | 2.052 | 2.051 |
| detectable_format:number_bullet_lists | 2.052 | 2.050 |
| paragraphs:paragraphs | 2.052 | 2.048 |
| keywords:frequency | 2.052 | 2.047 |
| count:count_increment_word | 2.043 | 2.046 |
| startend:end_checker | 2.043 | 2.041 |
| keywords:word_count_different_numbers | 2.043 | 2.037 |
| change_case:capital_word_frequency | 2.033 | 2.034 |
| length_constraints:number_words | 2.024 | 2.028 |
| count:lowercase_counting | 2.024 | 2.023 |
| first_word:first_word_sent | 2.014 | 2.018 |
| startend:quotation | 2.005 | 2.001 |
| change_case:english_lowercase | 1.977 | 1.981 |
| length_constraints:number_paragraphs | 1.977 | 1.976 |
| detectable_format:multiple_sections | 1.968 | 1.969 |
| count:counting_composition | 1.808 | 1.809 |
| language:response_language | 1.771 | 1.771 |
| combination:two_responses | 0.787 | 0.787 |
| combination:repeat_prompt | 0.759 | 0.763 |
| copy:copying_multiple | 0.731 | 0.728 |
| new:copy_span_idx | 0.731 | 0.726 |
| copy:copy | 0.712 | 0.714 |
| detectable_format:json_format | 0.703 | 0.706 |
| detectable_format:constrained_response | 0.703 | 0.703 |
| copy:copying_simple | 0.693 | 0.695 |

## Overall: ALL CHECKS PASS
