"""Offline regression tests for the paper prompt and explicit labeled verdicts."""

import importlib.util
from pathlib import Path
import string
import unittest


MODULE_PATH = Path(__file__).resolve().parents[2] / "if_rlvr" / "intentcheck.py"
SPEC = importlib.util.spec_from_file_location("intentcheck_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class IntentCheckPromptTests(unittest.TestCase):
    def test_exact_two_placeholders(self):
        fields = [field for _, field, _, _ in string.Formatter().parse(MODULE.INTENTCHECK_PROMPT) if field]
        self.assertEqual(fields, ["instruction_wo_strict", "response"])

    def test_renders_instruction_and_response_without_interpreting_payload_braces(self):
        prompt = MODULE.INTENTCHECK_PROMPT.format(
            instruction_wo_strict="Explain {x}", response='A response containing {"Score": 10}'
        )
        self.assertIn("<Instruction>\nExplain {x}\n</Instruction>", prompt)
        self.assertIn('<Response>\nA response containing {"Score": 10}\n</Response>', prompt)
        self.assertTrue(prompt.endswith("**Final Verification:** <YES/NO>"))

    def test_paper_checklist_and_parameters_are_retained(self):
        prompt = MODULE.INTENTCHECK_PROMPT
        for text in (
            "1. Instruction-Response Alignment",
            "2. Logical Coherence Evaluation",
            "3. Context-aware Instruction Verification",
            "4. Safety & Compliance Check",
            "Strict true/false determination for each checkpoint",
            "Zero tolerance for partial fulfillment",
            "Mandatory failure for any single unmet criterion",
            "First, present the analysis in an ordered checklist format.",
        ):
            self.assertIn(text, prompt)
        self.assertNotIn("Score", prompt)
        self.assertNotIn("1 to 10", prompt)

    def test_attribution_is_present(self):
        self.assertIn("2508.04632v2", MODULE.__doc__)
        self.assertIn("https://github.com/guox18/IFDecorator", MODULE.__doc__)
        self.assertIn("CC BY 4.0", MODULE.__doc__)


class IntentCheckVerdictTests(unittest.TestCase):
    def test_exact_paper_format_yes_and_no(self):
        self.assertEqual(MODULE.extract_intentcheck_verdict("**Final Verification:** YES"), 1)
        self.assertEqual(MODULE.extract_intentcheck_verdict("**Final Verification:** NO"), 0)

    def test_plain_labels_and_case_variants(self):
        for text, expected in (("Final Verification: YES", 1), ("final verification: no", 0),
                               ("FINAL VERIFICATION: YeS", 1)):
            with self.subTest(text=text):
                self.assertEqual(MODULE.extract_intentcheck_verdict(text), expected)

    def test_optional_balanced_label_and_vote_markdown(self):
        for marker in ("**", "__", "*", "_", "`"):
            for value, expected in (("YES", 1), ("NO", 0)):
                for label in (f"{marker}Final Verification:{marker}",
                              f"{marker}Final Verification{marker}:"):
                    text = f"{label} {marker}{value}{marker}"
                    with self.subTest(text=text):
                        self.assertEqual(MODULE.extract_intentcheck_verdict(text), expected)

    def test_heading_and_next_line_verdict(self):
        for text in ("### Final Verification:\nYES", "## **Final Verification:**\n**YES**",
                     "Final Verification\nYES", "**Final Verification**\nYES"):
            with self.subTest(text=text):
                self.assertEqual(MODULE.extract_intentcheck_verdict(text), 1)

    def test_balanced_markdown_around_complete_final_section(self):
        for marker in ("**", "__", "*", "_", "`"):
            for value, expected in (("YES", 1), ("NO", 0)):
                text = f"{marker}Final Verification: {value}{marker}"
                with self.subTest(text=text):
                    self.assertEqual(MODULE.extract_intentcheck_verdict(text), expected)

    def test_ordered_checklist_numbers_and_words_do_not_determine_vote(self):
        analysis = "1. Alignment: YES.\n2. Logic: NO.\n3. Context: 10.\n4. Safety: YES.\n"
        self.assertEqual(MODULE.extract_intentcheck_verdict(analysis + "**Final Verification:** NO"), 0)
        self.assertEqual(MODULE.extract_intentcheck_verdict(analysis + "**Final Verification:** YES"), 1)

    def test_surrounding_whitespace_and_crlf(self):
        self.assertEqual(MODULE.extract_intentcheck_verdict(" \r\n\t**Final Verification:**\r\n YES \r\n"), 1)

    def test_rejects_unlabeled_votes_and_numeric_scores(self):
        for text in ("YES", "NO", "1", "0", '{"Score": 10}', "1. YES\n2. NO",
                     "All checkpoints YES", "Final Verification: 1", "Final Verification: true",
                     "Final Verification: <YES/NO>"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MODULE.extract_intentcheck_verdict(text)

    def test_rejects_multiple_or_conflicting_finals(self):
        for a, b in (("YES", "NO"), ("NO", "YES"), ("YES", "YES"), ("NO", "NO")):
            text = f"**Final Verification:** {a}\nFinal Verification: {b}"
            with self.subTest(text=text), self.assertRaises(ValueError):
                MODULE.extract_intentcheck_verdict(text)

    def test_accepts_live_tulu_yes_and_no_with_explanations(self):
        examples = (
            ("**Final Verification:** YES\n\nThe response is fully aligned with the instruction.", 1),
            ("**Final Verification:** NO\n\nThe response does not meet the instruction.", 0),
        )
        for text, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(MODULE.extract_intentcheck_verdict(text), expected)

    def test_accepts_explanatory_prose_and_punctuation_after_vote(self):
        for value, expected in (("YES", 1), ("NO", 0)):
            for tail in (" because this is the assessment", "\nExplanation: all checks were considered",
                         ".", ".\n\nThe checklist supports this decision.",
                         "\nNo issues were omitted from the analysis."):
                with self.subTest(value=value, tail=tail):
                    self.assertEqual(MODULE.extract_intentcheck_verdict("Final Verification: " + value + tail), expected)

    def test_accepts_legacy_markdown_forms_with_explanatory_tail(self):
        for marker in ("**", "__", "*", "_", "`"):
            for value, expected in (("YES", 1), ("NO", 0)):
                for text in (f"{marker}Final Verification:{marker} {marker}{value}{marker}",
                             f"{marker}Final Verification: {value}{marker}"):
                    with self.subTest(text=text):
                        self.assertEqual(MODULE.extract_intentcheck_verdict(text + "\n\nThe response was assessed."), expected)

    def test_rejects_alternative_or_second_unlabeled_vote(self):
        for tail in ("\nNO", "\nYES", " / NO", " or NO", " and NO", ", NO", " (NO)",
                     "\nExplanation: all checks passed\n**NO**", "\nYES."):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                MODULE.extract_intentcheck_verdict("Final Verification: YES" + tail)

    def test_rejects_vote_word_prefixes(self):
        for word in ("YESman", "NOthing", "YES123", "NO1"):
            with self.subTest(word=word), self.assertRaises(ValueError):
                MODULE.extract_intentcheck_verdict("Final Verification: " + word)

    def test_rejects_missing_colon_inline_or_wrong_label(self):
        for text in ("Final Verification YES", "Verification: YES", "Final Verdict: YES",
                     "I would give Final Verification: YES", "1. Final Verification: YES"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MODULE.extract_intentcheck_verdict(text)

    def test_rejects_unbalanced_markdown(self):
        for text in ("**Final Verification: YES", "**Final Verification:** **YES",
                     "Final Verification: YES**", "Final Verification: `YES**",
                     "**Final Verification:**: YES"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MODULE.extract_intentcheck_verdict(text)

    def test_rejects_empty_and_non_string_input(self):
        for text in ("", " \n\t", None, 1, True, [], {}):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MODULE.extract_intentcheck_verdict(text)


if __name__ == "__main__":
    unittest.main()
