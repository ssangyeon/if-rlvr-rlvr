"""Offline tests for mode-aware LLM-verifier score aggregation."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np


RAY_TRAINER = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "ppo" / "ray_trainer.py"


def load_collector():
    """Compile only the collector so the test needs no Ray, Torch, or GPU."""
    module = ast.parse(RAY_TRAINER.read_text(encoding="utf-8"), filename=str(RAY_TRAINER))
    function = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "collect_if_llm_verifier_metrics"
    )
    isolated = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace = {"np": np}
    exec(compile(isolated, str(RAY_TRAINER), "exec"), namespace)
    return namespace["collect_if_llm_verifier_metrics"]


COLLECT = load_collector()


class IntentCheckMetricTests(unittest.TestCase):
    def test_missing_mode_preserves_legacy_positive_score_filter(self):
        metrics = COLLECT(
            {
                "llm_verifier_called": [1, 1, 1, 1, 1],
                "llm_verifier_score": [0, 1, 5, -1, np.nan],
            }
        )
        self.assertEqual(metrics["if_llm_verifier/score_mean"], 3.0)
        self.assertEqual(metrics["if_llm_verifier/score_min"], 1.0)
        self.assertEqual(metrics["if_llm_verifier/score_max"], 5.0)

    def test_explicit_geval_mode_preserves_legacy_behavior(self):
        metrics = COLLECT(
            {
                "llm_verifier_called": [1, 1, 1],
                "llm_verifier_score": [0, 1, 9],
                "llm_verifier_mode": ["geval", "geval", "geval"],
            }
        )
        self.assertEqual(metrics["if_llm_verifier/score_mean"], 5.0)
        self.assertEqual(metrics["if_llm_verifier/score_min"], 1.0)
        self.assertEqual(metrics["if_llm_verifier/score_max"], 9.0)

    def test_intentcheck_counts_no_and_yes_but_excludes_errors_and_uncalled(self):
        metrics = COLLECT(
            {
                "llm_verifier_called": [1, 1, 1, 1, 0, 1],
                "llm_verifier_score": [0, 1, -1, np.nan, 1, 2],
                "llm_verifier_mode": ["intentcheck"] * 6,
            }
        )
        self.assertEqual(metrics["if_llm_verifier/called_count"], 5.0)
        self.assertEqual(metrics["if_llm_verifier/score_mean"], 0.5)
        self.assertEqual(metrics["if_llm_verifier/score_min"], 0.0)
        self.assertEqual(metrics["if_llm_verifier/score_max"], 1.0)

    def test_mixed_modes_apply_their_own_valid_ranges(self):
        metrics = COLLECT(
            {
                "llm_verifier_called": [1, 1, 1, 1],
                "llm_verifier_score": [0, 1, 0, 8],
                "llm_verifier_mode": ["intentcheck", "intentcheck", "geval", "geval"],
            }
        )
        self.assertEqual(metrics["if_llm_verifier/score_mean"], 3.0)
        self.assertEqual(metrics["if_llm_verifier/score_min"], 0.0)
        self.assertEqual(metrics["if_llm_verifier/score_max"], 8.0)

    def test_mode_shape_mismatch_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "equal shape"):
            COLLECT(
                {
                    "llm_verifier_called": [1, 1],
                    "llm_verifier_score": [0, 1],
                    "llm_verifier_mode": ["intentcheck"],
                }
            )


if __name__ == "__main__":
    unittest.main()
