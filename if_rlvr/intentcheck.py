"""IFDecorator IntentCheck prompt and a strict, reward-independent verdict parser.

Prompt attribution: IFDecorator, arXiv:2508.04632v2, printed pages 27–28.
Paper: https://arxiv.org/abs/2508.04632v2
Authors' repository: https://github.com/guox18/IFDecorator
License: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/ .

The prompt wording follows the paper; this module only exposes the prompt and
parses its final YES/NO determination. It does not implement the paper's reward
combination or alter a caller's constraint reward.
"""

from __future__ import annotations

import re


INTENTCHECK_PROMPT = """Structured Evaluation Protocol
Perform a rigorous analysis of the instruction-response pair through these sequential checks:
1. Instruction-Response Alignment
• Verify explicit understanding of core instruction objectives
• Check for missing required components from instructions
2. Logical Coherence Evaluation
• Trace logical flow from instruction premises to response conclusions
• Detect reasoning gaps or unwarranted assumptions
• Flag contradictions within the response
3. Context-aware Instruction Verification
• Analyze response against instruction type:
– Query-type: Verify question resolution completeness
– Task-type: Validate step-by-step executable logic
– Creative-type: Assess objective-aligned originality
4. Safety & Compliance Check
• Identify potential risks or harmful implications
• Verify adherence to ethical guidelines

Evaluation Parameters
• Strict true/false determination for each checkpoint
• Zero tolerance for partial fulfillment
• Mandatory failure for any single unmet criterion

Evaluation Target
**Instruction:**
<Instruction>
{instruction_wo_strict}
</Instruction>
**Response:**
<Response>
{response}
</Response>

First, present the analysis in an ordered checklist format. Then, conclude with a final verdict using strict formatting:
**Final Verification:** <YES/NO>"""


_LABEL = r"Final[ \t]+Verification"
_LABEL_OCCURRENCES = re.compile(rf"(?<![A-Za-z0-9]){_LABEL}(?![A-Za-z0-9])", re.IGNORECASE)

# Keep label/verdict markup balanced. A colon may be inside the label's
# emphasis or immediately after it. A heading-style label can instead end at
# a newline, with the vote on the next line.
_LABEL_FORMS = [rf"{_LABEL}[ \t]*:", rf"{_LABEL}[ \t]*\r?\n"]
_VERDICT_FORMS = [r"YES", r"NO"]
for _marker in ("**", "__", "*", "_", "`"):
    _escaped = re.escape(_marker)
    _LABEL_FORMS.extend(
        [
            rf"{_escaped}{_LABEL}[ \t]*:{_escaped}",
            rf"{_escaped}{_LABEL}{_escaped}[ \t]*:",
            rf"{_escaped}{_LABEL}{_escaped}[ \t]*\r?\n",
        ]
    )
    _VERDICT_FORMS.append(rf"{_escaped}(?:YES|NO){_escaped}")

_FINAL_SECTION = re.compile(
    r"(?:\#{1,6}[ \t]+)?"
    + "(?:" + "|".join(_LABEL_FORMS) + ")"
    + r"\s*(?P<verdict>"
    + "|".join(_VERDICT_FORMS)
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_STANDALONE_VERDICT = re.compile("(?:" + "|".join(_VERDICT_FORMS) + r")[.!?]?", re.IGNORECASE)
_ALTERNATIVE_TAIL = re.compile(
    r"^\s*(?:[/\\|,;]|\b(?:or|and)\b|[\[(])\s*(?:or\s+)?"
    r"(?:\*\*|__|\*|_|`)?(?:YES|NO)(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def extract_intentcheck_verdict(text: str) -> int:
    """Return 1 for a single labeled YES, or 0 for labeled NO.

    Ordered-checklist analysis may precede the final section. The paper's
    ``Final Verification`` label is mandatory; numeric scores, standalone
    YES/NO words, multiple labels (even agreeing ones), and ambiguous alternate
    verdicts are rejected. Explanatory prose after the immediate labeled vote
    is allowed. This deliberately has no substring/number fallback.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("IntentCheck judgment must be a nonempty string")
    occurrences = list(_LABEL_OCCURRENCES.finditer(text))
    if len(occurrences) != 1:
        raise ValueError("IntentCheck requires exactly one Final Verification label")

    # The label must begin its own final section, not occur inside unrelated
    # prose such as 'I would give Final Verification: YES'.
    start = text.rfind("\n", 0, occurrences[0].start()) + 1
    final_section = text[start:].strip()
    match = _FINAL_SECTION.match(final_section)
    tail = final_section[match.end() :] if match is not None else ""
    if match is None:
        # A balanced emphasis span can cover the whole final section rather
        # than the label and vote separately: **Final Verification: YES**.
        for marker in ("**", "__", "*", "_", "`"):
            if final_section.startswith(marker):
                inner = final_section[len(marker) :]
                candidate = _FINAL_SECTION.match(inner)
                if candidate is not None and inner[candidate.end() :].startswith(marker):
                    match = candidate
                    tail = inner[candidate.end() + len(marker) :]
                    break
    if match is None:
        raise ValueError("IntentCheck final section must begin with a labeled YES or NO")
    if tail.startswith(("*", "_", "`")) or (tail.strip() and not tail.strip().strip("*_`")):
        raise ValueError("IntentCheck verdict contains unbalanced markdown")
    if _ALTERNATIVE_TAIL.match(tail) or any(
        _STANDALONE_VERDICT.fullmatch(line.strip()) for line in tail.splitlines() if line.strip()
    ):
        raise ValueError("IntentCheck contains ambiguous alternative verdicts")
    verdict = match.group("verdict").strip("*_`").upper()
    return 1 if verdict == "YES" else 0


__all__ = ["INTENTCHECK_PROMPT", "extract_intentcheck_verdict"]
