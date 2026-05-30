#!/usr/bin/env python
"""Reproducibility test: verl vendored IFEval verifier vs the open-instruct oracle.

Compares ``verl.utils.reward_score.ifeval_oi.verifier.score_ifeval`` (CANDIDATE)
against the *actual* ``open_instruct.ground_truth_utils.IFEvalVerifier`` (ORACLE)
over a large, stratified sample of real ``allenai/IF_multi_constraints_upto5``
ground-truth constraints crossed with a diverse pool of candidate responses
(empty / whitespace / random / structured / multilingual / thinking-wrapped / etc.).

Several IFEval checkers draw from ``random`` (when a kwarg is absent) and three use
``langdetect`` (non-deterministic unless seeded). open-instruct does not seed these,
so to test *code equivalence* fairly we (a) fix ``langdetect.DetectorFactory.seed``
once and (b) re-seed ``random`` to the same value immediately before each oracle and
candidate call. Both implementations import the same ``random`` / ``langdetect``
singletons, so seeding affects both identically.

Run:
    PYTHONPATH=/lustre/justinseo/if-verl/verl \
      /home/justinseo/miniconda3/envs/verl/bin/python \
      recipe/if_rlvr/tests/reproduce_ifeval.py --per-type 80
"""

from __future__ import annotations

import argparse
import ast
import collections
import random
import sys
import time
import types

OPEN_INSTRUCT = "/lustre/justinseo/if-verl/open-instruct"
sys.path.insert(0, OPEN_INSTRUCT)
import os as _os  # noqa: E402
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # recipe/if_rlvr
sys.modules.setdefault("beaker", types.ModuleType("beaker"))  # open_instruct.utils imports beaker

import langdetect  # noqa: E402

langdetect.DetectorFactory.seed = 0  # deterministic language detection for the comparison

from datasets import load_dataset  # noqa: E402

from open_instruct.ground_truth_utils import IFEvalVerifier  # noqa: E402  GOLD ORACLE
from ifeval_oi.verifier import score_ifeval  # noqa: E402  CANDIDATE (self-contained, verl-independent)

ORACLE = IFEvalVerifier()


def oracle_score(pred: str, label: str, seed: int) -> float:
    random.seed(seed)
    return ORACLE(None, pred, label).score


def candidate_score(pred: str, label: str, seed: int) -> float:
    random.seed(seed)
    return score_ifeval(pred, label)


# ---------------------------------------------------------------- response pool
_LONG = (
    "<<A Structured Title>>\n\n"
    "First, here is the opening paragraph. It contains several sentences! It also has a question? Yes.\n\n"
    "***\n\n"
    "* bullet one\n* bullet two\n* bullet three\n\n"
    "SECTION 1\nSome *highlighted* content and a [placeholder] and another [placeholder2].\n\n"
    "My answer is yes.\n\nP.S. this is a postscript."
)
_MULTI = "Dies ist ein deutscher Satz. 这是一个中文句子。 Это русское предложение. هذه جملة عربية."


def gen_responses(label: str) -> list[str]:
    base = [
        "",
        "   \n\t  ",
        "ok",
        "This is a simple, short answer with a few words.",
        _LONG,
        _MULTI,
        "THIS IS AN ALL CAPS RESPONSE WITH SEVERAL WORDS IN IT.",
        "this is an all lowercase response with several words in it.",
        '{"name": "value", "count": 3, "items": ["a", "b"]}',
        "<<My Great Title>>\n\nSome body paragraph here, with content.",
        "She said \"hello\" and then left.\n\nThe end.",
        ("My answer is maybe.\n\n***\n\nthis is a second paragraph all in lowercase letters "
         "and it mentions the coast once."),
    ]
    # thinking-wrapped / tag variants exercise remove_thinking_section
    wrapped = [
        "<think>let me reason about this carefully and at length</think>\n\n" + _LONG,
        "<think>reasoning here</think>" + base[9],
        "<|assistant|>" + base[3],
        "<answer>" + base[10] + "</answer>",
        "<think>a</think> middle </think> final answer with <<Title>> here",
        "reasoning without a close tag " + base[4],
    ]
    return base + wrapped


def instruction_ids(label: str) -> list[str]:
    parsed = ast.literal_eval(label)[0]
    return list(parsed["instruction_id"])


def crafted_passes(label: str) -> list[str]:
    """Best-effort responses designed to SATISFY a single-constraint label.

    Exercises the True-path of harder checkers (copy/last_word/end/quotation/...).
    Imperfect crafts merely fail (agreement is still validated); the goal is to
    observe oracle==1.0 agreement for as many constraint types as possible.
    """
    parsed = ast.literal_eval(label)[0]
    ids, kw = parsed["instruction_id"], parsed["kwargs"]
    if len(ids) != 1:
        return []
    iid, k = ids[0], (kw[0] or {})
    out = []
    body = "This is a reasonable response paragraph with several plain words in it"
    if iid == "keywords:existence" and k.get("keywords"):
        out.append("Here is text that naturally includes " + " and ".join(k["keywords"]) + " in context.")
    elif iid == "keywords:forbidden_words":
        out.append("A neutral sentence about geometry and weather and travel plans.")
    elif iid == "last_word:last_word_answer" and k.get("last_word"):
        out.append(f"My final answer here ends with the word {k['last_word']}")
    elif iid == "last_word:last_word_sent" and k.get("last_word"):
        out.append(f"This is a sentence. Here is the final one ending with {k['last_word']}.")
    elif iid == "startend:end_checker" and k.get("end_phrase"):
        out.append(f"{body}. {k['end_phrase']}")
    elif iid == "startend:quotation":
        out.append(f'"{body} and it is fully wrapped in quotation marks."')
    elif iid == "detectable_content:postscript" and k.get("postscript_marker"):
        out.append(f"{body}.\n\n{k['postscript_marker']} this is the postscript content.")
    elif iid == "detectable_content:number_placeholders" and k.get("num_placeholders"):
        n = int(k["num_placeholders"])
        out.append(body + " " + " ".join(f"[placeholder{i}]" for i in range(n + 1)))
    elif iid == "combination:two_responses":
        out.append("First complete response here.\n******\nSecond complete response here.")
    elif iid in ("combination:repeat_prompt",) and k.get("prompt_to_repeat"):
        out.append(f"{k['prompt_to_repeat']}\n\nNow my answer follows.")
    elif iid in ("copy:copying_simple", "copy:copy") and k.get("prompt_to_repeat"):
        out.append(k["prompt_to_repeat"])
    elif iid == "copy:copying_multiple" and k.get("prompt_to_repeat"):
        n = int(k.get("N", 2))
        sep = "\n" + "*" * 6 + "\n"
        out.append(sep.join([k["prompt_to_repeat"]] * n))
    elif iid == "detectable_format:title":
        out.append("<<A Fitting Title>>\n\n" + body + ".")
    elif iid == "change_case:english_lowercase":
        out.append("this is a fully lowercase response with several words and no capitals.")
    elif iid == "change_case:english_capital":
        out.append("THIS IS A FULLY UPPERCASE RESPONSE WITH SEVERAL WORDS IN IT.")
    elif iid == "punctuation:no_comma":
        out.append("This sentence deliberately avoids using any commas at all so it should pass.")
    elif iid == "detectable_format:multiple_sections" and k.get("section_spliter") and k.get("num_sections"):
        sp, n = k["section_spliter"], int(k["num_sections"])
        out.append("\n".join(f"{sp} {i + 1}\nContent for section {i + 1} here." for i in range(n)))
    elif iid == "length_constraints:number_paragraphs" and k.get("num_paragraphs"):
        n = int(k["num_paragraphs"])
        out.append(("\n\n***\n\n").join(f"Paragraph number {i + 1} content." for i in range(n)))
    elif iid == "first_word:first_word_answer" and k.get("first_word"):
        out.append(f"{k['first_word']} is how this answer begins and then continues with more text.")
    elif iid == "first_word:first_word_sent" and k.get("first_word"):
        fw = k["first_word"]
        out.append(f"{fw} opens this sentence. {fw} also opens the next one here.")
    elif iid == "detectable_format:number_bullet_lists" and k.get("num_bullets"):
        n = int(k["num_bullets"])
        out.append("Here are the points:\n" + "\n".join(f"* bullet item number {i + 1}" for i in range(n)))
    return out


def collect_labels(per_type: int):
    """Stratified: up to `per_type` distinct label strings per instruction_id (covers all 54)."""
    ds = load_dataset("allenai/IF_multi_constraints_upto5", split="train")
    by_type: dict[str, int] = collections.defaultdict(int)
    chosen: dict[str, None] = {}  # ordered set of label strings
    all_ids = set()
    for gt in ds["ground_truth"]:
        try:
            ids = instruction_ids(gt)
        except Exception:
            continue
        all_ids.update(ids)
        if any(by_type[i] < per_type for i in ids):
            if gt not in chosen:
                chosen[gt] = None
                for i in ids:
                    by_type[i] += 1
    return list(chosen.keys()), all_ids, by_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=80, help="labels collected per instruction_id")
    ap.add_argument("--max-mismatch-print", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()
    labels, all_ids, by_type = collect_labels(args.per_type)
    print(f"collected {len(labels)} distinct labels covering {len(all_ids)} instruction types "
          f"in {time.time()-t0:.1f}s")

    n_pairs = 0
    mismatches = []
    # per-instruction-type coverage of oracle True/False outcomes (single-constraint labels only,
    # so the per-type pass/fail is unambiguous)
    type_true = collections.Counter()
    type_false = collections.Counter()
    type_pairs = collections.Counter()
    score_hist = collections.Counter()
    determinism_flaky = 0

    t0 = time.time()
    for li, label in enumerate(labels):
        ids = instruction_ids(label)
        single = ids[0] if len(ids) == 1 else None
        responses = gen_responses(label) + crafted_passes(label)
        for ri, resp in enumerate(responses):
            seed = li * 1000 + ri
            o = oracle_score(resp, label, seed)
            c = candidate_score(resp, label, seed)
            n_pairs += 1
            score_hist[round(o, 4)] += 1
            if o != c:
                if len(mismatches) < 100000:
                    mismatches.append((label, resp, o, c))
            if single is not None:
                type_pairs[single] += 1
                if o == 1.0:
                    type_true[single] += 1
                elif o == 0.0:
                    type_false[single] += 1
            # determinism probe: re-run oracle WITHOUT fixed seed twice (real training behaviour)
            if ri == 4 and li < 400:
                a = ORACLE(None, resp, label).score
                b = ORACLE(None, resp, label).score
                if a != b:
                    determinism_flaky += 1

    dt = time.time() - t0
    print(f"\ncompared {n_pairs} (label,response) pairs in {dt:.1f}s "
          f"({n_pairs/max(dt,1e-9):.0f}/s)")
    print(f"MISMATCHES: {len(mismatches)}")
    print("score distribution (oracle):", dict(sorted(score_hist.items())))

    # coverage: every type must appear; report those lacking BOTH outcomes
    missing = sorted(all_ids - set(type_pairs.keys()) - {i for lab in labels for i in instruction_ids(lab)})
    no_true = sorted(t for t in type_pairs if type_true[t] == 0)
    no_false = sorted(t for t in type_pairs if type_false[t] == 0)
    print(f"\nsingle-constraint types exercised: {len(type_pairs)}")
    print(f"types with NO observed pass (oracle never 1.0): {no_true}")
    print(f"types with NO observed fail (oracle never 0.0): {no_false}")
    print(f"determinism probe: {determinism_flaky} flaky cases out of ~400 (unseeded re-run mismatch)")

    if mismatches:
        print(f"\n===== FIRST {min(args.max_mismatch_print, len(mismatches))} MISMATCHES =====")
        for label, resp, o, c in mismatches[: args.max_mismatch_print]:
            print(f"  oracle={o} candidate={c}\n    label={label[:200]}\n    resp={resp[:160]!r}")
        print("\nRESULT: FAIL")
        sys.exit(1)
    else:
        print("\nRESULT: PASS  (100% agreement between verl vendored verifier and open-instruct oracle)")


if __name__ == "__main__":
    main()
