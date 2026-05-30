#!/usr/bin/env python
"""End-to-end integration smoke test for the IF-RLVR recipe (CPU only).

Validates against real verl machinery:
  [1] verl RLHFDataset loads the parquet + applies the Qwen3 chat template for BOTH
      enable_thinking modes; dispatch fields survive.
  [2] ORTHOGONAL path (recommended): verl's stable custom_reward_function hook
      (if_reward_fn.compute_score) loaded through load_reward_manager with the default
      `naive` reward manager -> reward = fraction of constraints (no non-stop penalty).
  [3] OPTIONAL coupled path: the IFRewardManager (importlib) -> adds the non-stop penalty.

Run:
    PYTHONPATH=/lustre/justinseo/if-verl/verl \
      /home/justinseo/miniconda3/envs/verl/bin/python recipe/if_rlvr/tests/integration_smoke.py
"""

from __future__ import annotations

import asyncio
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl import DataProto
from verl.trainer.ppo.reward import load_reward_manager

MODEL = os.environ.get("IF_TEST_MODEL", "Qwen/Qwen3-8B")
PARQUET = os.environ.get("IF_TEST_PARQUET", "/lustre/justinseo/if-verl/data/ifeval_multi/val.parquet")
RECIPE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_FN_PATH = os.path.join(RECIPE_DIR, "if_reward_fn.py")
MANAGER_PATH = os.path.join(RECIPE_DIR, "if_reward_manager.py")

TITLE_GT = "[{'instruction_id': ['detectable_format:title'], 'kwargs': [None]}]"
TWO_GT = ("[{'instruction_id': ['detectable_format:title', 'change_case:english_lowercase'], "
          "'kwargs': [None, None]}]")

PASS = 0


def check(name, got, want):
    global PASS
    ok = abs(float(got) - float(want)) < 1e-6
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")
    PASS += int(not ok)


def build_item(tokenizer, ground_truth, response_text, max_resp=96, truncated=False):
    prompt_ids = tokenizer("Write something.", add_special_tokens=False)["input_ids"]
    resp_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if truncated:
        resp = (resp_ids * ((max_resp // max(len(resp_ids), 1)) + 1))[:max_resp]
        valid = max_resp
    else:
        resp = (resp_ids + [tokenizer.eos_token_id])[:max_resp]
        valid = len(resp)
        resp = resp + [pad_id] * (max_resp - len(resp))
    resp = resp[:max_resp]
    P = len(prompt_ids)
    attn = [1] * P + [1] * valid + [0] * (max_resp - valid)
    item = {
        "prompts": torch.tensor([prompt_ids], dtype=torch.long),
        "responses": torch.tensor([resp], dtype=torch.long),
        "attention_mask": torch.tensor([attn], dtype=torch.long),
        "data_source": np.array(["ifeval"], dtype=object),
        "reward_model": np.array([{"style": "rule", "ground_truth": ground_truth}], dtype=object),
    }
    return DataProto.from_single_dict(item)


def base_reward_cfg():
    return {
        "reward": {
            "custom_reward_function": {"path": None, "name": None},
            "reward_manager": {"source": "register", "name": "naive", "module": {"path": None, "name": None}},
            "sandbox_fusion": {"url": None, "max_concurrent": 64, "memory_limit_mb": 1024},
            "reward_model": {"enable": False},
        },
        "data": {"reward_fn_key": "data_source"},
    }


def test_dataset_loading(tokenizer):
    print("\n[1] verl RLHFDataset + Qwen3 chat template (both thinking modes)")
    from verl.utils.dataset.rl_dataset import RLHFDataset

    for enable_thinking in (True, False):
        cfg = OmegaConf.create({
            "prompt_key": "prompt", "max_prompt_length": 4096, "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": 1, "truncation": "error",
            "apply_chat_template_kwargs": {"enable_thinking": enable_thinking},
            "cache_dir": "~/.cache/verl/rlhf", "shuffle": False,
        })
        ds = RLHFDataset(data_files=PARQUET, tokenizer=tokenizer, config=cfg, processor=None, max_samples=16)
        row = ds[0]
        rendered = tokenizer.apply_chat_template(
            row["raw_prompt"], add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
        has_empty_think = "<think>\n\n</think>" in rendered
        check(f"thinking={enable_thinking}: empty-think-block-in-prompt matches mode", has_empty_think, not enable_thinking)
        check(f"thinking={enable_thinking}: data_source survives", row["data_source"] == "ifeval", True)


def test_orthogonal_path(tokenizer):
    print("\n[2] ORTHOGONAL: custom_reward_function (if_reward_fn) + default naive manager")
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    cfg = OmegaConf.create(base_reward_cfg())
    cfg.reward.custom_reward_function.path = REWARD_FN_PATH
    cfg.reward.custom_reward_function.name = "compute_score"
    mgr = load_reward_manager(cfg, tokenizer)
    print(f"  loaded manager: {type(mgr).__name__} (verl-native); reward fn = if_reward_fn.compute_score")
    check("default manager is verl-native NaiveRewardManager", type(mgr).__name__ == "NaiveRewardManager", True)

    def run(dp):
        return loop.run_until_complete(mgr.run_single(dp))

    # reward = raw fraction (verification_reward default 1.0); NO non-stop penalty in this path
    check("title satisfied -> 1.0", run(build_item(tokenizer, TITLE_GT, "<<A Title>>\n\nbody."))["reward_score"], 1.0)
    check("title NOT satisfied -> 0.0", run(build_item(tokenizer, TITLE_GT, "no title here."))["reward_score"], 0.0)
    check("1/2 satisfied -> 0.5", run(build_item(tokenizer, TWO_GT, "<<A Title>>\n\nHas CAPS."))["reward_score"], 0.5)
    check("thinking-wrapped title -> 1.0 (strip works)",
          run(build_item(tokenizer, TITLE_GT, "<think>plan</think>\n<<Real>>\nbody"))["reward_score"], 1.0)
    check("truncated+pass -> 1.0 (orthogonal path has NO non-stop penalty)",
          run(build_item(tokenizer, TITLE_GT, "<<A Title>>\n\nbody", truncated=True))["reward_score"], 1.0)
    loop.close()


def test_optional_manager(tokenizer):
    print("\n[3] OPTIONAL coupled: IFRewardManager (importlib) -> adds non-stop penalty")
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    cfg = OmegaConf.create(base_reward_cfg())
    cfg.reward.reward_manager.source = "importlib"
    cfg.reward.reward_manager.name = "IFRewardManager"
    cfg.reward.reward_manager.module.path = MANAGER_PATH
    mgr = load_reward_manager(cfg, tokenizer)
    print(f"  loaded manager: {type(mgr).__name__}; verification_reward={mgr.verification_reward} "
          f"stop_token_ids={sorted(mgr.stop_token_ids)}")
    check("loaded class is IFRewardManager", type(mgr).__name__ == "IFRewardManager", True)

    def run(dp):
        return loop.run_until_complete(mgr.run_single(dp))

    check("stopped+title -> 10.0", run(build_item(tokenizer, TITLE_GT, "<<T>>\nbody"))["reward_score"], 10.0)
    r = run(build_item(tokenizer, TITLE_GT, "<<T>>\nbody", truncated=True))
    check("truncated+pass -> non_stop_penalty 0.0 (override)", r["reward_score"], 0.0)
    check("  is_non_stop flag", r["reward_extra_info"]["is_non_stop"], 1.0)
    loop.close()


def main():
    print(f"loading tokenizer {MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    test_dataset_loading(tokenizer)
    test_orthogonal_path(tokenizer)
    test_optional_manager(tokenizer)
    print(f"\n{'='*60}\nRESULT: {'PASS' if PASS == 0 else f'FAIL ({PASS} checks failed)'}")
    raise SystemExit(1 if PASS else 0)


if __name__ == "__main__":
    main()
