"""Train the structured FinanceBench agent on Tinker.

Default model is the post-trained Qwen3.5-9B, deliberately not the Base model.
All experiment knobs are environment variables so each run is reproducible from
its logged manifest.
"""
import asyncio
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from tinker_cookbook import cli_utils
from tinker_cookbook.rl import train
from tinker_cookbook.rl.rollout_limits import TerminationRewardPolicy
import finance_env

BASE_MODEL = os.environ.get("MODEL", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")


def load_tinker_key() -> str:
    path = os.path.expanduser("~/.config/tinker/key")
    value = open(path).read().strip()
    if "=" in value:
        value = value.split("=", 1)[1].strip()
    if value.startswith("tinker-"):
        value = "tml-" + value[len("tinker-"):]
    return value


def tinker_evaluator_builder(split_name: str, dataset):
    """Build the pinned SDK evaluator from an already materialized RLDataset."""
    from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
    from tinker_cookbook.rl.train import _sanitize_filename_component
    return lambda: RLTestSetEvaluator(
        dataset=dataset,
        max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
        name="financebench_" + _sanitize_filename_component(split_name),
        num_groups_to_log=int(os.environ.get("EVAL_GROUPS_TO_LOG", "4")),
    )


async def main():
    steps = int(os.environ.get("STEPS", "50"))
    batch = int(os.environ.get("BATCH", "4"))
    group = int(os.environ.get("GROUP", "8"))
    lr = float(os.environ.get("LR", "2e-5"))
    max_turns = int(os.environ.get("MAX_TURNS", "8"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
    lora_rank = int(os.environ.get("LORA_RANK", "32"))
    temperature = float(os.environ.get("TEMPERATURE", "1.0"))
    compute_post_kl = os.environ.get("COMPUTE_POST_KL", "false").strip().lower() in {"1", "true", "yes", "on"}
    zero_reward_on_limit = os.environ.get("ZERO_REWARD_ON_LIMIT", "true").strip().lower() in {"1", "true", "yes", "on"}
    split_name = os.environ.get("SPLIT_NAME", "train96")
    renderer = os.environ.get("RENDERER", "nemotron3_ultra")
    seed = int(os.environ.get("SEED", "0"))
    remove_constant_reward_groups = os.environ.get("REMOVE_CONSTANT_REWARD_GROUPS", "false").strip().lower() in {"1", "true", "yes", "on"}
    run_name = os.environ.get("RUN_NAME", f"finbench_structured_{BASE_MODEL.lower().replace('/', '-')}_bs{batch}_gs{group}_lr{lr}_{datetime.now():%Y%m%d-%H%M%S}")
    log_path = os.environ.get("LOG_PATH", f"/tmp/tinker-examples/rl_finance/{run_name}")
    cli_utils.check_log_dir(log_path, behavior_if_exists="delete")

    builder = finance_env.FinanceDatasetBuilder(
        model_name_for_tokenizer=BASE_MODEL,
        batch_size=batch,
        group_size=group,
        renderer_name=renderer,
        max_turns=max_turns,
        max_generation_tokens=max_tokens,
        format_coef=0.0,
        seed=seed,
        split_name=split_name,
    )
    manifest = {
        "model": BASE_MODEL,
        "renderer": renderer,
        "split": split_name,
        "steps": steps,
        "batch": batch,
        "group": group,
        "learning_rate": lr,
        "max_turns": max_turns,
        "max_tokens": max_tokens,
        "lora_rank": lora_rank,
        "temperature": temperature,
        "compute_post_kl": compute_post_kl,
        "zero_reward_on_limit": zero_reward_on_limit,
        "seed": seed,
        "harness": "structured_sparse_agent_v2_grounded_reward",
        "reward_contract": {"answer_primary": True, "evidence_grounding": True, "finish_bonus": False, "format_penalty": False, "finish_required": finance_env.RewardConfig.from_env().require_finish},
        "reward_config": finance_env.RewardConfig.from_env().summary(),
        "remove_constant_reward_groups": remove_constant_reward_groups,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    eval_every = int(os.environ.get("EVAL_EVERY", "0"))
    eval_split = os.environ.get("EVAL_SPLIT", "eval")
    eval_dataset = None
    if eval_every > 0:
        eval_dataset, _ = await finance_env.FinanceDatasetBuilder(
            model_name_for_tokenizer=BASE_MODEL,
            batch_size=1,
            group_size=1,
            renderer_name=os.environ.get("RENDERER", "nemotron3_ultra"),
            max_turns=int(os.environ.get("EVAL_MAX_TURNS", os.environ.get("MAX_TURNS", "8"))),
            max_generation_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
            split_name=eval_split,
            seed=0,
        )()

    config = train.Config(
        model_name=BASE_MODEL,
        recipe_name="recipe_financebench_structured",
        renderer_name=renderer,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=lr,
        max_tokens=max_tokens,
        lora_rank=lora_rank,
        temperature=temperature,
        compute_post_kl=compute_post_kl,
        max_steps=steps,
        remove_constant_reward_groups=remove_constant_reward_groups,
        eval_every=int(os.environ.get("EVAL_EVERY", "0")),
        save_every=int(os.environ.get("SAVE_EVERY", "0")),
        evaluator_builders=[tinker_evaluator_builder(os.environ.get("EVAL_SPLIT", "eval"))] if int(os.environ.get("EVAL_EVERY", "0")) > 0 else [],
        rollout_json_export=True,
        enable_trace=True,
        num_groups_to_log=4,
    )
    await train.main(config)


if __name__ == "__main__":
    os.environ.setdefault("TINKER_API_KEY", load_tinker_key())
    asyncio.run(main())
