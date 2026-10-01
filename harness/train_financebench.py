"""Train the structured FinanceBench agent on Tinker.

Default model is Nemotron 3.5 Lightning.
All experiment knobs are environment variables so each run is reproducible from
its logged manifest.
"""
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import hashlib
import subprocess
import importlib.metadata
from training_audit import install_training_audit

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


def tinker_evaluator_builder(dataset, max_tokens: int, split_name: str = "dev"):
    """Capture an already-built dev dataset; cookbook factories are synchronous."""
    from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
    from tinker_cookbook.completers import TinkerTokenCompleter
    class MatchedDevEvaluator(RLTestSetEvaluator):
        async def __call__(self, sampling_client, **kwargs):
            policy = TinkerTokenCompleter(sampling_client, max_tokens=self.max_tokens,
                temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.2")))
            return await self.eval_token_completer(policy, **kwargs)
    def builder():
        return MatchedDevEvaluator(dataset=dataset, max_tokens=max_tokens, name="financebench_" + split_name)
    return builder


async def main():
    steps = int(os.environ.get("STEPS", "50"))
    batch = int(os.environ.get("BATCH", "8"))
    group = int(os.environ.get("GROUP", "8"))
    lr = float(os.environ.get("LR", "1e-5"))
    max_turns = int(os.environ.get("MAX_TURNS", "8"))
    split_name = os.environ.get("SPLIT_NAME", "train")
    renderer = os.environ.get("RENDERER", "nemotron3_ultra")
    seed = int(os.environ.get("SEED", "0"))
    dev_split = os.environ.get("DEV_SPLIT", os.environ.get("EVAL_SPLIT", "dev"))
    if split_name not in ("train", "train96") or dev_split != "dev":
        raise ValueError("Training uses train/train96; checkpoint evaluation must use dev, never eval")
    remove_constant_reward_groups = os.environ.get("REMOVE_CONSTANT_REWARD_GROUPS", "true").strip().lower() in {"1", "true", "yes", "on"}
    run_name = os.environ.get("RUN_NAME", f"finbench_structured_{BASE_MODEL.lower().replace('/', '-')}_bs{batch}_gs{group}_lr{lr}_{datetime.now():%Y%m%d-%H%M%S}")
    log_path = os.environ.get("LOG_PATH", f"/tmp/tinker-examples/rl_finance/{run_name}")
    if not os.environ.get("FINANCEBENCH_TARGETS"):
        raise ValueError("FINANCEBENCH_TARGETS must point to reviewed typed targets; see docs/WORKSHOP_AGENT_RUNBOOK.md")
    if not remove_constant_reward_groups:
        raise ValueError("Workshop runs require constant-group removal, including unresolved groups")
    if Path(log_path).exists() and any(Path(log_path).iterdir()):
        raise FileExistsError(f"Refusing to overwrite run: {log_path}")
    epochs = int(os.environ.get("EPOCHS", "3"))
    max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
    rank = int(os.environ.get("LORA_RANK", "32"))
    temperature = float(os.environ.get("TEMPERATURE", "1.0"))
    if min(steps, batch, group, epochs, max_turns, max_tokens) < 1:
        raise ValueError("Step, batch, group, epoch and token budgets must be positive")
    if os.environ.get("ZERO_REWARD_ON_LIMIT", "true").lower() not in ("1", "true", "yes", "on"):
        raise ValueError("Validated runs require ZERO_REWARD_ON_LIMIT=true")
    # Label validation occurs before creating a paid training client.
    from run_identity import require_current_judge
    require_current_judge(finance_env.build_judge(finance_env.RewardConfig.from_env()))
    from workshop_prepare import preflight
    fingerprint = preflight(os.environ.get("FINANCEBENCH_SPLIT", str(finance_env.hb.BASE / "split.json")), os.environ["FINANCEBENCH_TARGETS"], corpus=True)
    finance_env.load_financebench(split_name)
    finance_env.load_financebench(dev_split)

    builder = finance_env.FinanceDatasetBuilder(
        model_name_for_tokenizer=BASE_MODEL,
        batch_size=batch,
        group_size=group,
        renderer_name=renderer,
        max_turns=max_turns,
        format_coef=0.0,
        seed=seed,
        split_name=split_name,
        epochs=epochs,
        max_trajectory_tokens=int(os.environ.get("MAX_TRAJECTORY_TOKENS", "32768")),
    )
    manifest = {
        **fingerprint,
        "model": BASE_MODEL,
        "renderer": renderer,
        "split": split_name,
        "steps": steps,
        "step_semantics": "nominal batch cap; actual optimizer updates in optimizer_audit.jsonl",
        "epochs": epochs, "max_tokens": max_tokens, "lora_rank": rank, "temperature": temperature,
        "dev_split": dev_split, "zero_reward_on_limit": True,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dependencies": {p: importlib.metadata.version(p) for p in ("tinker", "tinker-cookbook", "chz")},
        "target_sha256": hashlib.sha256(Path(os.environ["FINANCEBENCH_TARGETS"]).read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(Path(os.environ.get("FINANCEBENCH_SPLIT", str(finance_env.hb.BASE / "split.json"))).read_bytes()).hexdigest(),
        "batch": batch,
        "group": group,
        "learning_rate": lr,
        "max_turns": max_turns,
        "seed": seed,
        "harness": "workshop-v1",
        "reward_contract": {"answer_primary": True, "evidence_grounding": True, "finish_bonus": False, "format_penalty": False, "finish_required": finance_env.RewardConfig.from_env().require_finish},
        "reward_config": finance_env.RewardConfig.from_env().summary(),
        "remove_constant_reward_groups": remove_constant_reward_groups,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, "experiment_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    install_training_audit(train, Path(log_path) / "optimizer_audit.jsonl")
    os.environ["WORKSHOP_GROUP_AUDIT"] = str(Path(log_path) / "group_audit.jsonl")
    evaluator_builders = []
    if int(os.environ.get("EVAL_EVERY", "0")) > 0:
        dev_dataset, _ = await finance_env.FinanceDatasetBuilder(
            model_name_for_tokenizer=BASE_MODEL, batch_size=1, group_size=1,
            renderer_name=renderer, max_turns=max_turns,
            max_trajectory_tokens=int(os.environ.get("MAX_TRAJECTORY_TOKENS", "32768")),
            split_name=dev_split, seed=seed)()
        evaluator_builders = [tinker_evaluator_builder(dev_dataset, max_tokens, dev_split)]
    config = train.Config(
        model_name=BASE_MODEL,
        recipe_name="recipe_financebench_structured",
        renderer_name=renderer,
        log_path=log_path,
        dataset_builder=builder,
        learning_rate=lr,
        max_tokens=max_tokens,
        lora_rank=rank,
        temperature=temperature,
        compute_post_kl=True,
        rollout_error_tolerance=False,
        max_steps=steps,
        remove_constant_reward_groups=remove_constant_reward_groups,
        termination=TerminationRewardPolicy(zero_reward_on_limit=True, skip_grading_on_timeout=True),
        eval_every=int(os.environ.get("EVAL_EVERY", "0")),
        save_every=int(os.environ.get("SAVE_EVERY", "0")),
        evaluator_builders=evaluator_builders,
        rollout_json_export=True,
        enable_trace=True,
        num_groups_to_log=4,
    )
    await train.main(config)
    # Cookbook periodic evaluations precede each update; explicitly evaluate the
    # final saved checkpoint so the last optimizer update is never unmeasured.
    checkpoint_path = Path(log_path) / "checkpoints.jsonl"
    checkpoints = [json.loads(line) for line in checkpoint_path.read_text().splitlines() if line.strip()]
    final = next((c for c in reversed(checkpoints) if c.get("name") == "final"), None)
    if not final or not final.get("sampler_path"):
        raise RuntimeError("Training ended without a final sampler checkpoint; dev evaluation not complete")
    from argparse import Namespace
    from workshop_eval import evaluate
    await evaluate(Namespace(condition="R1", run_id=run_name + "-final-dev", split="dev",
        out=str(Path(log_path) / "final_dev"), model=BASE_MODEL, renderer=renderer,
        checkpoint=final["sampler_path"], training_seed=seed, sampling_seed=None,
        max_turns=max_turns, max_tokens=max_tokens,
        temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.2"))))


if __name__ == "__main__":
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = load_tinker_key()
    asyncio.run(main())
