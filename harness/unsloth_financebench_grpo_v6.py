#!/usr/bin/env python3
"""One bounded, on-policy Unsloth GRPO update from the FinanceBench SFT adapter.

The current calculator-free tool session supplies receipts. Each whole trajectory
gets the same v6 additive reward used by the saved-trace audit. Unresolved judge
outputs are excluded; a flat group never produces a fake optimizer update.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
from pathlib import Path

import unsloth  # noqa: F401; import before transformers
import torch
from unsloth import FastLanguageModel

import financebench_harness as hb
from component_reward import ComponentJudge
from live_additive_reward import LiveAdditiveJudge
from teacher_runtime import TOOL_NAMES, TeacherHarnessSession, submission_answer_text
from financebench_onpolicy_grpo import action_logprobs, parse_tool_calls
from generate_teacher_traces import _lfm_text_tool_calls


def load_train_rows(split_path: Path, ids: list[str]) -> list[dict]:
    split = json.loads(split_path.read_text())
    train = {r["financebench_id"]: r for r in split["train"]}
    if len(ids) != len(set(ids)) or any(i not in train for i in ids):
        raise ValueError("IDs must be unique and belong to frozen train96")
    rows = []
    for question_id in ids:
        row = train[question_id]
        company = row.get("company") or str(row.get("doc_name", "")).split("_")[0]
        question = str(row["question"]).strip()
        if company and company.lower() not in question.lower():
            question = f"About {company}: {question}"
        rows.append(dict(financebench_id=question_id, question=question,
                         gold=str(row["answer"]), doc_name=str(row["doc_name"])))
    return rows


def tool_schemas(dataset_path: Path) -> list[dict]:
    with dataset_path.open() as fh:
        row = json.loads(next(fh))
    tools = row["tools"]
    if tuple(t["function"]["name"] for t in tools) != TOOL_NAMES:
        raise ValueError("SFT tool schema differs from active calculator-free harness")
    return tools


def prompt_ids(tokenizer, messages, tools, max_seq_length, max_new_tokens):
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    encoded = inner.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                        tokenize=True, return_tensors="pt")
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.shape[1] + max_new_tokens > max_seq_length:
        raise RuntimeError(f"context would exceed {max_seq_length} tokens")
    return ids


def rollout(model, tokenizer, index, tools, row, seed, args):
    torch.manual_seed(seed)
    random.seed(seed)
    session = TeacherHarnessSession(index, max_turns=args.max_turns)
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    messages = [{"role": "system", "content": session.system_prompt},
                {"role": "user", "content": row["question"]}]
    trace, action_records = [], []
    termination = "max_turns"
    model.eval()
    for turn in range(args.max_turns):
        session.start_turn()
        try:
            ids = prompt_ids(tokenizer, messages, tools, args.max_seq_length, args.max_new_tokens)
        except RuntimeError as exc:
            if not str(exc).startswith("context would exceed "):
                raise
            termination = "context_limit"
            print(json.dumps({"stage": "context_limit", "id": row["financebench_id"],
                              "seed": seed, "turn": turn + 1}), flush=True)
            break
        print(json.dumps({"stage": "turn_start", "id": row["financebench_id"],
                          "seed": seed, "turn": turn + 1, "prompt_tokens": int(ids.shape[1])}), flush=True)
        ids = ids.to(next(model.parameters()).device)
        with torch.no_grad():
            generated = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                       max_new_tokens=args.max_new_tokens, do_sample=True,
                                       temperature=args.temperature, top_p=args.top_p,
                                       pad_token_id=inner.eos_token_id)
        completion = generated[0, ids.shape[1]:].detach().cpu()
        text = inner.decode(completion.tolist(), skip_special_tokens=False)
        calls = parse_tool_calls(text)
        if not calls:
            calls = [dict(name=c["function"]["name"],
                          arguments=json.loads(c["function"]["arguments"]))
                     for c in _lfm_text_tool_calls(text, 0)]
        action_records.append(dict(prompt_token_ids=ids[0].detach().cpu().tolist(),
                                   token_ids=completion.tolist(), text=text))
        print(json.dumps({"stage": "turn_generated", "id": row["financebench_id"],
                          "seed": seed, "turn": turn + 1, "tokens": int(completion.numel()),
                          "tool_names": [c["name"] for c in calls]}), flush=True)
        if not completion.numel():
            termination = "empty_generation"
            break
        if not calls:
            trace.append({"role": "assistant", "content": text})
            termination = "assistant_without_tool"
            break
        openai_calls = []
        for n, call in enumerate(calls):
            call_id = f"g{seed}-t{turn}-c{n}"
            openai_calls.append(dict(id=call_id, type="function",
                                     function=dict(name=call["name"], arguments=call["arguments"])))
        messages.append({"role": "assistant", "content": None, "tool_calls": openai_calls})
        trace.append({"role": "assistant", "content": text, "tool_calls": calls})
        for call, named in zip(calls, openai_calls):
            result = session.execute(call["name"], call["arguments"], named["id"])
            messages.append({"role": "tool", "tool_call_id": named["id"], "content": result.content})
            trace.append({"role": "tool", "tool_call_id": named["id"], "content": result.content})
            if result.should_stop:
                termination = "finish" if session.accepted is not None else "terminal_rejected"
                break
        if termination in ("finish", "terminal_rejected"):
            break
    return dict(financebench_id=row["financebench_id"], question=row["question"], gold=row["gold"],
                answer_text=submission_answer_text(session.accepted), submission=session.accepted,
                receipts=session.state.receipts, trace=trace, action_records=action_records,
                termination_reason=termination, seed=seed)


def group_advantages(records):
    groups = {}
    for record in records:
        groups.setdefault(record["financebench_id"], []).append(record)
    retained = 0
    for group in groups.values():
        valid = [r for r in group if r["score"]["reward"] is not None]
        if len(valid) < 2:
            for record in group:
                record["advantage"] = None
            continue
        mean = sum(r["score"]["reward"] for r in valid) / len(valid)
        std = math.sqrt(sum((r["score"]["reward"] - mean) ** 2 for r in valid) / len(valid))
        for record in group:
            record["advantage"] = ((record["score"]["reward"] - mean) / std
                                   if record in valid and std > 1e-8 else None)
        retained += int(std > 1e-8)
    return retained


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to saved SFT adapter")
    ap.add_argument("--split", required=True)
    ap.add_argument("--teacher-dataset", required=True, help="Schema source only; not training examples")
    ap.add_argument("--ids", default="financebench_id_04672,financebench_id_01865")
    ap.add_argument("--all-train-corpus", action="store_true",
                    help="Index every filing named by frozen train96, not just this batch")
    ap.add_argument("--full-corpus", action="store_true",
                    help="Index the complete configured filing corpus")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rollouts-from", default="", help="Resume exact saved on-policy rollouts")
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=101)
    args = ap.parse_args()
    if args.group_size < 2 or args.max_turns < 2 or args.max_new_tokens < 32:
        raise ValueError("invalid smoke run bounds")
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    rows = load_train_rows(Path(args.split), ids)
    tools = tool_schemas(Path(args.teacher_dataset))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "split.sha256").write_text(hashlib.sha256(Path(args.split).read_bytes()).hexdigest())
    if args.full_corpus and args.all_train_corpus:
        raise ValueError("choose only one corpus scope")
    corpus_rows = json.loads(Path(args.split).read_text())["train"] if args.all_train_corpus else rows
    corpus_docs = sorted({row["doc_name"] for row in corpus_rows})
    if args.rollouts_from:
        previous = Path(args.rollouts_from)
        prior_config = json.loads((previous.parent / "config.json").read_text())
        for key in ("model", "split", "teacher_dataset", "ids", "group_size", "max_turns",
                    "max_new_tokens", "max_seq_length", "temperature", "top_p", "seed"):
            if prior_config[key] != getattr(args, key):
                raise ValueError(f"saved rollout protocol mismatch: {key}")
        if (previous.parent / "split.sha256").read_text() != (out / "split.sha256").read_text():
            raise ValueError("saved rollout split hash mismatch")
        records = [json.loads(line) for line in previous.read_text().splitlines() if line.strip()]
        if len(records) != len(rows) * args.group_size or any(
            sum(r["financebench_id"] == row["financebench_id"] for r in records) != args.group_size
            for row in rows
        ):
            raise ValueError("incomplete saved rollout group")
        (out / "rollouts.jsonl").write_bytes(previous.read_bytes())
        print(json.dumps({"stage": "resume_rollouts", "records": len(records),
                          "source_sha256": hashlib.sha256(previous.read_bytes()).hexdigest()}), flush=True)
        index = None
    else:
        print(json.dumps({"stage": "index", "questions": ids,
                          "corpus_scope": "full" if args.full_corpus else "selected",
                          "diagnostic_corpus_docs": None if args.full_corpus else corpus_docs}), flush=True)
        index = hb.build_index() if args.full_corpus else hb.build_index(doc_names=corpus_docs)
    print(json.dumps({"stage": "model_load", "adapter": args.model}), flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.model,
        max_seq_length=args.max_seq_length, load_in_4bit=True, load_in_16bit=False,
        full_finetuning=False, fast_inference=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("SFT adapter loaded frozen; refusing optimizer run")
    if not args.rollouts_from:
        judge = LiveAdditiveJudge(ComponentJudge())
        records = []
        with (out / "rollouts.jsonl").open("w") as fh:
            for n, row in enumerate(rows):
                asyncio.run(judge.rubric(row["financebench_id"], row["question"], row["gold"]))
                for generation in range(args.group_size):
                    record = rollout(model, tokenizer, index, tools, row,
                                     args.seed + n * 1000 + generation, args)
                    score = judge.score_sync(row["financebench_id"], row["question"], row["gold"],
                                             record["submission"], record["receipts"])
                    record["score"] = score
                    if score["unresolved"]:
                        print(json.dumps({"stage": "unresolved", "id": row["financebench_id"],
                                          "reason": score.get("reason")}), flush=True)
                    else:
                        print(json.dumps({"stage": "rollout", "id": row["financebench_id"],
                                          "generation": generation, "reward": score["reward"],
                                          "F": score["F"], "E": score["E"], "A": score["A"]}), flush=True)
                    records.append(record)
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
    retained = group_advantages(records)
    if not retained:
        (out / "summary.json").write_text(json.dumps(dict(optimizer_steps=0,
            retained_groups=0, records=len(records), reason="no within-question reward variance"), indent=2))
        print(json.dumps({"stage": "signal_gate_failed", "retained_groups": 0}), flush=True)
        return
    for record in records:
        if record["advantage"] is None:
            continue
        with torch.no_grad():
            for action in record["action_records"]:
                old = action_logprobs(model, action["prompt_token_ids"], action["token_ids"], args.max_seq_length)
                action["old_logprobs"] = old.detach().cpu().tolist()
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    terms = sum(len(r["action_records"]) for r in records if r["advantage"] is not None)
    losses = []
    for record in records:
        if record["advantage"] is None:
            continue
        for action in record["action_records"]:
            new = action_logprobs(model, action["prompt_token_ids"], action["token_ids"], args.max_seq_length)
            old = torch.tensor(action["old_logprobs"], device=new.device, dtype=new.dtype)
            if not len(new) or len(new) != len(old):
                raise RuntimeError("generated-token log probability mismatch")
            ratio = torch.exp(new - old.detach())
            advantage = record["advantage"]
            loss = -torch.minimum(ratio * advantage,
                torch.clamp(ratio, 0.8, 1.2) * advantage).mean() / terms
            loss.backward()
            losses.append(float(loss.detach().cpu()))
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    if not math.isfinite(float(norm)):
        raise RuntimeError("nonfinite gradient")
    optimizer.step()
    model.save_pretrained(out / "adapter-step-1")
    tokenizer.save_pretrained(out / "adapter-step-1")
    summary = dict(optimizer_steps=1, retained_groups=retained, records=len(records),
                   trainable_parameters=trainable, loss=sum(losses), grad_norm=float(norm),
                   adapter=str(out / "adapter-step-1"), scorer_version="workshop-rubric-v6-additive")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"stage": "complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
