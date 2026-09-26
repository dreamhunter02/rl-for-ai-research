#!/usr/bin/env python3
"""Minimal local FinanceBench-to-LoRA GRPO bridge trial.

This is intentionally small and explicit. It uses the existing FinanceBench
index/tool implementations for rollout, then computes a one-step, group-relative
policy-gradient update over the reconstructed assistant action tokens. It is a
bridge validation, not a production trainer: the rollout backend is the local
OpenAI-compatible endpoint and the update backend is Unsloth/PEFT.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

# Unsloth must be imported before transformers/trl.
import unsloth  # noqa: F401
import torch
from openai import OpenAI
from unsloth import FastLanguageModel

import financebench_harness as hb

# The container mounts the project at /workspace while the host-side harness
# uses ~/Documents/Research/rl-for-ai-research.
if Path("/workspace/split.json").exists():
    hb.BASE = Path("/workspace")
    hb.FILINGS = hb.BASE / "filings"
    hb.TEXTDIR = hb.BASE / "text"
    hb.DATA = hb.BASE / "data"
    hb.CACHE = hb.BASE / "artifacts" / "page_cache"
SYSTEM = """You are a financial-filings retrieval agent.

Your job is to answer the user's question using the SEC filing corpus. Search
before answering. Start with bm25_search to identify the likely document and
passages, then use grep_document for exact accounting terms or regexes and
read/read_table to inspect bounded evidence. Use calculate only for arithmetic.
Do not invent values or rely on outside knowledge.

When you have enough evidence, call finish(answer, evidence_document, evidence_page) exactly once. Put the complete answer in the answer argument, including units and the requested conclusion. Do not stop after a search hit unless it actually resolves the question.
"""

TOOLS = [
    {"type": "function", "function": {"name": "bm25_search", "description": "Ranked keyword search over prose and tables.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "items": {"type": "string"}}, "company": {"type": "string"}, "year": {"type": "integer"}, "filing_type": {"type": "string"}, "document_id": {"type": "string"}, "scope": {"type": "string", "enum": ["prose", "tables", "both"]}, "top_k": {"type": "integer"}}, "required": ["query_list"]}}},
    {"type": "function", "function": {"name": "grep_document", "description": "Search one filing for exact terms or regexes.", "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}, "patterns": {"type": "array", "items": {"type": "string"}}, "page_start": {"type": "integer"}, "page_end": {"type": "integer"}, "context_lines": {"type": "integer"}, "grep_type": {"type": "string"}}, "required": ["document_id", "patterns"]}}},
    {"type": "function", "function": {"name": "search_tables", "description": "Search table-like filing pages.", "parameters": {"type": "object", "properties": {"query_list": {"type": "array", "items": {"type": "string"}}, "document_id": {"type": "string"}, "company": {"type": "string"}, "year": {"type": "integer"}, "top_k": {"type": "integer"}}, "required": ["query_list"]}}},
    {"type": "function", "function": {"name": "read", "description": "Read a bounded page or passage with provenance.", "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}, "page": {"type": "integer"}, "start": {"type": "integer"}, "end": {"type": "integer"}, "passage_id": {"type": "string"}}, "required": ["document_id"]}}},
    {"type": "function", "function": {"name": "read_table", "description": "Read a table candidate and neighboring context.", "parameters": {"type": "object", "properties": {"table_id": {"type": "string"}, "include_neighbors": {"type": "boolean"}}, "required": ["table_id"]}}},
    {"type": "function", "function": {"name": "calculate", "description": "Evaluate safe basic arithmetic.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "finish", "description": "Submit the final answer and primary evidence provenance.", "parameters": {"type": "object", "properties": {"answer": {"type": "string"}, "evidence_document": {"type": "string"}, "evidence_page": {"type": "integer"}}, "required": ["answer"]}}},
]


def _arg_list(args: dict[str, Any], key: str) -> list[str]:
    value = args.get(key, [])
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        return [value]
    return [str(value)] if value else []


def execute_local(index: hb.StructuredIndex, name: str, args: dict[str, Any]) -> str:
    if name == "bm25_search":
        filters = {"company": args.get("company", ""), "year": args.get("year", -1), "filing_type": args.get("filing_type", ""), "document_id": args.get("document_id", "")}
        top_k = max(1, min(int(args.get("top_k", 3)), 8))
        out: dict[str, Any] = {"queries": _arg_list(args, "query_list"), "filters": filters}
        scope = str(args.get("scope", "both")).lower()
        if scope in ("prose", "both"):
            out["prose_hits"] = index.search_prose(out["queries"], filters, top_k)
        if scope in ("tables", "table", "both"):
            out["table_hits"] = index.search_tables(out["queries"], filters, top_k)
        return json.dumps(out, ensure_ascii=False)
    if name == "grep_document":
        out = index.grep_document(args["document_id"], _arg_list(args, "patterns"), int(args.get("page_start", -1)), int(args.get("page_end", -1)), int(args.get("context_lines", 2)))
        return json.dumps({"matches": out, "backend_used": "page_text"}, ensure_ascii=False)
    if name == "search_tables":
        filters = {"document_id": args.get("document_id", ""), "company": args.get("company", ""), "year": args.get("year", -1)}
        out = index.search_tables(_arg_list(args, "query_list"), filters, max(1, min(int(args.get("top_k", 3)), 8)))
        return json.dumps({"queries": _arg_list(args, "query_list"), "table_hits": out}, ensure_ascii=False)
    if name == "read":
        return json.dumps(index.read(args["document_id"], int(args.get("page", -1)), int(args.get("start", 0)), int(args.get("end", hb.MAX_READ)), args.get("passage_id", "")), ensure_ascii=False)
    if name == "read_table":
        return json.dumps(index.read_table(args["table_id"], bool(args.get("include_neighbors", True))), ensure_ascii=False)
    if name == "calculate":
        try:
            return json.dumps({"expression": args["expression"], "value": hb.calculate(args["expression"])})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    if name == "finish":
        return json.dumps({"finish": True, "answer": str(args.get("answer", "")), "evidence_document": str(args.get("evidence_document", "")), "evidence_page": int(args.get("evidence_page", -1))})
    return json.dumps({"error": f"Unknown tool {name}"})


def load_rows(ids: list[str]) -> list[dict[str, Any]]:
    rows = json.loads((hb.BASE / "split.json").read_text())["train"]
    wanted = set(ids)
    out = []
    for row in rows:
        if row.get("financebench_id") not in wanted or not str(row.get("answer") or "").strip():
            continue
        company = row.get("company") or row.get("doc_name", "").split("_")[0]
        question = str(row["question"]).strip()
        if company.lower() not in question.lower():
            question = f"About {company}: {question}"
        out.append({"financebench_id": row["financebench_id"], "question": question, "gold": str(row["answer"]).strip()})
    missing = wanted - {r["financebench_id"] for r in out}
    if missing:
        raise ValueError(f"ids not found in train split: {sorted(missing)}")
    return out


def rollout(client: OpenAI, index: hb.StructuredIndex, row: dict[str, Any], model_name: str, seed: int, max_turns: int, max_tokens: int, temperature: float) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["question"]}]
    trace: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    answer = ""
    termination = "max_turns"
    finish_metadata: dict[str, Any] = {}
    for turn in range(max_turns):
        choice = "auto" if turn < max_turns - 1 else {"type": "function", "function": {"name": "finish"}}
        response = client.chat.completions.create(model=model_name, messages=messages, tools=TOOLS, tool_choice=choice, max_tokens=max_tokens, temperature=temperature, seed=seed)
        msg = response.choices[0].message
        if not msg.tool_calls:
            answer = msg.content or ""
            trace.append({"role": "assistant", "content": answer})
            termination = "assistant"
            break
        assistant_tool_calls = []
        messages.append({"role": "assistant", "content": msg.content or None, "tool_calls": []})
        tool_items = []
        for tc in msg.tool_calls:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {}
            call = {"name": tc.function.name, "arguments": args, "call_id": tc.id, "turn": turn}
            calls.append(call)
            assistant_tool_calls.append(call)
            observation = execute_local(index, tc.function.name, args)[:4000]
            messages[-1]["tool_calls"].append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": raw}})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})
            tool_items.append({"role": "tool", "call_id": tc.id, "content": observation})
            if tc.function.name == "finish":
                answer = str(args.get("answer", "")).strip()
                page = args.get("evidence_page", -1)
                finish_metadata = {"evidence_document": str(args.get("evidence_document", "")), "evidence_page": int(page) if str(page).lstrip("-").isdigit() else -1}
        trace.append({"role": "assistant", "tool_calls": assistant_tool_calls})
        trace.extend(tool_items)
        if answer:
            termination = "finish"
            break
    answer_reward, answer_parts = hb.score_answer(row["gold"], answer)
    evidence_reward, evidence_parts = hb.score_evidence(row["gold"], answer, trace)
    reward = hb.grounded_reward(answer_reward, evidence_reward)
    return {"financebench_id": row["financebench_id"], "question": row["question"], "gold": row["gold"], "answer_text": answer, "trace": trace, "tool_calls": calls, "termination_reason": termination, "finish_ok": float(termination == "finish"), "finish_metadata": finish_metadata, "answer_reward": answer_reward, "evidence_reward": evidence_reward, "reward": reward, "reward_parts": {**answer_parts, **evidence_parts}, "seed": seed}


def action_sequence(record: dict[str, Any], observation_cap: int = 1600) -> tuple[str, list[tuple[int, int]]]:
    """Serialize one complete stateful trajectory and mark assistant actions."""
    text = f"SYSTEM:\n{SYSTEM}\nUSER:\n{record['question']}\n"
    spans: list[tuple[int, int]] = []
    for item in record["trace"]:
        role = item.get("role")
        if role == "assistant":
            if item.get("tool_calls"):
                action = "ASSISTANT_TOOL:\n" + json.dumps(item["tool_calls"], ensure_ascii=False, sort_keys=True) + "\n"
            else:
                action = "ASSISTANT_FINAL:\n" + str(item.get("content", "")) + "\n"
            start = len(text)
            text += action
            spans.append((start, len(text)))
        elif role == "tool":
            text += "TOOL_OUTPUT:\n" + str(item.get("content", ""))[:observation_cap] + "\n"
    return text, spans


def sequence_logprob(model: torch.nn.Module, processor: Any, text: str, spans: list[tuple[int, int]], max_length: int) -> tuple[torch.Tensor, int]:
    tokenizer = processor.tokenizer
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True, truncation=True, max_length=max_length)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    device = next(p for p in model.parameters() if p.requires_grad).device
    model_inputs = {k: v.to(device) for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
    outputs = model(**model_inputs)
    logits = outputs.logits[:, :-1, :]
    labels = model_inputs["input_ids"][:, 1:]
    logps = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)[0]
    selected: list[int] = []
    for token_index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        if any(end > span_start and start < span_end for span_start, span_end in spans):
            if token_index > 0 and token_index - 1 < logps.shape[0]:
                selected.append(token_index - 1)
    if not selected:
        raise RuntimeError("trajectory action tokens were truncated or not tokenized")
    indices = torch.tensor(selected, device=logps.device, dtype=torch.long)
    return logps.index_select(0, indices).sum(), len(selected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/hf/models--Qwen--Qwen3.5-4B")
    ap.add_argument("--endpoint", default="http://spark-a16b.local:18360/v1")
    ap.add_argument("--endpoint-model", default="Qwen3.5-4B")
    ap.add_argument("--ids", default="financebench_id_01226,financebench_id_01936")
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-seq-length", type=int, default=16384)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--learning-rate", type=float, default=1e-6)
    ap.add_argument("--output-dir", default="/workspace/results/financebench_grpo_trial_qwen35_4b_r32_16k")
    ap.add_argument("--rollouts-file", default="", help="Use already-generated harness trajectories and skip endpoint/corpus rollout.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    started = time.time()
    if args.rollouts_file:
        records = [json.loads(line) for line in Path(args.rollouts_file).read_text().splitlines() if line.strip()]
        ids = sorted({r["financebench_id"] for r in records})
        print(json.dumps({"stage": "rollouts_loaded", "file": args.rollouts_file, "questions": ids, "records": len(records)}), flush=True)
    else:
        rows = load_rows(ids)
        client = OpenAI(base_url=args.endpoint, api_key="none")
        print(json.dumps({"stage": "rollout_start", "questions": ids, "group_size": args.group_size}), flush=True)
        index = hb.build_index()
        records = []
        for row_index, row in enumerate(rows):
            for generation in range(args.group_size):
                rec = rollout(client, index, row, args.endpoint_model, args.seed + row_index * 100 + generation, args.max_turns, args.max_tokens, args.temperature)
                records.append(rec)
                print(json.dumps({"stage": "rollout", "id": rec["financebench_id"], "generation": generation, "reward": rec["reward"], "answer_reward": rec["answer_reward"], "evidence_reward": rec["evidence_reward"], "termination": rec["termination_reason"], "tools": len(rec["tool_calls"])}), flush=True)
    (out / "rollouts.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["financebench_id"], []).append(record)
    advantages: dict[int, float] = {}
    group_stats = {}
    for fid, group in groups.items():
        rewards = [float(r["reward"]) for r in group]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
        denom = std if std > 1e-6 else 1.0
        for r, reward_value in zip(group, rewards):
            advantages[id(r)] = (reward_value - mean) / denom
        group_stats[fid] = {"rewards": rewards, "mean": mean, "std": std}
    if not any(stat["std"] > 1e-6 for stat in group_stats.values()):
        raise RuntimeError(f"all rollout groups have zero reward variance: {group_stats}")

    print(json.dumps({"stage": "model_load", "model": args.model}), flush=True)
    model, processor = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=args.max_seq_length, load_in_4bit=False, load_in_16bit=True, full_finetuning=False, fast_inference=False)
    model = FastLanguageModel.get_peft_model(model, r=args.lora_rank, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_alpha=args.lora_rank, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407, max_seq_length=args.max_seq_length)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)

    before = sum(float(p.detach().float().abs().sum()) for p in model.parameters() if p.requires_grad)
    valid = 0
    total_loss = 0.0
    action_tokens = 0
    for record in records:
        advantage = advantages[id(record)]
        sequence, spans = action_sequence(record)
        logprob, count = sequence_logprob(model, processor, sequence, spans, args.max_seq_length)
        contribution = -float(advantage) * logprob / max(1, count) / len(records)
        contribution.backward()
        total_loss += float(contribution.detach().cpu())
        action_tokens += count
        valid += 1
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = sum(float(p.detach().float().abs().sum()) for p in model.parameters() if p.requires_grad)
    model.save_pretrained(out / "adapter-step-1")
    processor.save_pretrained(out / "adapter-step-1")

    metrics = {"stage": "optimizer_step_complete", "questions": ids, "records": len(records), "valid_action_sequences": valid, "group_stats": group_stats, "mean_reward": sum(r["reward"] for r in records) / len(records), "mean_answer_reward": sum(r["answer_reward"] for r in records) / len(records), "mean_evidence_reward": sum(r["evidence_reward"] for r in records) / len(records), "loss_sum": total_loss, "action_tokens": action_tokens, "grad_norm": float(grad_norm), "trainable_checksum_before": before, "trainable_checksum_after": after, "parameter_delta_l1": abs(after - before), "learning_rate": args.learning_rate, "lora_rank": args.lora_rank, "max_seq_length": args.max_seq_length, "elapsed_s": time.time() - started, "adapter": str(out / "adapter-step-1")}
    (out / "trial_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
