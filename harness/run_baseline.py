"""Corrected zero-RL baseline eval.

Fixes two bugs in baseline_eval.py WITHOUT touching its verified parser/grader:
  (1) Prompt assembly: baseline_eval.py sends only `mi.chunks[0]` (a bare
      'system\n' header) to client.sample, so the model never sees the real
      prompt and regurgitates the tool-schema. Here we send the FULL prompt
      (all chunks).
  (2) Context guard: with the real prompt, the agentic trajectory can exceed
      the model's 65536-token window -> BadRequestError. We cap the prompt
      at CONTEXT_BUDGET - max_tokens by dropping the OLDEST assistant/tool
      pairs (keep system + most recent context) and grade normally.

Semantics otherwise match baseline_eval.py: same split, BM25 tool, N_SAMPLES,
MAX_TURNS, grade(), TextAnswerReward, summary math. Streams one JSON record
per line to --out (survives a crash); --start resumes past completed questions.
"""
import os, sys, json, argparse, time

sys.path.insert(0, os.path.expanduser("~/Documents/Research/rl-for-ai-research/harness"))
os.environ.setdefault("OPENAI_API_KEY", "")
import finance_env as fe
import financebench_harness as hb

from tinker import ServiceClient, SamplingParams, ModelInput
from tinker_cookbook import tokenizer_utils
from tinker_cookbook.renderers import get_renderer

BASE = os.path.expanduser("~/Documents/Research/rl-for-ai-research")
MODEL = "Qwen/Qwen3.5-9B-Base"
MODEL_4B = "Qwen/Qwen3.5-4B"
PROJECT = "5485278b-9573-47cd-816c-9e380e84461f"
MAX_TURNS = 5
N_SAMPLES = 3
CONTEXT_BUDGET = 65536          # model context window (from BadRequestError detail)
MAX_TOKENS = 1024               # generation budget (== baseline_eval.py sp)
PROMPT_CAP = CONTEXT_BUDGET - MAX_TOKENS  # max prompt tokens we will send

# Reuse the VERIFIED parser + answer extraction + grader verbatim from baseline_eval.py
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "baseline_eval", os.path.join(os.path.dirname(__file__), "baseline_eval.py"))
_be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_be)  # does NOT run main() (guarded by __main__)
parse_tool_call = _be.parse_tool_call
extract_answer = _be.extract_answer
grade = _be.grade


def _load_key():
    s = open(os.path.expanduser("~/.config/tinker/key")).read().strip()
    if s.startswith("tinker_api_key="):
        s = s.split("=", 1)[1].strip()
    if s.startswith("tinker-"):
        s = "tml-" + s[len("tinker-"):]
    return s


def build_prompt_tokens(renderer, msgs, cap):
    """Render msgs -> tokens; if over `cap`, drop oldest assistant+tool pairs."""
    mi = renderer.build_generation_prompt(msgs)
    toks = [t for ch in mi.chunks for t in ch.tokens] if mi.chunks else []
    if len(toks) <= cap:
        return toks
    # msgs: [system, user, (assistant, tool)*...]. Drop from the start of the
    # middle (after the initial user), oldest pairs first.
    m = list(msgs)
    while len(toks) > cap and len(m) > 2:
        # find first index > 0 that is an assistant message; drop it + its tool reply
        idx = None
        for j in range(1, len(m)):
            if m[j]["role"] == "assistant":
                idx = j
                break
        if idx is None:
            m.pop(1)
        else:
            m.pop(idx)
            if idx + 1 < len(m) and m[idx + 1]["role"] == "tool":
                m.pop(idx + 1)
        mi = renderer.build_generation_prompt(m)
        toks = [t for ch in mi.chunks for t in ch.tokens] if mi.chunks else []
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(BASE, "split.json"))
    ap.add_argument("--out", default=os.path.join(BASE, "baseline_eval.jsonl"))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only first N after --start (0=all)")
    ap.add_argument("--model", default=MODEL,
                    help=f"model name on Tinker (default: {MODEL}; use {MODEL_4B} for the instruct 4B)")
    args = ap.parse_args()
    model_name = args.model

    split = json.load(open(args.split))
    qa = split["eval"]
    if args.limit:
        qa = qa[args.start: args.start + args.limit]
    print(f"eval set: {len(qa)} questions (start={args.start})", flush=True)

    doc_names, docs_text, scorer, _passages = hb.build_index()
    tool = fe.Bm25Tool(scorer, docs_text, doc_names, n_results=3)
    tokenizer = tokenizer_utils.get_tokenizer(model_name)
    renderer = get_renderer("qwen3_5_disable_thinking", tokenizer)
    schemas = [tool.search.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(tools=schemas, system_prompt=fe.FINANCE_TASK_INSTRUCTIONS)
    client = ServiceClient(project_id=PROJECT, api_key=_load_key()).create_sampling_client(base_model=model_name)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=MAX_TOKENS)

    outfh = open(args.out, "a")
    dbg = open(args.out + ".dbg", "a")
    t_start = time.time()
    n_total = len(split["eval"])
    for j, d in enumerate(qa):
        i = args.start + j  # absolute question index
        msgs = list(prefix) + [{"role": "user", "content": d["question"]}]
        best = 0.0
        final_ans = None
        for s in range(N_SAMPLES):
            for turn in range(MAX_TURNS):
                tokens = build_prompt_tokens(renderer, msgs, PROMPT_CAP)
                dbg.write(f"Q{i} s{s} turn{turn} prompt={len(tokens)}\n"); dbg.flush()
                resp = client.sample(prompt=ModelInput.from_ints(tokens), num_samples=1,
                                     sampling_params=sp, include_prompt_logprobs=False).result()
                seq = resp.sequences[0]
                text = tokenizer.decode(list(seq.tokens_np), skip_special_tokens=True)
                tc = parse_tool_call(text)
                dbg.write(f"   -> {'TOOL '+str(tc) if tc else 'FINAL '+repr(final_ans if False else (extract_answer(text)))[:80]} len={len(text)}\n"); dbg.flush()
                if tc is not None:
                    content = ""
                    for q in tc:
                        hits = scorer(q, 3)
                        content += f"Query: {q}\n"
                        if not hits:
                            content += "  (no matches)\n"
                        for j, (doc, st, en, _t) in enumerate(hits):
                            content += f"Document {j+1} [{doc} chars {st}-{en}]:\n"
                            content += f"{docs_text[doc][st:en]}\n"
                        content += "\n"
                    msgs.append({"role": "assistant", "content": text})
                    msgs.append({"role": "tool", "content": content})
                    continue
                final_ans = extract_answer(text)
                break
            gold = d["answer"][0] if isinstance(d["answer"], list) else d["answer"]
            gr = grade(final_ans, gold) if final_ans else 0.0
            best = max(best, gr)
            if best == 1.0:
                break
        gold = d["answer"][0] if isinstance(d["answer"], list) else d["answer"]
        rec = {"doc": d.get("doc_name", d.get("doc", "")), "question": d["question"],
               "gold": gold, "best_reward": best, "final_ans": final_ans}
        outfh.write(json.dumps(rec) + "\n")
        outfh.flush()
        print(f"[{i+1}/{n_total}] {rec['doc']}: {best:.2f} ({final_ans})", flush=True)

    outfh.close()
    records = [json.loads(l) for l in open(args.out) if l.strip()]
    correct = sum(1 for r in records if r["best_reward"] >= 1.0)
    partial = sum(1 for r in records if 0.5 <= r["best_reward"] < 1.0)
    mean = sum(r["best_reward"] for r in records) / max(1, len(records))
    print(f"\n=== BASELINE SUMMARY (n={len(records)}) ===")
    print(f"correct(exact): {correct}  partial: {partial}  mean_reward: {mean:.3f}")
    print(f"accuracy: {100*correct/len(records):.1f}%")
    print(f"(elapsed {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
