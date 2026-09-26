"""Zero-RL baseline eval: untrained Qwen3.5-9B-Base + BM25 tool on held-out QA.

Runs a manual agentic loop (no Tinker RL): for each held-out question, the
model searches the SEC filings corpus via a BM25 tool for up to MAX_TURNS turns
and emits a final "Answer:". We grade with the same TextAnswerReward semantics
used in training (correct -> 1.0, else format/answer handling).

Outputs a JSONL of per-question records + a summary to stdout.
"""
import os, sys, json, re, asyncio, time, argparse

sys.path.insert(0, os.path.expanduser("~/Documents/Research/rl-for-ai-research/harness"))
os.environ.setdefault("OPENAI_API_KEY", "")
import finance_env as fe
import financebench_harness as hb

from tinker import ServiceClient, SamplingClient, LoraConfig, SamplingParams, ModelInput
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.recipes.search_tool.tools import TextAnswerReward

BASE = os.path.expanduser("~/Documents/Research/rl-for-ai-research")
MODEL = "Qwen/Qwen3.5-9B-Base"
PROJECT = "5485278b-9573-47cd-816c-9e380e84461f"
MAX_TURNS = 5
N_SAMPLES = 3  # majority-vote-style: answer correct if ANY sample is correct
def _load_key():
    s = open(os.path.expanduser("~/.config/tinker/key")).read().strip()
    # key file may be "tinker_api_key=tinker-..." or a bare key
    if s.startswith("tinker_api_key="):
        s = s.split("=", 1)[1].strip()
    # tinker 0.30.1 validates prefix 'tml-' (new key format); older keys are 'tinker-'
    if s.startswith("tinker-"):
        s = "tml-" + s[len("tinker-"):]
    return s

KEY = _load_key()

# ---- tool-call parsing (Qwen3.5 function-calling format) ----
import re as _re, json as _json

_TC_OPEN  = "\u003ctool_call\u003e"
_TC_CLOSE = "\u003c/tool_call\u003e"
_FN_OPEN  = "\u003cfunction="
_PARA_O   = "\u003cparameter="
_PARA_C   = "\u003c/parameter\u003e"

def _parse_xml_tool_call(text):
    m = _re.search(_re.escape(_TC_OPEN) + r"\s*(.*?)\s*" + _re.escape(_TC_CLOSE), text, _re.S)
    if not m:
        return None
    inner = m.group(1)
    fn = _re.search(_re.escape(_FN_OPEN) + r"([\w.\-]+)", inner)
    if not fn or fn.group(1) != "search":
        return None
    params = {}
    for pm in _re.finditer(_re.escape(_PARA_O) + r"([\w_]+)\s*>(.*?)\s*" + _re.escape(_PARA_C), inner, _re.S):
        params[pm.group(1)] = pm.group(2).strip()
    raw = params.get("query_list") or params.get("queries")
    if raw is None:
        return None
    if raw.strip().startswith("["):
        try:
            return [str(q) for q in _json.loads(raw)]
        except Exception:
            pass
    return [q.strip() for q in _re.split(r"[\n,]", raw) if q.strip()]

def parse_tool_call(text):
    """Return list of query strings if the model emitted a search() call, else None."""
    m = _re.search(_re.escape(_TC_OPEN) + r"\s*(\{.*?\})\s*" + _re.escape(_TC_CLOSE), text, _re.S)
    if m:
        try:
            obj = _json.loads(m.group(1))
            if obj.get("name") == "search":
                ql = obj.get("arguments", {}).get("query_list") or obj.get("arguments", {}).get("queries")
                if isinstance(ql, str):
                    try:
                        ql = _json.loads(ql)
                    except Exception:
                        ql = [ql]
                if isinstance(ql, list):
                    return [str(q) for q in ql]
        except Exception:
            pass
    return _parse_xml_tool_call(text)


def extract_answer(text):
    m = re.search(r"(?im)^answer:\s*(.+)$", text)
    return m.group(1).strip() if m else None

def grade(ans, gold):
    """TextAnswerReward-ish: 1.0 if numeric/clean match, 0.5 partial, else 0.0."""
    if not ans:
        return 0.0
    a, g = ans.lower().strip().strip("\"'."), str(gold).lower().strip().strip("\"'.")
    if a == g:
        return 1.0
    # numeric compare (strip $, commas, units)
    def num(s):
        m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
        if not m:
            return None
        try:
            return float(m.group(0))
        except ValueError:
            return None
    na, ng = num(a), num(g)
    if na is not None and ng is not None and abs(na - ng) <= max(1e-6, abs(ng) * 0.05):
        return 1.0
    if g in a or a in g:
        return 0.5
    return 0.0

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(BASE, "split.json"))
    ap.add_argument("--out", default=os.path.join(BASE, "baseline_eval.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="only first N questions (0=all)")
    args = ap.parse_args()

    split = json.load(open(args.split))
    qa = split["eval"]
    if args.limit:
        qa = qa[: args.limit]
    print(f"eval set: {len(qa)} questions", flush=True)

    # BM25 tool
    doc_names, docs_text, scorer, _passages = hb.build_index()
    tool = fe.Bm25Tool(scorer, docs_text, doc_names, n_results=3)

    # tokenizer + renderer
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer = get_renderer("qwen3_5_disable_thinking", tokenizer)
    schemas = [tool.search.to_spec()]
    prefix = renderer.create_conversation_prefix_with_tools(tools=schemas, system_prompt=fe.FINANCE_TASK_INSTRUCTIONS)

    client = ServiceClient(project_id=PROJECT, api_key=KEY).create_sampling_client(base_model=MODEL)

    reward = TextAnswerReward(gold_answers=["x"], format_coef=0.1)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)

    records = []
    for i, d in enumerate(qa):
        msgs = list(prefix) + [{"role": "user", "content": d["question"]}]
        best = 0.0
        final_ans = None
        for s in range(N_SAMPLES):
            for turn in range(MAX_TURNS):
                mi = renderer.build_generation_prompt(msgs)
                resp = client.sample(prompt=mi, num_samples=1,
                                     sampling_params=sp, include_prompt_logprobs=False)
                if hasattr(resp, "result"):
                    resp = resp.result()
                seq = resp.sequences[0]
                text = tokenizer.decode(list(seq.tokens_np), skip_special_tokens=True)
                tc = parse_tool_call(text)
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
                # final answer turn
                final_ans = extract_answer(text)
                break
            r = reward({"text": final_ans or "", "metadata": {}}) if final_ans else -0.1
            # use our own grade for consistent 0/0.5/1.0
            gold = d["answer"][0] if isinstance(d["answer"], list) else d["answer"]
            gr = grade(final_ans, gold) if final_ans else 0.0
            best = max(best, gr)
            if best == 1.0:
                break
        gold = d["answer"][0] if isinstance(d["answer"], list) else d["answer"]
        records.append({"doc": d.get("doc_name", d.get("doc", "")), "question": d["question"],
                        "gold": gold, "best_reward": best, "final_ans": final_ans})
        print(f"[{i+1}/{len(qa)}] {records[-1]['doc']}: {best:.2f} ({final_ans})", flush=True)
        if i % 10 == 0:
            json.dump(records, open(args.out, "w"), indent=2)

    json.dump(records, open(args.out, "w"), indent=2)
    correct = sum(1 for r in records if r["best_reward"] >= 1.0)
    partial = sum(1 for r in records if 0.5 <= r["best_reward"] < 1.0)
    mean = sum(r["best_reward"] for r in records) / max(1, len(records))
    print(f"\n=== BASELINE SUMMARY (n={len(records)}) ===")
    print(f"correct(exact): {correct}  partial: {partial}  mean_reward: {mean:.3f}")
    print(f"accuracy: {100*correct/len(records):.1f}%")

if __name__ == "__main__":
    asyncio.run(main())
