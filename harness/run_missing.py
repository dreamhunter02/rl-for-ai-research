"""Fill in specific q_idx (position in split['eval']) missing from baseline_eval.jsonl."""
import os, sys, json, time
sys.path.insert(0, os.path.expanduser("~/Documents/Research/rl-for-ai-research/harness"))
import run_baseline as rb

IDX = [3, 4, 5, 9, 10, 35]

def main():
    out = os.path.expanduser("~/Documents/Research/rl-for-ai-research/baseline_eval.jsonl")
    split = json.load(open(os.path.join(rb.BASE, "split.json")))
    qa = split["eval"]
    model_name = rb.MODEL

    import financebench_harness as hb, finance_env as fe
    from tinker_cookbook import tokenizer_utils
    from tinker_cookbook.renderers import get_renderer
    from tinker import ServiceClient, SamplingParams, ModelInput
    doc_names, docs_text, scorer, _p = hb.build_index()
    tool = fe.Bm25Tool(scorer, docs_text, doc_names, n_results=3)
    tok = tokenizer_utils.get_tokenizer(model_name)
    renderer = get_renderer("qwen3_5_disable_thinking", tok)
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[tool.search.to_spec()], system_prompt=fe.FINANCE_TASK_INSTRUCTIONS)
    client = ServiceClient(project_id=rb.PROJECT, api_key=rb._load_key()).create_sampling_client(base_model=model_name)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=rb.MAX_TOKENS)
    parse_tool_call = rb.parse_tool_call
    extract_answer = rb.extract_answer
    grade = rb.grade

    t0 = time.time()
    with open(out, "a") as outfh:
        for i in IDX:
            d = qa[i]
            gold = d["answer"][0] if isinstance(d["answer"], list) else d["answer"]
            msgs = list(prefix) + [{"role": "user", "content": d["question"]}]
            best = 0.0
            final_ans = None
            for s in range(rb.N_SAMPLES):
                for turn in range(rb.MAX_TURNS):
                    tokens = rb.build_prompt_tokens(renderer, msgs, rb.PROMPT_CAP)
                    resp = client.sample(prompt=ModelInput.from_ints(tokens), num_samples=1,
                                         sampling_params=sp, include_prompt_logprobs=False).result()
                    seq = resp.sequences[0]
                    text = tok.decode(list(seq.tokens_np), skip_special_tokens=True)
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
                    final_ans = extract_answer(text)
                    gr = grade(final_ans, gold) if final_ans else 0.0
                    best = max(best, gr)
                    break
                if best == 1.0:
                    break
            rec = {"doc": d.get("doc_name", d.get("doc", "")), "question": d["question"],
                   "gold": gold, "best_reward": best, "final_ans": final_ans}
            outfh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            outfh.flush()
            print(f"[q{i}] {rec['doc']}: {best:.2f} ({final_ans})", flush=True)
    print(f"(elapsed {time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
