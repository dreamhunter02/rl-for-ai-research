"""Build a standalone, lossless before/after trace report from saved dev12 results."""
import json
from pathlib import Path
from make_trace_viewer import esc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/paper_2026_rl4llm'

def load(name):
    rows = [json.loads(line) for line in (RESULTS / name).read_text().splitlines() if line.strip()]
    assert len(rows) == len({r['financebench_id'] for r in rows}) == 12
    return {r['financebench_id']: r for r in rows}

def pretty(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return value
    return json.dumps(value, ensure_ascii=False, indent=2)

def block(value):
    return '<pre>' + esc(pretty(value)) + '</pre>'

def panel(row, label):
    score = row['score']
    status = 'grounded' if score.get('grounded_success') else 'correct' if score.get('correct') else 'unsuccessful'
    out = [f'<section class="panel"><h3>{esc(label)} <span class="badge {status}">{status}</span></h3>',
           f'<p>F {score.get("F", 0)} · A {score.get("A", 0)} · G {score.get("G", 0)} · Reward {score.get("reward", 0):.2f}</p>',
           f'<p class="meta">{len(row.get("tool_calls", []))} tool calls · {esc(row.get("wall_s", "unknown"))} seconds · Stop: {esc(row.get("termination_reason"))}</p>',
           '<h4>Final answer</h4>' + block(row.get('answer_text', '')),
           '<details><summary>Typed submission & citations</summary>' + block(row.get('submission')) + '</details>',
           '<details><summary>Full scorer output (including unresolved status)</summary>' + block(score) + '</details>',
           '<h4>Complete chronological trace</h4>']
    for i, message in enumerate(row.get('trace', []), 1):
        role = message.get('role', 'unknown')
        calls = message.get('tool_calls', [])
        names = ', '.join(c.get('name', c.get('function', {}).get('name', '?')) for c in calls)
        title = f'{i:02d} · {role}' + (f' → {names}' if names else '')
        out.append(f'<details class="message {esc(role)}"><summary>{esc(title)}</summary>{block(message)}</details>')
    out.append('<details><summary>Full original record</summary>' + block(row) + '</details></section>')
    return ''.join(out)

def main():
    before = load('qwen35_4b_current_harness_baseline_dev12_v2.jsonl')
    after = load('qwen35_4b_qlora_dev12_20260930.jsonl')
    frozen = json.loads((ROOT / 'split.json').read_text())
    assert set(before) == set(after) == {r['financebench_id'] for r in frozen['dev']}
    table, cards = [], []
    for fid, base in before.items():
        sft = after[fid]
        assert base['question'] == sft['question']
        change = sft['score'].get('correct', 0) - base['score'].get('correct', 0)
        verdict = 'Improved' if change > 0 else 'Regressed' if change < 0 else 'Unchanged'
        cells = ''.join(f'<td>{r["score"].get(k, 0)}</td>' for k in ('correct', 'grounded_success', 'F') for r in (base, sft))
        table.append(f'<tr><td><a href="#{fid}">{esc(fid.removeprefix("financebench_id_"))}</a></td>{cells}<td>{verdict}</td></tr>')
        cards.append(f'<article id="{fid}" data-search="{esc(fid + " " + base["question"])}"><h2>{esc(fid)} · {verdict}</h2><p class="question">{esc(base["question"])}</p><div class="gold"><b>Reference answer:</b> {esc(base.get("gold"))}</div><div class="columns">{panel(base,"Before · Base Qwen3.5-4B")}{panel(sft,"After · QLoRA SFT")}</div></article>')
    summary = []
    for label, key in [('Correct + finished', 'correct'), ('Correct + grounded', 'grounded_success'), ('Valid finish', 'F')]:
        counts = [int(sum(r['score'].get(key, 0) for r in group.values())) for group in (before, after)]
        summary.append(f'<div class="metric">{label}<strong>{counts[0]}/12 → {counts[1]}/12</strong></div>')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>FinanceBench dev12 · Base vs SFT traces</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#182638;font:15px/1.5 system-ui,sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1{font-size:30px;margin:0}h2{font-size:21px}h3{font-size:17px}h4{margin-bottom:8px}a{color:#175ca0}.meta{color:#526478;font-size:13px}.metrics,.columns{display:grid;gap:18px}.metrics{grid-template-columns:repeat(3,1fr);margin:20px 0}.metric,.panel,article,nav{background:white;border:1px solid #d9e2ed;border-radius:10px;padding:18px}.metric strong{display:block;font-size:28px}.columns{grid-template-columns:1fr 1fr}.panel{min-width:0;background:#fafcff}article{margin-top:24px;scroll-margin-top:15px}.question{font-size:18px}.gold{background:#fff3d9;padding:12px;margin-bottom:18px;border-radius:6px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 ui-monospace,monospace;max-height:560px;overflow:auto;padding:12px;background:#edf2f8;border-radius:6px}summary{cursor:pointer;padding:9px;font-weight:600}details{border:1px solid #dce4ee;border-radius:5px;margin:8px 0}.assistant{border-left:4px solid #397dba}.tool{border-left:4px solid #74869a}.badge{font-size:11px;padding:4px 8px;border-radius:20px;display:inline-block}.grounded{background:#d7f3e6;color:#135c38}.correct{background:#fff0c7;color:#704c00}.unsuccessful{background:#fce2e2;color:#8a2525}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid #e2e8f0}button,input{font:inherit;padding:9px 12px;border:1px solid #b9c9da;border-radius:5px;margin:5px}button{cursor:pointer;background:white}input{min-width:260px}.table-wrap{overflow:auto}.note{border-left:4px solid #d1a337;padding:12px;background:#fff8e6}footer{padding:25px;color:#526478}@media(max-width:850px){.columns,.metrics{grid-template-columns:1fr}main{padding:12px}}@media print{pre{max-height:none}button,input{display:none}.columns{grid-template-columns:1fr}article{break-before:page}}
    </style></head><body><main><h1>FinanceBench dev12 · Before & after SFT</h1><p>Qwen3.5-4B → 4-bit QLoRA SFT · 12 paired questions · September 30, 2026</p>'''
    page += '<div class="metrics">' + ''.join(summary) + '</div>'
    page += '<p class="note">Provisional comparison: the historical baseline has no complete launch manifest. SFT evaluation used 8 turns, 1,024 output tokens per call, temperature 0, seed 0. One unresolved grading case in each run; neither is counted as successful. Training used 4-bit weights; evaluation used a BF16 base plus the adapter. This is development data, not final benchmark evaluation.</p>'
    page += '<nav><input id="search" aria-label="Filter questions" placeholder="Find question or ID…"><button id="expand">Expand all trace details</button><button id="collapse">Collapse all details</button><div class="table-wrap"><table><thead><tr><th>Question</th><th>Correct before</th><th>Correct after</th><th>Grounded before</th><th>Grounded after</th><th>Finish before</th><th>Finish after</th><th>Correctness change</th></tr></thead><tbody>' + ''.join(table) + '</tbody></table></div></nav>'
    page += ''.join(cards) + '<footer>Self-contained offline report. All recorded messages, tool arguments, observations, submissions and scorer fields are included without truncation. Events are chronological within each run; matching row numbers do not imply matching actions.</footer></main><script>document.getElementById("search").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("article").forEach(a=>a.hidden=!a.dataset.search.toLowerCase().includes(q))});document.getElementById("expand").onclick=()=>document.querySelectorAll("article:not([hidden]) details").forEach(d=>d.open=true);document.getElementById("collapse").onclick=()=>document.querySelectorAll("details").forEach(d=>d.open=false);</script></body></html>'
    path = ROOT / 'results/trace_comparisons/qwen35_4b_base_vs_sft_dev12.html'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    print(json.dumps({'output':str(path),'pairs':len(cards),'bytes':path.stat().st_size}))

if __name__ == '__main__':
    main()
