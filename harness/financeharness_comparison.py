"""Full-denominator summaries and model comparison artifacts, never invented scores."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import statistics


def aggregate(records, expected):
    ids = [r['financebench_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate question ID')
    if set(ids) != set(expected):
        raise ValueError('full dataset denominator mismatch')
    n = len(records)
    rewards = [r['score']['reward'] for r in records
               if r.get('score', {}).get('unresolved') is False and r['score'].get('reward') is not None]
    resolved = [r for r in records if r.get('correctness_score', {}).get('unresolved') is False]
    correct = sum(r['correctness_score'].get('correct') == 1 for r in resolved)
    missing = n-len(resolved)
    return {'n': n, 'mean_resolved_reward': statistics.mean(rewards) if rewards else None,
            'reward_resolved': len(rewards), 'reward_unresolved': n-len(rewards),
            'correct': correct, 'correctness_resolved': len(resolved), 'correctness_unresolved': missing,
            'pure_correctness_pct': 100*correct/n if n and not missing else None,
            'correctness_bounds_pct': [100*correct/n, 100*(correct+missing)/n] if n else [None, None],
            'accepted_finishes': sum(r.get('submission') is not None for r in records),
            'errors': sum(r.get('termination_reason') == 'error' for r in records),
            'mean_tool_calls': statistics.mean(r['tool_call_count'] for r in records) if n else None}


def graph_series(models):
    return {'models': [label for label, _ in models],
            'reward_x100': [s['mean_resolved_reward']*100 if s['reward_unresolved'] == 0 else None
                           for _, s in models],
            'correctness_pct': [s['pure_correctness_pct'] for _, s in models]}


def paired_examples(full, baseline):
    control = {r['financebench_id']: r for r in baseline}
    out = {'improvements': [], 'regressions': []}
    for r in full:
        b = control.get(r['financebench_id'])
        if not b or r.get('correctness_score', {}).get('unresolved') is not False or b.get('correctness_score', {}).get('unresolved') is not False:
            continue
        a, c = r['correctness_score']['correct'], b['correctness_score']['correct']
        if a == 1 and c == 0:
            out['improvements'].append(r['financebench_id'])
        elif a == 0 and c == 1:
            out['regressions'].append(r['financebench_id'])
    return out


def svg_graph(series):
    w, h = 1040, 460
    left, top, right, bottom = 80, 55, 990, 335
    labels = series['models']
    xpos = [left+(right-left)*i/max(1, len(labels)-1) for i in range(len(labels))]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="80" y="28" font-family="sans-serif" font-size="20">Financeharness · full FinanceBench model comparison</text>']
    for value in range(0, 101, 20):
        y = bottom-(bottom-top)*value/100
        parts.extend([f'<path d="M{left},{y} H{right}" stroke="#dce3ec"/>',
                      f'<text x="65" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="14">{value}</text>'])
    for x, label in zip(xpos, labels):
        parts.append(f'<text x="{x}" y="365" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    for key, color, label in [('reward_x100', '#167b73', 'Mean reward × 100 (not accuracy)'),
                              ('correctness_pct', '#7656cf', 'Pure correctness (%)')]:
        last = None
        for x, val in zip(xpos, series[key]):
            if val is None:
                last = None
                continue
            y = bottom-(bottom-top)*val/100
            if last:
                parts.append(f'<path d="M{last[0]},{last[1]} L{x},{y}" stroke="{color}" stroke-width="3" fill="none"/>')
            parts.extend([f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>',
                          f'<text x="{x+8}" y="{y-8}" font-family="sans-serif" font-size="13" fill="{color}">{val:.1f}</text>'])
            last = (x, y)
        lx = 80 if key == 'reward_x100' else 580
        parts.append(f'<text x="{lx}" y="410" font-family="sans-serif" font-size="15" fill="{color}">{html.escape(label)}</text>')
    parts.append('<text x="80" y="439" font-family="sans-serif" font-size="12" fill="#555">Lines connect categorical model labels; no model-size trend is implied. Unresolved/incomplete series are omitted.</text></svg>')
    return '\n'.join(parts)


def load_projection(path):
    keep = ('financebench_id', 'question', 'gold', 'answer_text', 'submission', 'termination_reason',
            'tool_call_count', 'tool_calls', 'score', 'correctness_score',
            'correctness_candidate', 'candidate_source')
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                rows.append({k: record.get(k) for k in keep})
    return rows


def build_report(config, run_root):
    run_root = Path(run_root)
    excluded = config.get('excluded_incomplete_runs', [])
    if {job['name'] for job in config['jobs']} & {entry['name'] for entry in excluded}:
        raise ValueError('A run cannot be both included and excluded from the final comparison')
    data = json.loads(Path(config['split_file']).read_text())
    expected = {r['financebench_id'] for name in ('train', 'dev', 'eval') for r in data[name]}
    if config.get('expected_count') is not None and len(expected) != config['expected_count']:
        raise ValueError('Frozen report denominator differs from expected count')
    results, groups = [], {}
    unadjudicated = []
    for job in config['jobs']:
        directory = Path(job.get('result_directory') or job['name'])
        if not directory.is_absolute():
            directory = run_root/directory
        file = directory/'scored.jsonl'
        if not file.exists():
            continue
        records = load_projection(file)
        if {r['financebench_id'] for r in records} != expected:
            continue
        result = aggregate(records, expected)
        result.update(result_directory=str(directory), scored_path=str(file))
        if any(not isinstance(r.get(kind), dict) or type(r[kind].get('unresolved')) is not bool
               for r in records for kind in ('score', 'correctness_score')):
            unadjudicated.append(job['name'])
        results.append((job['label'], result))
        groups[job['name']] = records
    if config.get('require_complete') and (len(results) != len(config['jobs']) or unadjudicated):
        raise ValueError('Final comparison incomplete: every configured question needs both score outcomes, including explicit unresolved outcomes')
    out = run_root/'report'
    out.mkdir(parents=True, exist_ok=True)
    (out/'excluded_incomplete_runs.json').write_text(json.dumps(excluded, indent=2))
    (out/'results.json').write_text(json.dumps(dict(results), indent=2))
    fields = ['model', 'n', 'mean_resolved_reward', 'reward_resolved', 'reward_unresolved',
              'correct', 'pure_correctness_pct', 'correctness_unresolved', 'accepted_finishes', 'mean_tool_calls', 'errors',
              'result_directory', 'scored_path']
    with (out/'results.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for label, stats in results:
            writer.writerow({'model': label, **{k: stats[k] for k in fields[1:]}})
    (out/'comparison.svg').write_text(svg_graph(graph_series(results)))
    parts = ['<!doctype html><meta charset="utf-8"><title>Financeharness comparison</title>',
             '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;color:#172432}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}article{padding:20px;background:#f3f6fa;margin:20px 0}</style>',
             '<h1>Financeharness: model comparison</h1><p>All 150 public FinanceBench questions. GPT 5.6 Terra judge. Composite reward is distinct from independently judged pure correctness.</p>',
             '<img src="comparison.svg" width="100%" alt="Reward and correctness comparison">',
             '<h2>Reward scoring</h2><table><tr><th>Model</th><th>Mean reward</th><th>Resolved / 150</th><th>Accepted finishes</th><th>Mean calls</th><th>Scored source</th></tr>']
    def show(value, precision=3):
        return 'Unresolved' if value is None else f'{value:.{precision}f}'
    for label, s in results:
        parts.append(f'<tr><td>{html.escape(label)}</td><td>{show(s["mean_resolved_reward"])}</td><td>{s["reward_resolved"]}/150</td><td>{s["accepted_finishes"]}/150</td><td>{show(s["mean_tool_calls"],2)}</td><td style="font-size:12px;overflow-wrap:anywhere">{html.escape(s["scored_path"])}</td></tr>')
    parts.append('</table><h2>Pure correctness</h2><table><tr><th>Model</th><th>Confirmed correct / 150</th><th>Correctness %</th><th>Unresolved</th><th>Possible % interval</th></tr>')
    for label, s in results:
        bounds = s['correctness_bounds_pct']
        parts.append(f'<tr><td>{html.escape(label)}</td><td>{s["correct"]}/150</td><td>{show(s["pure_correctness_pct"],1)}</td><td>{s["correctness_unresolved"]}</td><td>{bounds[0]:.1f}–{bounds[1]:.1f}</td></tr>')
    parts.append('</table><p>Reward means disclose resolved denominators; unknown scores are never replaced by zero. A definitive correctness percentage requires all judgments resolved.</p>')
    if excluded:
        parts.append('<h2>Cancelled — incomplete runs</h2><p>Excluded from full-dataset tables and graph. No accuracy estimate or full-150 score is assigned.</p><table><tr><th>Policy</th><th>Status</th><th>Completed episodes</th><th>Accepted finishes</th><th>Reason / preserved source</th></tr>')
        for entry in excluded:
            parts.append(f'<tr><td>{html.escape(entry.get("label", entry["name"]))}</td><td>Cancelled — incomplete</td><td>{entry["completed"]}/{entry["expected"]}</td><td>{entry.get("accepted_finishes", "Not measured")}</td><td>{html.escape(entry.get("reason", ""))}<br>{html.escape(entry.get("result_directory", ""))}</td></tr>')
        parts.append('</table><p>Partial runs preserve operational diagnostics; missing episodes are not scored as incorrect or extrapolated to a full release.</p>')
    if 'qwen4b' in groups and 'qwen4b_minimal' in groups:
        examples = paired_examples(groups['qwen4b'], groups['qwen4b_minimal'])
        (out/'paired_examples.json').write_text(json.dumps(examples, indent=2))
        index = {key: {r['financebench_id']: r for r in rows} for key, rows in groups.items()}
        for category in ('improvements', 'regressions'):
            parts.append(f'<h2>Paired {category}</h2>')
            for fid in examples[category][:3]:
                r = index['qwen4b'][fid]
                parts.append(f'<article><h3>{html.escape(fid)}</h3><p>{html.escape(r["question"])}</p>')
                for key, label in [('qwen4b_minimal','Minimal tools'),('qwen4b','Financeharness')]:
                    parts.append(f'<h4>{label}</h4><pre>{html.escape(json.dumps(index[key][fid],indent=2))}</pre>')
                parts.append('</article>')
    (out/'comparison.html').write_text('\n'.join(parts))
    return dict(results)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-root', required=True)
    args = ap.parse_args()
    print(json.dumps(build_report(json.loads(Path(args.config).read_text()), args.run_root), indent=2))
