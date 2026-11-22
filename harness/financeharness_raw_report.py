"""Offline, unscored trajectory review. Uses only the Python standard library."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def _json(value):
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def _details(label, value):
    return '<details><summary>'+html.escape(label)+'</summary><pre>'+_json(value)+'</pre></details>'


def _actor_question(row):
    # Exact stdlib equivalent of pinned generate_teacher_traces._rows input normalization.
    question = str(row['question']).strip()
    company = row.get('company') or row.get('doc_name', '').split('_')[0]
    if company and company.lower() not in question.lower():
        question = 'About '+company+': '+question
    return question


def _reference(row):
    answer = row.get('answer')
    # Also support already-normalized split fixtures, whose answer is a one-item list.
    if isinstance(answer, list):
        answer = answer[0] if answer else ''
    return str(answer or '').strip()


def build_report(config, output, partial_preview=False):
    """Render a fresh HTML snapshot; validate frozen IDs before creating output."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    bundle = config.get('bundle_questions', False)
    question_dir = output.with_name(output.stem+'_questions')
    if bundle and question_dir.exists():
        raise FileExistsError(question_dir)
    split = json.loads(Path(config['split_file']).read_text())
    frozen = [row for key in ('train', 'dev', 'eval') for row in split.get(key, [])]
    expected_count = config.get('expected_count', 150)
    ids = [row['financebench_id'] for row in frozen]
    if len(ids) != expected_count or len(set(ids)) != len(ids):
        raise ValueError('Frozen questions must contain exactly '+str(expected_count)+' unique IDs')
    questions = {row['financebench_id']: row for row in frozen}
    jobs = config['jobs']
    names = [job['name'] for job in jobs]
    if len(names) != len(set(names)) or not jobs:
        raise ValueError('Configured arms must be nonempty and uniquely named')
    excluded = config.get('excluded_incomplete_runs', [])
    if set(names) & {row.get('name') for row in excluded}:
        raise ValueError('Cancelled arms cannot enter the continuing comparison')
    runs, lineage = [], []
    for job in jobs:
        path = Path(job['raw_file'])
        data = path.read_bytes()
        records = [json.loads(line) for line in data.splitlines() if line.strip()]
        index = {}
        for record in records:
            fid = record['financebench_id']
            if fid in index or fid not in questions:
                raise ValueError('Duplicate or unknown ID in '+job['name'])
            row = questions[fid]
            if record.get('question') != _actor_question(row):
                raise ValueError('Question mismatch in '+job['name']+': '+str(fid))
            reference = _reference(row)
            if 'gold' in record and record['gold'] != reference:
                raise ValueError('Reference mismatch in '+job['name']+': '+str(fid))
            index[fid] = record
        if not partial_preview and set(index) != set(ids):
            raise ValueError('Incomplete raw arm '+job['name']+': '+str(len(index))+'/'+str(expected_count))
        runs.append((job, index))
        lineage.append({'name': job['name'], 'raw_file': str(path), 'sha256': hashlib.sha256(data).hexdigest(),
                        'completed': len(index)})
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Financeharness raw trajectories</title>',
            '<style>body{font:16px system-ui;margin:24px;background:#f5f6f8} .panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px} .policy-panel{background:white;padding:14px;border:1px solid #ccc;border-radius:8px} pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px} details{margin:10px 0} h2{margin-top:36px}@media(max-width:1000px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:600px){.panels{grid-template-columns:1fr}}</style>',
            '<h1>Financeharness raw trajectory review'+(' — PARTIAL PREVIEW' if partial_preview else '')+'</h1>',
            '<p>Scores: Deferred—not judged. Operational termination and accepted finish are not reward or correctness. Reference answers are for review only; they were not supplied to actors.</p>',
            _details('Input provenance (SHA256)', lineage)]
    if excluded:
        page.append('<h2>Cancelled incomplete runs</h2><p>No full-dataset score or accuracy is assigned; no extrapolation.</p>')
        for item in excluded:
            page.append('<h3>'+html.escape(item.get('label', item.get('name', 'Cancelled')))+' — Cancelled, incomplete</h3><p>'+str(item.get('completed', '?'))+' / '+str(item.get('expected', expected_count))+' episodes; '+html.escape(item.get('reason', 'User cancelled'))+'</p>'+_details('Preserved cancellation metadata', item))
    index_page = page.copy()
    if bundle:
        output.parent.mkdir(parents=True, exist_ok=True)
        question_dir.mkdir()  # Refuse any existing snapshot directory.
        index_page.append('<h2>Questions</h2><ol>')
    for number, row in enumerate(frozen, 1):
        fid = row['financebench_id']
        if bundle:
            filename = 'question_'+str(number).zfill(3)
            page = index_page[:4]
            links = ['<a href="../'+html.escape(output.name)+'">All questions</a>']
            if number > 1:
                links.append('<a href="question_'+str(number-1).zfill(3)+'.html">Previous</a>')
            if number < len(frozen):
                links.append('<a href="question_'+str(number+1).zfill(3)+'.html">Next</a>')
            page.append('<nav>'+' | '.join(links)+'</nav><p><a download href="'+filename+'.json">Download full raw records including exact request contexts (JSON)</a></p>')
            index_page.append('<li><a href="'+html.escape(question_dir.name)+'/'+filename+'.html">'+html.escape(str(fid))+' — '+html.escape(row['question'])+'</a></li>')
        page.append('<section><h2>'+html.escape(str(fid))+'</h2><h3>Frozen original question</h3><pre>'+html.escape(row['question'])+'</pre><h3>Exact actor question</h3><pre>'+html.escape(_actor_question(row))+'</pre>'+_details('Reference answer', _reference(row))+'<div class="panels">')
        for job, records in runs:
            page.append('<article class="policy-panel"><h3>'+html.escape(job.get('label', job['name']))+'</h3>')
            record = records.get(fid)
            if record is None:
                page.append('<p>Missing raw episode (partial preview only)</p></article>')
                continue
            submission = record.get('submission')
            count = record.get('actual_tool_call_count', record.get('tool_call_count', len(record.get('tool_calls', []))))
            page.append('<p>Termination: '+html.escape(str(record.get('termination_reason', 'not recorded')))+'<br>Attempted tool calls: '+html.escape(str(count))+'<br>Accepted finish: '+('Yes' if submission is not None else 'No')+'</p><p>Scores: Deferred—not judged</p>')
            # Display saved explicit text and all typed fields, without reconstructing an answer from tool observations.
            page.append('<h4>Explicit actor answer</h4><pre>'+_json(record.get('answer_text', record.get('answer', '')) )+'</pre>')
            if submission is not None:
                page.append('<h4>Accepted typed submission</h4><pre>'+_json(submission)+'</pre>')
            for label, key in [('Attempted tool calls', 'tool_calls'), ('Tool observations / full trace', 'trace'),
                               ('Authentic source receipts', 'source_receipts'), ('Provider error', 'error'),
                               ('Generation metadata', 'generation_metadata'), ('Persistent state', 'persistent_state'),
                               ('Exact request contexts', 'request_contexts')]:
                if bundle and key == 'request_contexts':
                    page.append('<p>Exact request contexts: preserved in the full raw JSON download.</p>')
                    continue
                page.append(_details(label, record.get(key, [])))
            page.append('</article>')
        page.append('</div></section>')
        if bundle:
            with (question_dir/(filename+'.html')).open('x', encoding='utf-8') as stream:
                stream.write('\n'.join(page+['</html>']))
            with (question_dir/(filename+'.json')).open('x', encoding='utf-8') as stream:
                json.dump({'frozen_question': row, 'arms': {job['name']: records.get(fid) for job, records in runs}}, stream, ensure_ascii=False)
    if bundle:
        page = index_page+['</ol>']
    page.append('</html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(page))
    return {'output': str(output), 'questions': len(frozen), 'partial_preview': partial_preview, 'arms': lineage,
            'question_directory': str(question_dir) if bundle else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--partial-preview', action='store_true', help='Explicitly permit missing episodes; never a final report')
    parser.add_argument('--single-page', action='store_true', help='Opt into a potentially large single HTML instead of a paged bundle')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config['bundle_questions'] = not args.single_page
    print(json.dumps(build_report(config, args.output, args.partial_preview), indent=2))


if __name__ == '__main__':
    main()
