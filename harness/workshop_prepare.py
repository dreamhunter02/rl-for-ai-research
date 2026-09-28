"""Offline preparation and preflight; never automatically approves gold targets."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from workshop_reward import parse_number, validate_target, SCORER_VERSION, EpisodeState, numbers_equal, typed_number


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped_split(split, dev_size=12, seed=0):
    groups = {}
    for row in split['train']: groups.setdefault(row['doc_name'], []).append(row)
    docs = sorted(groups)
    random.Random(seed).shuffle(docs)
    selected, count = [], 0
    for doc in docs:
        if count < dev_size and len(groups[doc]) < len(split['train'])-count:
            selected.append(doc)
            count += len(groups[doc])
    if not selected: raise ValueError('Cannot create document-disjoint train/dev with available documents')
    return {'train': [r for r in split['train'] if r['doc_name'] not in selected],
            'dev': [r for r in split['train'] if r['doc_name'] in selected], 'eval': split['eval']}


def draft_target(row):
    gold = str(row['answer']).strip()
    target = {'reviewed': False, 'question': row['question'], 'original_answer': gold,
              'derived': False, 'support': [], 'review_notes': 'Verify answer type, required facts, precision, units, periods, source spans and arithmetic before setting reviewed=true.'}
    try:
        value, unit, scale, precision = parse_number(gold)
        target.update(answer_type='numeric', value=str(value), unit=unit, scale=scale, precision=precision)
    except ValueError:
        target.update(answer_type='text', aliases=[gold], required_facts=[])
    for evidence in row.get('evidence', []):
        page = evidence.get('evidence_page_num')
        quote = evidence.get('evidence_text', '')
        if page is not None:
            target['support'].append({'document_id': row['doc_name'], 'page': int(page)+1, 'quote': quote,
                                      'claim': 'answer', 'normalization': 'FinanceBench zero-based -> one-based'})
    return target


def prepare(split_path, output, seed=0, split_manifest=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads(Path(split_path).read_text())
    if split_manifest:
        ids = json.loads(Path(split_manifest).read_text())
        by_id = {r['financebench_id']: r for rows in original.values() for r in rows}
        mapping = {'train': 'train96_ids', 'dev': 'dev12_ids', 'eval': 'eval42_ids'}
        scheduled = [qid for key in mapping.values() for qid in ids[key]]
        if len(set(scheduled)) != len(scheduled) or set(scheduled) != set(by_id):
            raise ValueError('Frozen manifest must partition all source questions exactly once')
        original_eval = {r['financebench_id'] for r in original['eval']}
        if set(ids['eval42_ids']) != original_eval:
            raise ValueError('Frozen manifest may not move evaluation questions')
        split = {name: [by_id[qid] for qid in ids[key]] for name, key in mapping.items()}
    elif 'dev' in original:
        split = original
    else:
        split = grouped_split(original, seed=seed)
    for filename in ('split.json', 'targets.draft.json', 'protocol.json'):
        if (out/filename).exists(): raise FileExistsError(out/filename)
    targets = {r['financebench_id']: draft_target(r) for rows in split.values() for r in rows}
    docsets = {name: {r['doc_name'] for r in rows} for name, rows in split.items()}
    protocol = {'schema_version': 1, 'scorer_version': SCORER_VERSION, 'seed': seed,
        'evaluation_status': 'previously inspected', 'conditions': ['B0', 'B1', 'R1'],
        'question_ids': {name: [r['financebench_id'] for r in rows] for name, rows in split.items()},
        'document_overlap': {f'{a}/{b}': sorted(docsets[a]&docsets[b]) for a,b in [('train','dev'), ('train','eval'), ('dev','eval')]},
        'bootstrap_resamples': 10000, 'checkpoint_selection': 'dev grounded success, correctness, lower cost; base included',
        'source_split_sha256': sha256(split_path)}
    (out/'split.json').write_text(json.dumps(split, indent=2))
    (out/'targets.draft.json').write_text(json.dumps(targets, indent=2))
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    return {name: len(rows) for name, rows in split.items()}


def preflight(split_path, targets_path, corpus=False):
    split, targets = json.loads(Path(split_path).read_text()), json.loads(Path(targets_path).read_text())
    seen, errors = set(), []
    for name in ('train', 'dev', 'eval'):
        if not split.get(name): errors.append(f'Missing/nonempty split: {name}')
        for row in split.get(name, []):
            qid = row['financebench_id']
            if qid in seen: errors.append(f'Duplicate question across splits: {qid}')
            seen.add(qid)
            try:
                target = validate_target(targets[qid])
                if target.get('derived') and target['answer_type'] == 'numeric':
                    probe = EpisodeState()
                    operands = {}
                    for key, operand in target['operands'].items():
                        span = operand['support'][0]
                        receipt = probe.record('read', {**span, 'text': span['quote']})
                        operands[key] = {**operand, 'receipt_id': receipt, 'quote': span['quote']}
                    probe.turn = 1
                    calculation = probe.calculate(target['expression'], operands)
                    if not calculation['provenance_valid'] or not numbers_equal(
                        typed_number(target['value'], target['unit'], target['scale']),
                        typed_number(calculation['value'], target['unit'], target['scale']), target['precision']):
                        raise ValueError('Reviewed derivation does not reproduce the target answer')
            except (ValueError, KeyError) as exc: errors.append(f'{qid}: {exc}')
    if errors: raise ValueError('\n'.join(errors))
    result = {'split_sha256': sha256(split_path), 'targets_sha256': sha256(targets_path), 'questions': len(seen)}
    if corpus:
        import financebench_harness as hb
        docs = sorted({r['doc_name'] for rows in split.values() for r in rows})
        index = hb.build_index(doc_names=docs)
        if not set(docs).issubset(index.pages_by_doc): raise ValueError('Corpus missing required documents')
        digest = hashlib.sha256()
        for page in index.pages:
            digest.update(json.dumps(page, sort_keys=True).encode())
        for qid in seen:
            spans = targets[qid].get('support', []) + [s for o in targets[qid].get('operands', {}).values() for s in o.get('support', [])]
            for support in spans:
                page = next((p for p in index.pages_by_doc.get(support['document_id'], []) if p['page']==support['page']), {})
                if not support.get('quote') or support['quote'] not in page.get('text', ''):
                    errors.append(f'{qid}: supporting quote not found exactly on declared page')
        if errors: raise ValueError('\n'.join(errors))
        result.update(corpus_sha256=digest.hexdigest(), pages=len(index.pages))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command', required=True)
    p=sub.add_parser('prepare'); p.add_argument('--split', default='split.json'); p.add_argument('--out', required=True); p.add_argument('--seed', type=int, default=0); p.add_argument('--split-manifest')
    p=sub.add_parser('preflight'); p.add_argument('--split', required=True); p.add_argument('--targets', required=True); p.add_argument('--corpus', action='store_true'); p.add_argument('--out')
    args=ap.parse_args()
    result=prepare(args.split,args.out,args.seed,args.split_manifest) if args.command=='prepare' else preflight(args.split,args.targets,args.corpus)
    if args.command=='preflight' and args.out: Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
