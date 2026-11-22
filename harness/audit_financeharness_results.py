"""Read-only final Financeharness audit; never changes episodes or gold answers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REWARD_VERSION = 'workshop-rubric-v6-additive'
CORRECTNESS_VERSION = 'financeharness-correctness-v1'
JUDGE_MODEL = 'openai/openai/gpt-5.6-terra'
FINANCIAL_KEYS = ('model', 'split_sha256', 'corpus_scope', 'max_tool_calls', 'max_tokens',
                  'temperature', 'seed', 'forced_finish', 'budget_cues', 'extraction_mode',
                  'recent_turn_policy', 'harness_mode', 'memory_version', 'submission_contract',
                  'require_citations', 'prompt_char_budget', 'recent_turns', 'system_prompt')
CUE_KEYS = {'turns_remaining', 'next_action', 'countdown', 'last_turn_reminder'}
MODELS = {'qwen4b': 'Qwen3.5-4B', 'qwen4b_minimal': 'Qwen3.5-4B',
          'lfm': 'LiquidAI/LFM2.5-8B-A1B', 'qwen9b': 'nvidia/qwen/qwen3.5-9b',
          'gemini': 'gcp/google/gemini-3.8-flash', 'glm': 'nvidia/zai-org/glm-5.3-flash'}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _cue_paths(value, path=''):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            location = f'{path}.{key}'
            if key in CUE_KEYS or key in {'synthetic', 'synthetic_finish', 'forced_finish', 'budget_cues'} and child is True:
                found.append(location)
            found.extend(_cue_paths(child, location))
        # Only tool observation content is parsed; ordinary actor/question prose
        # and static instructions mentioning finish are never treated as cues.
        if value.get('role') == 'tool' and isinstance(value.get('content'), str):
            try:
                found.extend(_cue_paths(json.loads(value['content']), f'{path}.tool_payload'))
            except (ValueError, TypeError):
                pass
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_cue_paths(child, f'{path}[{index}]'))
    return found


def audit_records(raw_records, scored_records, frozen_questions, protocol, expected_protocol,
                  *, expected_count=150):
    """Return findings and denominators; genuine unresolved judgments do not fail."""
    errors = []
    questions = list(frozen_questions)
    frozen = {q.get('financebench_id'): q for q in questions}
    if len(questions) != expected_count or len(frozen) != expected_count or None in frozen:
        errors.append('Frozen questions must contain exactly the expected unique IDs')
    for key in FINANCIAL_KEYS:
        if key not in expected_protocol or key not in protocol or protocol[key] != expected_protocol[key]:
            errors.append(f'Protocol pin missing or mismatched: {key}')
    for key, value in expected_protocol.items():
        if key.startswith('corpus_') and key != 'corpus_scope' and protocol.get(key) != value:
            errors.append(f'Corpus identity pin missing or mismatched: {key}')
    if not any(key in expected_protocol for key in ('corpus_identities', 'corpus_manifest_sha256')):
        errors.append('Independent corpus identity pin is required')
    if protocol.get('max_tool_calls') != 50 or protocol.get('max_tokens') != 1024:
        errors.append('Financial action/token budgets must be 50 / 1024')
    if protocol.get('forced_finish') is not False or protocol.get('budget_cues') is not False:
        errors.append('Protocol enables synthetic finish or budget cues')

    mapped = {}
    for kind, records in [('raw', list(raw_records)), ('scored', list(scored_records))]:
        ids = [r.get('financebench_id') for r in records]
        if len(ids) != expected_count or len(set(ids)) != expected_count or set(ids) != set(frozen):
            errors.append(f'{kind}: missing, duplicate, or foreign question IDs')
        mapped[kind] = {r.get('financebench_id'): r for r in records}
        for row in records:
            fid = row.get('financebench_id')
            prefix = f'{kind}:{fid}'
            question = frozen.get(fid)
            if question:
                if row.get('question') != question.get('question'):
                    errors.append(f'{prefix}: question differs from frozen text')
                gold = question.get('answer')
                gold = gold[0] if isinstance(gold, list) and gold else gold
                if row.get('gold') != gold:
                    errors.append(f'{prefix}: gold differs from frozen reference')
            if row.get('model') != expected_protocol.get('model'):
                errors.append(f'{prefix}: actor model differs')
            calls = row.get('tool_calls')
            count = row.get('tool_call_count')
            if not isinstance(calls, list) or type(count) is not int or count < 0 or count > 50 or count != len(calls):
                errors.append(f'{prefix}: invalid attempted tool count or cap')
            if row.get('submission') is not None and not any(call.get('name') == 'finish' for call in calls or []):
                errors.append(f'{prefix}: accepted submission has no actual finish attempt')
            for field in ('request_contexts', 'trace'):
                if not isinstance(row.get(field), list):
                    errors.append(f'{prefix}: missing saved {field}')
            if _cue_paths({key: row.get(key) for key in ('request_contexts', 'trace', 'tool_calls',
                                                        'forced_finish', 'synthetic_finish', 'budget_cues')}):
                errors.append(f'{prefix}: synthetic finish or structured budget cue')

    for fid in set(mapped['raw']) & set(mapped['scored']):
        raw, scored = mapped['raw'][fid], mapped['scored'][fid]
        for key, value in raw.items():
            if key not in ('score', 'correctness_score', 'correctness_candidate', 'candidate_source') and scored.get(key) != value:
                errors.append(f'scored:{fid}: actor field changed: {key}')

    reward_resolved = correctness_resolved = pure_correct = 0
    rewards = []
    for row in mapped['scored'].values():
        fid = row.get('financebench_id')
        score = row.get('score', {})
        correct = row.get('correctness_score', {})
        if score.get('unresolved') is True:
            if 'reward' not in score or score.get('reward') is not None:
                errors.append(f'{fid}: unresolved reward must be null')
        elif score.get('unresolved') is False:
            reward_resolved += 1
            values = [score.get(key) for key in ('E', 'G', 'A', 'B', 'F')]
            if score.get('scorer_version') != REWARD_VERSION or not all(_number(v) and 0 <= v <= 1 for v in values):
                errors.append(f'{fid}: invalid resolved v6 components/version')
            else:
                E, G, A, B, F = values
                target = .15*E + F*(.20*G + .45*A + .15*B + .05)
                if B not in (0, 1) or F not in (0, 1) or B == 1 and A != 1:
                    errors.append(f'{fid}: invalid binary reward components')
                if F != int(row.get('submission') is not None):
                    errors.append(f'{fid}: finish flag differs from accepted submission')
                if not _number(score.get('reward')) or abs(score['reward'] - target) > 1e-9:
                    errors.append(f'{fid}: v6 reward arithmetic mismatch')
                else:
                    rewards.append(score['reward'])
        else:
            errors.append(f'{fid}: missing reward unresolved status')
        if correct.get('unresolved') is True:
            if 'correct' not in correct or correct.get('correct') is not None:
                errors.append(f'{fid}: unresolved correctness must be null')
        elif correct.get('unresolved') is False:
            correctness_resolved += 1
            if type(correct.get('correct')) is not int or correct['correct'] not in (0, 1) or correct.get('scorer_version') != CORRECTNESS_VERSION:
                errors.append(f'{fid}: invalid independent correctness/version')
            else:
                pure_correct += correct['correct']
        else:
            errors.append(f'{fid}: missing independent correctness unresolved status')
    flags = [{'financebench_id': fid, 'flag': 'known_reference_ambiguity_00684',
              'action': 'Disclose separately; preserve the frozen gold and judge uncertainty.'}
             for fid in frozen if str(fid).endswith('00684')]
    return {'ok': not errors, 'errors': errors,
            'denominators': {'expected': expected_count, 'raw': len(raw_records), 'scored': len(scored_records),
                             'reward_resolved': reward_resolved, 'reward_unresolved': len(scored_records)-reward_resolved,
                             'correctness_resolved': correctness_resolved, 'correctness_unresolved': len(scored_records)-correctness_resolved},
            'counts': {'provider_errors': sum(r.get('termination_reason') == 'error' for r in raw_records),
                       'accepted_submissions': sum(r.get('submission') is not None for r in raw_records),
                       'pure_correct': pure_correct},
            'mean_reward_resolved': sum(rewards)/len(rewards) if rewards else None,
            'known_reference_flags': flags,
            'provider_infrastructure': protocol.get('provider_infrastructure', {}),
            'provider_settings': {key: protocol.get(key) for key in ('workers', 'base_url', 'reasoning_effort')},
            'question_hash': protocol.get('split_sha256')}


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_run(result_dir, frozen_questions, expected_protocol, *, expected_count=150,
              source_manifest=None, source_manifest_sha256=None):
    """Inspect one local result directory; perform no source/output writes."""
    root = Path(result_dir)
    try:
        protocol = json.loads((root/'protocol.json').read_text())
        if source_manifest is not None:
            blob = Path(source_manifest).read_bytes()
            if not source_manifest_sha256 or hashlib.sha256(blob).hexdigest() != source_manifest_sha256:
                raise ValueError('Source experiment manifest SHA256 mismatch')
            manifest = json.loads(blob)
            if manifest.get('corpus_sha256') != expected_protocol.get('corpus_identities'):
                raise ValueError('Source manifest corpus identities mismatch')
            if manifest.get('split_sha256') != expected_protocol.get('split_sha256'):
                raise ValueError('Source manifest frozen question hash mismatch')
            if len(manifest.get('question_ids', [])) != expected_count or set(manifest['question_ids']) != {q['financebench_id'] for q in frozen_questions}:
                raise ValueError('Source manifest question identities mismatch')
            if expected_count == 150 and len(manifest['corpus_sha256']) != 84:
                raise ValueError('Full-release source manifest must pin all 84 corpus PDFs')
            for name, digest in manifest.get('source_sha256', {}).items():
                if name in {'eval_stateful_harness.py', 'finance_env.py', 'financebench_harness.py',
                            'teacher_runtime.py', 'persistent_state.py', 'observations.py'} and protocol.get('source_sha256', {}).get(name) != digest:
                    raise ValueError(f'Source protocol actor implementation hash mismatch: {name}')
            protocol['corpus_identities'] = manifest['corpus_sha256']
        report = audit_records(_read_rows(root/'base.jsonl'), _read_rows(root/'scored.jsonl'),
                               frozen_questions, protocol, expected_protocol, expected_count=expected_count)
        scoring = json.loads((root/'scoring_protocol.json').read_text())
        for key, value in [('judge_model', JUDGE_MODEL), ('reward_version', REWARD_VERSION),
                           ('correctness_version', CORRECTNESS_VERSION)]:
            if scoring.get(key) != value:
                report['errors'].append(f'Scoring protocol mismatch: {key}')
        report['ok'] = not report['errors']
        report['result_directory'] = str(root)
        return report
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {'ok': False, 'errors': [f'Unreadable/incomplete artifact: {type(exc).__name__}: {exc}'],
                'result_directory': str(root)}


def load_frozen_questions(path):
    """Load JSON rows or a train/dev/eval split union; never select by outcomes."""
    path = Path(path)
    value = json.loads(path.read_text()) if path.suffix != '.jsonl' else _read_rows(path)
    if isinstance(value, list):
        return value
    return [row for key in ('train', 'dev', 'eval') for row in value.get(key, [])]


def audit_report(config, frozen_questions=None):
    """Audit report-only jobs with independently pinned financial/corpus identity."""
    if config.get('report_only') is not True:
        return {'ok': False, 'errors': ['Final config must be report_only=true'], 'jobs': []}
    questions = frozen_questions if frozen_questions is not None else load_frozen_questions(config['split_file'])
    split_hash = hashlib.sha256(Path(config['split_file']).read_bytes()).hexdigest()
    corpus = config.get('corpus_identities')
    corpus_hash = config.get('corpus_manifest_sha256')
    if config.get('corpus_manifest'):
        actual = hashlib.sha256(Path(config['corpus_manifest']).read_bytes()).hexdigest()
        if actual != corpus_hash:
            return {'ok': False, 'errors': ['Frozen corpus manifest hash missing or mismatched'], 'jobs': []}
    source_manifests = {}
    for name, pin in config.get('corpus_manifests', {}).items():
        blob = Path(pin['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != pin['sha256']:
            return {'ok': False, 'errors': [f'Experiment manifest SHA256 mismatch: {name}'], 'jobs': []}
        manifest = json.loads(blob)
        identities = manifest.get('corpus_sha256')
        if not isinstance(identities, dict) or not identities:
            return {'ok': False, 'errors': [f'Experiment manifest lacks corpus identities: {name}'], 'jobs': []}
        if corpus is None:
            corpus = identities
        elif corpus != identities:
            return {'ok': False, 'errors': ['Experiment manifests have different corpus identities'], 'jobs': []}
        source_manifests[name] = pin
    reports = []
    for job in config['jobs']:
        root = Path(job['result_directory'])
        try:
            expected = dict(config.get('financial_protocol', {}), **job.get('expected_protocol', {}))
            expected.update(model=job.get('model', MODELS.get(job['name'])), split_sha256=split_hash,
                            max_tool_calls=50, max_tokens=1024, temperature=0, seed=0,
                            forced_finish=False, budget_cues=False, corpus_scope='frozen_split_sources',
                            extraction_mode='plain', recent_turn_policy='complete',
                            submission_contract=True, require_citations=True, prompt_char_budget=28000)
            mode = job.get('harness_mode', 'minimal' if job['name'].endswith('_minimal') else 'full')
            expected.update(harness_mode=mode, memory_version='persistent-selectable-turns-v4' if mode == 'full' else None,
                            recent_turns=3 if mode == 'full' else None)
            # Static prompt is pinned to the current explicitly selected harness
            # contract, not copied from the observed protocol.
            if 'system_prompt' not in expected:
                from eval_stateful_harness import system_prompt
                expected['system_prompt'] = system_prompt(True, mode)
            if corpus is not None:
                expected['corpus_identities'] = corpus
            elif corpus_hash is not None:
                expected['corpus_manifest_sha256'] = corpus_hash
            manifest_pin = source_manifests.get(job.get('source_manifest'), {})
            report = audit_run(root, questions, expected, expected_count=config.get('expected_count', 150),
                               source_manifest=manifest_pin.get('path'), source_manifest_sha256=manifest_pin.get('sha256'))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report = {'ok': False, 'errors': [f'Job artifact unavailable: {type(exc).__name__}: {exc}']}
        report['name'] = job['name']
        reports.append(report)
    return {'ok': bool(reports) and all(report['ok'] for report in reports),
            'errors': [], 'jobs': reports, 'question_hash': split_hash,
            'corpus_manifest_sha256': corpus_hash, 'corpus_identities': corpus}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh audit JSON path; existing files refused')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        report = audit_report(json.loads(args.config.read_text()))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        report = {'ok': False, 'errors': [f'Audit input unavailable: {type(exc).__name__}: {exc}'], 'jobs': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'ok': report['ok'], 'output': str(args.output)}))
    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
