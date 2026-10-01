import importlib
import os
import pytest


def module():
    assert importlib.util.find_spec('run_identity'), 'shared configuration identity missing'
    return importlib.import_module('run_identity')


def test_judge_and_scorer_change_identity_without_secrets(monkeypatch):
    m=module(); monkeypatch.setenv('COMPONENT_JUDGE_ENDPOINT','https://a/v1');monkeypatch.setenv('COMPONENT_JUDGE_API_KEY','DO-NOT-LOG')
    one=m.judging_identity();monkeypatch.setenv('COMPONENT_JUDGE_ENDPOINT','https://b/v1');two=m.judging_identity()
    assert one!=two and 'DO-NOT-LOG' not in str(one)
    monkeypatch.setenv('FINANCEBENCH_SCORER','legacy');assert two!=m.judging_identity()


def test_resume_without_manifest_fails_closed(tmp_path):
    m=module();out=tmp_path/'old.jsonl';out.write_text('{}\n')
    with pytest.raises(ValueError):m.verify_run_config(out,{'new':1})


def test_missing_component_judge_fails_before_paid_rollouts(monkeypatch):
    m=module(); assert hasattr(m,'require_current_judge'), 'preflight judge gate missing'
    monkeypatch.setenv('FINANCEBENCH_SCORER','components')
    with pytest.raises(ValueError):m.require_current_judge(None)
