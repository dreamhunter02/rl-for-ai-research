import importlib
from types import SimpleNamespace


def test_base_and_adapter_commands_have_matching_generation_settings(tmp_path):
    assert importlib.util.find_spec('run_matched_dev12'), 'background supervisor missing'
    m=importlib.import_module('run_matched_dev12')
    args=SimpleNamespace(root=tmp_path,run_dir=tmp_path/'run',base_url='http://test/v1',
        base_model='base',sft_model='base:lora',max_turns=8,max_tokens=1024)
    a=m.eval_command(args,'base');b=m.eval_command(args,'sft')
    for key in ('--base-url','--split-file','--targets','--max-turns','--max-tokens','--temperature','--seed'):
        assert a[a.index(key)+1]==b[b.index(key)+1]
    assert a[a.index('--model')+1]=='base' and b[b.index('--model')+1]=='base:lora'
    assert a[a.index('--out')+1]!=b[b.index('--out')+1]
