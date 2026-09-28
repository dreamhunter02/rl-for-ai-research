import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from workshop_prepare import grouped_split, draft_target
from workshop_report import build_report, paired_interval, validate_records
from training_audit import install_training_audit


class ReportingTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for run,values in [('B1',[0,1,0]),('R1',[1,1,0])]:
            for i,v in enumerate(values):
                rows.append(dict(run_id=run,condition=run,question_id=str(i),document_id='D'+str(i),
                    evaluation_signature='same',F=1,A=v,G=v,correct=v,grounded_success=v,unresolved=False,stop_reason='accepted_finish',
                    output_tokens=10,tool_calls=2,latency_s=.1))
        return rows

    def test_paired_effect_and_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            main,failures,paired=build_report(self.fixture(),['0','1','2'],tmp,resamples=100)
            self.assertAlmostEqual(paired[0]['delta_pp'],100/3)
            self.assertEqual(paired[0]['wins'],1)
            self.assertEqual(main[0]['N_questions'],3)
            self.assertIsNone(main[0]['cost_usd'])

    def test_missing_and_duplicate_evaluations_fail(self):
        rows=self.fixture()
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError): validate_records(bad,['0','1','2'])

    def test_cluster_bootstrap_deterministic(self):
        a={'a':0,'b':1};b={'a':1,'b':1}
        self.assertEqual(paired_interval(a,b,resamples=100),paired_interval(a,b,resamples=100))

    def test_split_has_no_train_dev_document_overlap(self):
        split={'train':[{'doc_name':str(i//2),'financebench_id':str(i)} for i in range(20)],'eval':[]}
        out=grouped_split(split,dev_size=5)
        self.assertFalse({r['doc_name'] for r in out['train']}&{r['doc_name'] for r in out['dev']})
        self.assertEqual(len(out['train'])+len(out['dev']),20)

    def test_target_drafts_require_review(self):
        t=draft_target({'question':'Revenue?','answer':'USD 20 million','doc_name':'D','evidence':[{'evidence_page_num':0,'evidence_text':'hello'}]})
        self.assertFalse(t['reviewed']);self.assertEqual(t['support'][0]['page'],1)

    def test_actual_optimizer_updates_skip_empty_data(self):
        async def train_step(data_D,num_substeps=1,metrics=None): return []
        module=SimpleNamespace(train_step=train_step)
        with tempfile.TemporaryDirectory() as tmp:
            counter=install_training_audit(module,Path(tmp)/'audit.jsonl')
            asyncio.run(module.train_step([],metrics={}))
            asyncio.run(module.train_step([1,2],metrics={}))
            self.assertEqual(counter['optimizer_updates'],1)
            self.assertEqual(counter['skipped_batches'],1)

    def test_all_constant_group_never_calls_optimizer_path(self):
        called=[]
        class Group:
            def get_total_rewards(self):return [0,0]
        async def train_step(data_D,num_substeps=1,metrics=None):called.append('optimizer');return []
        async def group_step(config,i_batch,training_client,checkpoint_mgr,tokenizer,env_group_builders_P,trajectory_groups_P):
            called.append('group');return 'client',{}
        async def save(training_client,checkpoint_mgr,i_batch):return 'unchanged-client',{}
        module=SimpleNamespace(train_step=train_step,do_train_step_and_get_sampling_client=group_step,
            save_checkpoint_and_get_sampling_client=save,remove_constant_reward_groups=lambda groups:groups[:1])
        with tempfile.TemporaryDirectory() as tmp:
            counter=install_training_audit(module,Path(tmp)/'audit.jsonl')
            result=asyncio.run(module.do_train_step_and_get_sampling_client(None,0,None,None,None,[object()],[Group()]))
            self.assertEqual(called,[])
            self.assertEqual(counter['optimizer_updates'],0)
            self.assertEqual(counter['skipped_batches'],1)
