"""Exercise real episode/reward method bodies without importing the absent SDK.

Only the external tool decorator/result container are replaced. These tests do
not validate renderer or SDK contracts; test_finance_env.py supplies that gate.
"""
import ast
import asyncio
import json
import os
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Annotated
import financebench_harness as hb
from observations import bounded_observation, compact_hits
from reward_calculation import RewardConfig
from workshop_reward import EpisodeState, score_submission, validate_submission


def load_episode_classes():
    source=ast.parse(Path(hb.__file__).with_name('finance_env.py').read_text())
    names={'Bm25Tool','FinanceAnswerReward','FinanceEpisodeEnv','FinanceRLDataset','CoercingTool'}
    import random, math
    namespace=dict(globals(), Message=dict, Env=object, RLDataset=object,
        ToolResult=object, Tool=object, ToolInput=SimpleNamespace,
        InitialObservationOverflow=SimpleNamespace, tool=lambda f:f, random=random, math=math,
        simple_tool_result=lambda content,should_stop=False:SimpleNamespace(
            messages=[{'role':'tool','content':content}],should_stop=should_stop))
    module=ast.Module(body=[n for n in source.body if isinstance(n,ast.ImportFrom) and n.module=='__future__'
                         or getattr(n,'name',None) in names],type_ignores=[])
    exec(compile(module,'finance_env_business_logic','exec'),namespace)
    return namespace


@patch.dict(os.environ, {'FINANCEBENCH_SCORER':'legacy'})
class EpisodeLogicTests(unittest.TestCase):
    def setUp(self):
        self.classes=load_episode_classes()
        index=hb.StructuredIndex([{'document_id':'D','page':1,'text':'Revenue in 2023 was USD 100 million.'}],[],[])
        self.tools=self.classes['Bm25Tool'](index)
        self.target={'reviewed':True,'answer_type':'numeric','value':'100','unit':'USD','scale':'million','precision':0,
            'support':[{'document_id':'D','page':1,'quote':'Revenue in 2023 was USD 100 million.'}]}

    async def episode(self, **overrides):
        self.tools.state.turn=1
        result=await self.tools.read(document_id='D',page=1)
        payload=json.loads(result.messages[0]['content'])
        self.tools.state.turn=2
        args=dict(answer_type='numeric',value='100',unit='USD',scale='million',
            citations=[{'document_id':'D','page':1,'receipt_id':payload['receipt_id']}])
        args.update(overrides)
        finish=await self.tools.finish(**args)
        reward=self.classes['FinanceAnswerReward'](['100'],episode_state=self.tools.state,target=self.target)
        value,metrics=await reward(result.messages+finish.messages)
        return value,metrics,finish

    def test_live_grounding_does_not_depend_on_tool_call_ids(self):
        value,metrics,finish=asyncio.run(self.episode())
        self.assertEqual((value,metrics['F'],metrics['A'],metrics['G']),(1,1,1,1))
        self.assertTrue(finish.should_stop)

    def test_finish_echo_cannot_validate_fabricated_citation(self):
        value,metrics,_=asyncio.run(self.episode(citations=[{'document_id':'FAKE','page':99,'receipt_id':'r1'}]))
        self.assertEqual(value,0)
        self.assertTrue(metrics['fabricated_citation'])

    def test_wrong_unit_or_value_cannot_receive_answer_credit(self):
        for args in [{'value':'999'},{'unit':'EUR'},{'scale':'billion'}]:
            with self.subTest(args=args):
                self.setUp()
                value,metrics,_=asyncio.run(self.episode(**args))
                self.assertEqual((value,metrics['A']),(0,0))

    def test_attempted_invalid_finish_never_scores(self):
        value,metrics,finish=asyncio.run(self.episode(value='100 or 999'))
        self.assertEqual((value,metrics['F']),(0,0))
        self.assertFalse(finish.should_stop)

    def test_epoch_plan_contains_36_batches(self):
        dataset=self.classes['FinanceRLDataset'](list(range(96)),8,epochs=3)
        self.assertEqual(len(dataset),36)
        for epoch in range(3):
            self.assertEqual(sorted(x for i in range(12) for x in dataset.get_batch(epoch*12+i)),list(range(96)))

    def test_timeout_contributes_zero_finish_metric(self):
        class Inner:
            async def step(self,*args,**kwargs):return SimpleNamespace(episode_done=True,metrics={})
        wrapper=self.classes['FinanceEpisodeEnv'](Inner(),self.tools)
        result=asyncio.run(wrapper.step(None))
        self.assertEqual(result.metrics['F'],0)
        self.assertEqual(result.metrics['used_finish'],0)
        self.assertEqual(result.metrics['finish_missing'],1)

    def test_initial_overflow_contributes_zero_metrics(self):
        class Inner:
            async def initial_observation(self): return SimpleNamespace(metrics={'max_tokens':1})
        wrapper=self.classes['FinanceEpisodeEnv'](Inner(),self.tools)
        result=asyncio.run(wrapper.initial_observation())
        self.assertEqual(result.metrics['F'],0)
        self.assertEqual(result.metrics['finish_missing'],1)

    def test_nonobject_arguments_reach_sdk_validation(self):
        class Wrapped:
            async def run(self,call): return call
            def to_spec(self): return {}
        for value in (None, 1, [1,2]):
            call=SimpleNamespace(arguments=value,call_id='call')
            result=asyncio.run(self.classes['CoercingTool'](Wrapped()).run(call))
            self.assertIs(result,call)

    def test_ambiguous_judge_preserves_unresolved_status(self):
        class Judge:
            async def judge(self,**kwargs):return {'verdict':'ambiguous','confidence':.99}
        state=EpisodeState(accepted={'answer_type':'text','answer_text':'Paper products'})
        reward=self.classes['FinanceAnswerReward'](['Packaging'],question='Industry?',episode_state=state,
            target={'reviewed':True,'answer_type':'text','aliases':['Packaging']},judge=Judge())
        _,metrics=asyncio.run(reward([]))
        self.assertTrue(metrics['unresolved'])

if __name__=='__main__':unittest.main()
