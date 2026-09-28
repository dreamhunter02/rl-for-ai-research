"""Real SDK contract tests. Explicitly skipped if the pinned cookbook is absent."""
import asyncio
import importlib.util
import json
import unittest

SDK_AVAILABLE = importlib.util.find_spec('tinker_cookbook') is not None


@unittest.skipUnless(SDK_AVAILABLE, 'requires pinned tinker-cookbook; see requirements-workshop.txt')
class FinanceEnvTests(unittest.TestCase):
    def test_nonobject_arguments_return_recoverable_error(self):
        from finance_env import Bm25Tool, CoercingTool
        from tinker_cookbook.tool_use.types import ToolInput
        for value in (None, 1, [1,2]):
            result=asyncio.run(CoercingTool(Bm25Tool(None).finish).run(ToolInput(arguments=value)))
            self.assertFalse(result.should_stop)

    def test_finish_terminal_and_validation_recoverable(self):
        from finance_env import Bm25Tool, CoercingTool
        from tinker_cookbook.tool_use.types import ToolInput
        tools = Bm25Tool(None)
        finish = CoercingTool(tools.finish)
        bad = asyncio.run(finish.run(ToolInput(arguments={'answer_type':'numeric','value':'1 or 2','unit':'USD','scale':'ones'})))
        self.assertFalse(bad.should_stop)
        good = asyncio.run(finish.run(ToolInput(arguments={'answer_type':'numeric','value':'2','unit':'USD','scale':'ones'})))
        self.assertTrue(good.should_stop)
        self.assertEqual(tools.state.accepted['value'],'2')
        self.assertIsNone(Bm25Tool(None).state.accepted)

    def test_schema_list_coercion_updates_arguments_only(self):
        from finance_env import CoercingTool
        from tinker_cookbook.tool_use.types import ToolInput
        class Fake:
            name='example'
            def to_spec(self): return {'parameters':{'properties':{'items':{'type':'array'},'text':{'type':'string'}}}}
            async def run(self,call): return call
        call=ToolInput(arguments={'items':'["one"]','text':'["leave as text"]'},call_id='id')
        result=asyncio.run(CoercingTool(Fake()).run(call))
        self.assertEqual(result.arguments['items'],['one'])
        self.assertEqual(result.arguments['text'],'["leave as text"]')
        self.assertEqual(call.arguments['items'],'["one"]')
        self.assertEqual(result.call_id,'id')

    def test_remainder_batches_and_epochs(self):
        from finance_env import FinanceRLDataset
        dataset=FinanceRLDataset(list(range(10)),4,epochs=3,seed=0)
        self.assertEqual(len(dataset),9)
        for epoch in range(3):
            self.assertEqual(sorted(v for i in range(3) for v in dataset.get_batch(epoch*3+i)),list(range(10)))
        with self.assertRaises(IndexError): dataset.get_batch(9)

    def test_read_receipt_matches_visible_text(self):
        from finance_env import Bm25Tool, CoercingTool
        from financebench_harness import StructuredIndex
        from tinker_cookbook.tool_use.types import ToolInput
        index=StructuredIndex([{'document_id':'D','page':1,'text':'"\\\n'*10000}],[],[])
        tools=Bm25Tool(index);tools.state.turn=7
        result=asyncio.run(CoercingTool(tools.read).run(ToolInput(arguments={'document_id':'D','page':1})))
        payload=json.loads(result.messages[0]['content'])
        self.assertLessEqual(len(result.messages[0]['content']),4000)
        self.assertEqual(payload['turns_remaining'],1)
        self.assertEqual(payload['text'],tools.state.receipts[payload['receipt_id']]['text'])

    def test_triple_finish_cannot_restore_acceptance(self):
        from finance_env import Bm25Tool
        from tinker_cookbook.tool_use.types import ToolInput
        tools=Bm25Tool(None)
        call=ToolInput(arguments={'answer_type':'numeric','value':'2','unit':'USD','scale':'ones'})
        for _ in range(3): asyncio.run(tools.finish.run(call))
        self.assertIsNone(tools.state.accepted)

    def test_judge_receives_alias_when_required_facts_empty(self):
        from finance_env import FinanceAnswerReward
        from workshop_reward import EpisodeState
        class Judge:
            async def judge(self,**kwargs):
                self.gold=kwargs['gold'];return {'verdict':'contradicted','confidence':1,'numeric_ok':True}
        state=EpisodeState(accepted={'answer_type':'text','answer_text':'Oil'})
        judge=Judge()
        reward=FinanceAnswerReward(['Packaging'],question='Industry?',episode_state=state,
            target={'reviewed':True,'answer_type':'text','aliases':['Packaging'],'required_facts':[]},judge=judge)
        asyncio.run(reward([]))
        self.assertIn('Packaging',judge.gold)

    def test_dev_evaluator_accepts_prebuilt_dataset(self):
        from finance_env import FinanceRLDataset
        from train_financebench import tinker_evaluator_builder
        class Builder:
            def logging_tags(self): return ["test"]
        dataset=FinanceRLDataset([Builder()],1)
        evaluator=tinker_evaluator_builder(dataset,1024)()
        self.assertEqual(evaluator.max_tokens,1024)
        self.assertEqual(len(evaluator.env_group_builders_P),1)
