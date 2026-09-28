"""Offline reproductions of defects reviewed at upstream f29df1a."""
import unittest
import json
import tempfile
from pathlib import Path
import financebench_harness as hb


class ReviewRegressions(unittest.TestCase):
    def test_wrong_scalar_answers_do_not_match(self):
        for gold, candidate in [('3.2%', '4.1%'), ('0.05', '0.06'),
                                ('$(1,234) million', '$1,234 million'),
                                ('$1577.00', '1,200, 1,577 and 1,890')]:
            with self.subTest(gold=gold, candidate=candidate):
                self.assertEqual(hb.score_answer(gold, candidate)[0], 0)

    def test_plural_scale_matches(self):
        self.assertEqual(hb.score_answer('$1577 million', '$1,577 millions')[0], 1)

    def test_empty_call_id_does_not_erase_visible_evidence(self):
        trace = [
            {'role': 'assistant', 'tool_calls': [{'name': 'read', 'arguments': {}, 'call_id': ''}]},
            {'role': 'tool', 'name': 'read', 'call_id': '',
             'content': '{"document_id":"D","page":1,"text":"Revenue was $1577 million."}'}]
        strong, weak = hb._trace_evidence_text(trace)
        self.assertIn('Revenue was $1577 million.', strong)
        self.assertEqual(weak, '')

    def test_default_read_continues_past_8000(self):
        index = hb.StructuredIndex([{'document_id':'D','page':1,'text':'x'*10000}], [], [])
        result = index.read('D', page=1, start=8000)
        self.assertTrue(result['text'])
        self.assertGreater(result['end'], 8000)

    def test_prepare_preserves_recorded_split_ids(self):
        from workshop_prepare import prepare
        rows=[{'financebench_id':str(i),'doc_name':'D','question':'Value?','answer':'1'} for i in range(5)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'source.json').write_text(json.dumps({'train':rows[:4],'eval':rows[4:]}))
            manifest={'train96_ids':['0','2','3'],'dev12_ids':['1'],'eval42_ids':['4']}
            (root/'ids.json').write_text(json.dumps(manifest))
            prepare(root/'source.json',root/'out',split_manifest=root/'ids.json')
            result=json.loads((root/'out/split.json').read_text())
            self.assertEqual([r['financebench_id'] for r in result['train']],manifest['train96_ids'])
            self.assertEqual([r['financebench_id'] for r in result['dev']],manifest['dev12_ids'])

    def test_company_filter_accepts_spaced_company_name(self):
        index=hb.StructuredIndex([],[],[])
        self.assertTrue(index._allowed({'company':'Americanexpress'}, {'company':'American Express'}))

if __name__ == '__main__': unittest.main()
