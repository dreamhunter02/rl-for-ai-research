import unittest

from review_targets_v2 import repair_text_support
import financebench_harness as hb


class TextSupportRepairTests(unittest.TestCase):
    def target(self, quote, count=1):
        return {'support':[{'document_id':'D','page':1,'quote':quote,'claim':'answer'} for _ in range(count)]}

    def test_acquisitions_split_into_three_bounded_claims(self):
        text='On August 1, 2022 first. On March 17, 2023 second. On May 31, 2023 third.'
        target=self.target(text)
        repair_text_support('financebench_id_01079',target)
        self.assertEqual(len(target['support']),3)
        self.assertEqual([s['claim'] for s in target['support']],['acquisition_1','acquisition_2','acquisition_3'])

    def test_credit_facilities_split_into_two_bounded_claims(self):
        start='On May 26, 2023, PepsiCo entered into a new $4,200,000,000'
        text=f'{start} first agreement. PepsiCo may also amend. {start} second agreement. PepsiCo may also amend.'
        target=self.target(text)
        repair_text_support('financebench_id_00882',target)
        self.assertEqual(len(target['support']),2)
        self.assertTrue(all(len(s['quote'])<=hb.MAX_READ for s in target['support']))

    def test_multiple_supports_become_distinct_required_claims(self):
        target=self.target('short evidence',2)
        repair_text_support('financebench_id_other',target)
        self.assertEqual([s['claim'] for s in target['support']],['answer_evidence_1','answer_evidence_2'])

    def test_boeing_customer_context_is_optional_and_evidence_has_alternatives(self):
        target={'support':[{'document_id':'D','page':page,'quote':'short','claim':'answer'} for page in (8,10,14)]}
        repair_text_support('financebench_id_01290',target)
        self.assertEqual(len(target['required_facts']),1)
        self.assertEqual({s['claim'] for s in target['support']},{'commercial_airlines','us_government'})
        self.assertEqual({s['page'] for s in target['support']},{3,8,10,40})

    def test_oversized_unspecialized_support_fails_closed(self):
        target=self.target('x'*(hb.MAX_READ+1))
        with self.assertRaises(ValueError):
            repair_text_support('financebench_id_other',target)


if __name__=='__main__':
    unittest.main()
