import unittest
from workshop_reward import numeric_equal, validate_submission, EpisodeState, score_submission


class NumericTests(unittest.TestCase):
    def test_adversarial_numeric_pairs(self):
        cases = [
            ("3.2%", "4.1%", False), ("-5.0%", "5.0%", False),
            ("0.05", "0.06", False), ("$(1,234) million", "$1,234 million", False),
            ("$1577.00", "1,200, 1,577 and 1,890", False),
            ("$1577 million", "$1,577 millions", True),
            ("4.2 million", "4.2 billion", False), ("USD 100", "EUR 100", False),
            ("0.00%", "0.4%", False), ("$2000", "$2001", False),
            ("1.2", "1.249", True), ("1.2", "1.25", False),
            ("4.2 billion", "4200 million", True), ("-0.05", "0.05", False),
        ]
        for gold, candidate, expected in cases:
            with self.subTest(gold=gold, candidate=candidate):
                self.assertEqual(numeric_equal(gold, candidate), expected)

    def test_strict_scalar(self):
        for value in ["NaN", "Infinity", "1 or 2", ""]:
            with self.assertRaises(ValueError):
                validate_submission({"answer_type": "numeric", "value": value, "unit": "USD", "scale": "ones"})

    def test_unsupported_answer_type_rejected(self):
        with self.assertRaises(ValueError):
            validate_submission({"answer_type": "guess", "answer_text": "hello"})


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.state = EpisodeState()
        self.receipt = self.state.record("read", {"document_id": "D", "page": 2,
            "text": "Revenue in 2023 was USD 100 million. Profit in 2023 was USD 20 million."})
        self.state.turn = 1
        self.target = {"answer_type": "numeric", "value": "20", "unit": "percent", "scale": "ones",
            "precision": 1, "derived": True, "reviewed": True, "support": [],
            "expression": "profit/revenue*100", "operands": {
                "profit": {"value": "20", "unit": "USD", "scale": "million", "metric": "Profit", "period": "2023", "support": [{"document_id":"D","page":2,"quote":"Profit in 2023 was USD 20 million."}]},
                "revenue": {"value": "100", "unit": "USD", "scale": "million", "metric": "Revenue", "period": "2023", "support": [{"document_id":"D","page":2,"quote":"Revenue in 2023 was USD 100 million."}]}}}

    def test_calculator_constant_is_not_provenance(self):
        calc = self.state.calculate("20", {})
        self.assertFalse(calc["provenance_valid"])

    def test_operands_must_appear_in_delivered_quote(self):
        with self.assertRaises(ValueError):
            self.state.calculate("profit/revenue*100", {
                "profit": {"value": "99", "receipt_id": self.receipt, "quote": "Profit in 2023 was USD 99 million."},
                "revenue": {"value": "100", "receipt_id": self.receipt, "quote": "Revenue in 2023 was USD 100 million."}})

    def test_valid_calculation_and_citation(self):
        calc = self.state.calculate("profit/revenue*100", {
            "profit": {"value": "20", "receipt_id": self.receipt, "quote": "Profit in 2023 was USD 20 million.", "metric": "Profit", "period": "2023", "unit": "USD", "scale": "million"},
            "revenue": {"value": "100", "receipt_id": self.receipt, "quote": "Revenue in 2023 was USD 100 million.", "metric": "Revenue", "period": "2023", "unit": "USD", "scale": "million"}})
        answer = {"answer_type": "numeric", "value": "20", "unit": "percent", "scale": "ones",
            "calc_id": calc["calc_id"], "citations": [{"receipt_id": self.receipt, "document_id": "D", "page": 2}]}
        self.state.turn += 1
        result = score_submission(self.target, answer, self.state)
        self.assertEqual(result["A"], 1)
        self.assertEqual(result["G"], 1)
        self.assertEqual(result["reward"], 1)

    def test_fabricated_citation_veto(self):
        answer = {"answer_type": "numeric", "value": "20", "unit": "percent", "scale": "ones",
            "citations": [{"receipt_id": "fake", "document_id": "D", "page": 2}]}
        result = score_submission(self.target, answer, self.state)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["A"], 1)

    def test_state_isolation(self):
        self.assertFalse(EpisodeState().receipts)

    def test_unreviewed_target_fails_closed(self):
        with self.assertRaises(ValueError):
            score_submission({**self.target, "reviewed": False}, {}, self.state)

    def test_text_overlap_is_not_verified(self):
        result = score_submission({"answer_type": "text", "reviewed": True, "aliases": ["Packaging industry"]},
            {"answer_type": "text", "answer_text": "Packaging industry and oil exploration"}, self.state)
        self.assertTrue(result["unresolved"])


class ProvenanceAttackTests(unittest.TestCase):
    def test_accounting_negative_cannot_support_positive_operand(self):
        state=EpisodeState()
        receipt=state.record('read',{'document_id':'D','page':1,'text':'Profit in 2023 was USD (100) million.'})
        state.turn=1
        with self.assertRaises(ValueError):
            state.calculate('x*1',{'x':{'value':'100','unit':'USD','scale':'million','metric':'Profit','period':'2023','receipt_id':receipt,'quote':'Profit in 2023 was USD (100) million.'}})

    def test_same_turn_read_is_not_yet_visible(self):
        state=EpisodeState()
        receipt=state.record('read',{'document_id':'D','page':1,'text':'Profit in 2023 was USD 100 million.'})
        with self.assertRaises(ValueError):
            state.calculate('x*1',{'x':{'value':'100','unit':'USD','scale':'million','metric':'Profit','period':'2023','receipt_id':receipt,'quote':'Profit in 2023 was USD 100 million.'}})

class ReviewRegressionTests(unittest.TestCase):
    def test_unhashable_citation_ids_fail_validation(self):
        target={'reviewed':True,'answer_type':'numeric','value':'1','unit':'USD','scale':'ones','precision':0}
        for field in ('receipt_id','document_id'):
            for bad in (['r1'], {'x':'r1'}, 7, True):
                citation={'receipt_id':'r1','document_id':'D','page':1,field:bad}
                answer={'answer_type':'numeric','value':'1','unit':'USD','scale':'ones','citations':[citation]}
                with self.subTest(field=field,bad=bad):
                    self.assertEqual(score_submission(target,answer,EpisodeState())['F'],0)

    def test_derived_answer_scale_conversion_preserves_grounding(self):
        state=EpisodeState()
        quote='Revenue in 2023 was USD 100 million.'
        receipt=state.record('read',{'document_id':'D','page':1,'text':quote})
        state.turn=1
        spec={'value':'100','unit':'USD','scale':'million','metric':'Revenue','period':'2023',
            'support':[{'document_id':'D','page':1,'quote':quote}]}
        calc=state.calculate('revenue*1',{'revenue':{**spec,'receipt_id':receipt,'quote':quote}})
        state.turn=2
        target={'reviewed':True,'answer_type':'numeric','value':'100','unit':'USD','scale':'million','precision':0,
            'derived':True,'expression':'revenue*1','operands':{'revenue':spec}}
        answer={'answer_type':'numeric','value':'100000000','unit':'USD','scale':'ones','calc_id':calc['calc_id'],
            'citations':[{'receipt_id':receipt,'document_id':'D','page':1}]}
        result=score_submission(target,answer,state)
        self.assertEqual((result['A'],result['G'],result['reward']),(1,1,1))

    def test_reviewed_average_can_use_divisor_two(self):
        state=EpisodeState()
        receipt=state.record('read',{'document_id':'D','page':1,'text':'Assets in 2022 were 100. Assets in 2023 were 200.'})
        state.turn=1
        operands={name:{'value':value,'unit':'USD','scale':'ones','metric':'Assets','period':year,
            'receipt_id':receipt,'quote':f'Assets in {year} were {value}.'}
            for name,value,year in [('prior','100','2022'),('current','200','2023')]}
        calc=state.calculate('(prior+current)/2',operands)
        self.assertEqual(calc['value'],'150')
        self.assertTrue(calc['provenance_valid'])

    def test_large_finite_submission_scores_zero_without_exception(self):
        target={'reviewed':True,'answer_type':'numeric','value':'1','unit':'USD','scale':'ones','precision':2}
        answer={'answer_type':'numeric','value':'1000000000000000000000000000000','unit':'USD','scale':'ones'}
        self.assertEqual(score_submission(target,answer,EpisodeState())['A'],0)

    def test_calculation_must_be_visible_before_finish(self):
        fixture=ProvenanceTests();fixture.setUp();state=fixture.state
        operands={name:{**op,'receipt_id':fixture.receipt,'quote':op['support'][0]['quote']} for name,op in fixture.target['operands'].items()}
        calc=state.calculate('profit/revenue*100',operands)
        answer={'answer_type':'numeric','value':'20','unit':'percent','scale':'ones','calc_id':calc['calc_id'],
                'citations':[{'receipt_id':fixture.receipt,'document_id':'D','page':2}]}
        self.assertEqual(score_submission(fixture.target,answer,state)['G'],0)
        state.turn+=1
        self.assertEqual(score_submission(fixture.target,answer,state)['G'],1)
