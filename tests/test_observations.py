import json
import unittest
from observations import bounded_observation, compact_hits
from financebench_harness import StructuredIndex


class ObservationTests(unittest.TestCase):
    def test_compact_search_excludes_full_page(self):
        hits = [{"document_id": "D", "page": i, "text": "padding " * 50000 + " revenue 123", "_length": 9, "score": 1} for i in range(8)]
        out = compact_hits(hits, ["revenue"], limit=5)
        self.assertEqual(len(out), 5)
        self.assertTrue(all("text" not in x and "_length" not in x for x in out))
        self.assertIn("revenue", out[0]["snippet"])
        self.assertLess(len(bounded_observation({"hits": out})), 4001)

    def test_escaping_stays_within_serialized_cap(self):
        raw = bounded_observation({"text": '"\\\n' * 10000, "document_id": "D", "page": 1, "start": 0, "end": 30000, "total_chars": 30000})
        self.assertLessEqual(len(raw), 4000)
        obj = json.loads(raw)
        self.assertEqual(obj["next_start"], len(obj["text"]))
        self.assertTrue(obj["has_more"])

    def test_reads_are_page_scoped_and_pageable(self):
        index = StructuredIndex([{"document_id": "D", "page": 1, "text": "a" * 10000},
                                 {"document_id": "D", "page": 2, "text": "b" * 10000}], [], [])
        first = index.read("D", 1, 0, 8000)
        self.assertEqual(first["page_start"], first["page_end"])
        self.assertTrue(first["has_more"])
        second = index.read("D", 1, first["next_start"], 10000)
        self.assertEqual(second["start"], first["end"])
        self.assertEqual(index.read("D", -1)["page_end"], 1)

    def test_cross_document_passage_rejected(self):
        index = StructuredIndex([], [{"document_id": "D", "page": 1, "passage_id": "p", "text": "abc"}], [])
        self.assertIn("error", index.read("OTHER", passage_id="p"))

    def test_default_continuation_does_not_stall(self):
        page={'document_id':'D','page':1,'text':'x'*5000}
        index=StructuredIndex([page],[],[{**page,'table_id':'t'}])
        for read in (lambda start:index.read('D',1,start=start),lambda start:index.read_table('t',start=start)):
            first=read(0);second=read(first['next_start'])
            self.assertGreater(second['end'],first['end'])
            self.assertTrue(second['text'])
