import unittest

from hybrid_defense import (build_index, document_tokens, lexical_consensus, lexical_scores,
                            ordered, reciprocal_rank_fusion, tokenize)


class HybridTests(unittest.TestCase):
    def test_new_documents_are_eligible(self):
        scores = reciprocal_rank_fusion({"new": 9, "old": 3}, {"new": 1, "old": 0.1})
        self.assertEqual(ordered(scores)[0], "new")

    def test_vector_scale_does_not_dominate(self):
        lexical = {"good": 2, "other": 1}
        first = reciprocal_rank_fusion({"attack": 9, "good": 5}, lexical)
        second = reciprocal_rank_fusion({"attack": 900000, "good": 500000}, lexical)
        self.assertEqual(first, second)
        self.assertEqual(ordered(first)[0], "good")

    def test_lexical_can_supply_missing_dense_candidate(self):
        scores = reciprocal_rank_fusion({"dense": 1}, {"lexical": 1})
        self.assertIn("lexical", scores)

    def test_consensus_rejects_vector_only_document(self):
        self.assertEqual(lexical_consensus({"a": 100, "b": 1}, {"b": 1}), {"b": 1})

    def test_empty_lexical_can_abstain(self):
        self.assertEqual(lexical_consensus({"a": 1}, {}), {})

    def test_index_queries_use_same_normalization(self):
        docs = [{"text": "Alpha treatment"}, {"text": "Beta control"}, {"text": "Gamma sample"}]
        scores = lexical_scores(build_index([document_tokens(doc) for doc in docs]),
                                ["1", "2", "3"], "ALPHA")
        self.assertEqual(ordered(scores)[0], "1")
        self.assertEqual(tokenize("ＡＬＰＨＡ"), ["alpha"])

    def test_bad_settings_fail(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion({}, {}, rank_constant=0)
        with self.assertRaises(ValueError):
            lexical_consensus({}, {}, window=0)


if __name__ == "__main__":
    unittest.main()
