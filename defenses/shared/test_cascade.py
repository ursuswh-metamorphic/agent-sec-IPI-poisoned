import unittest

from cascade_defense import budgeted_fusion, reverse_consensus, subset_bm25
from hybrid_defense import build_index, ordered, tokenize


class CascadeTests(unittest.TestCase):
    def test_reverse_limits_dense_candidates(self):
        result = reverse_consensus({"a": 3, "b": 2, "c": 1}, {"a": 1, "b": 2, "c": 99}, 2)
        self.assertEqual(ordered(result), ["b", "a"])

    def test_reverse_zero_signal_uses_dense(self):
        result = reverse_consensus({"a": 3, "b": 2}, {"a": 0, "b": 0})
        self.assertEqual(ordered(result), ["a", "b"])

    def test_budgeted_fallback_keeps_candidates(self):
        dense = {"a": 3, "b": 2}
        self.assertEqual(budgeted_fusion(dense, {"a": 0, "b": 0}), dense)

    def test_budgeted_prefers_lexical_supported_candidate(self):
        result = budgeted_fusion({"a": 9999, "b": 2}, {"a": 0, "b": 1})
        self.assertEqual(ordered(result)[0], "b")

    def test_subset_matches_full_corpus_bm25(self):
        texts = ["alpha treatment", "beta control", "gamma sample", "delta results"]
        index = build_index([tokenize(t) for t in texts])
        result = subset_bm25(index, {"a": 0, "b": 1, "c": 2, "d": 3}, "alpha", ["a", "c"])
        full = index.get_scores(tokenize("alpha"))
        self.assertAlmostEqual(result["a"], full[0])
        self.assertAlmostEqual(result["c"], full[2])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            budgeted_fusion({}, {}, lexical_weight=2)


if __name__ == "__main__":
    unittest.main()
