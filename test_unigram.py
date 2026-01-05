import math
from tokenization import NllbTokenizer
from torch import tensor
from unigram import collect_unigram_counts
from unigram import compute_unigram_entropy
from unigram import create_unigram_distribution_from_counts
import unittest


class TestUnigram(unittest.TestCase):

    def test_unigram_distribution1(self):
        tokenizer = NllbTokenizer("600M")
        token_counts = collect_unigram_counts(
            "test_files/small.txt", tokenizer, show_progress=False
        )
        expected = {
            256047: 2,
            1: 3,
            1617: 1,
            7875: 2,
            1398: 1,
            7531: 1,
            248075: 2,
            2: 2,
            117: 1,
            4843: 1,
            281: 1,
            349: 1,
            61154: 1,
            5472: 1,
        }
        self.assertEqual(token_counts, expected)

    def test_unigram_distribution_from_counts1(self):
        tokenizer = NllbTokenizer("600M")
        token_counts = collect_unigram_counts(
            "test_files/small.txt", tokenizer, show_progress=False
        )
        unigram_dist = create_unigram_distribution_from_counts(
            token_counts,
            len(tokenizer),
            {0, 1, 3} | set(range(256001, len(tokenizer))),
            k_smoother=0,
        )
        self.assertAlmostEqual(unigram_dist(7875), 2 / 15)
        self.assertAlmostEqual(unigram_dist(1398), 1 / 15)
        with self.assertRaises(KeyError):
            unigram_dist(256047)
        with self.assertRaises(KeyError):
            unigram_dist(0)

    def test_unigram_distribution_from_counts2(self):
        tokenizer = NllbTokenizer("600M")
        token_counts = collect_unigram_counts(
            "test_files/small.txt", tokenizer, show_progress=False
        )
        unigram_dist = create_unigram_distribution_from_counts(
            token_counts,
            len(tokenizer),
            {0, 1, 3} | set(range(256001, len(tokenizer))),
            k_smoother=1,
        )
        self.assertAlmostEqual(unigram_dist(7875), 3 / (15 + 255997))
        self.assertAlmostEqual(unigram_dist(1398), 2 / (15 + 255997))
        self.assertAlmostEqual(unigram_dist(10000), 1 / (15 + 255997))
        with self.assertRaises(KeyError):
            unigram_dist(256047)
        with self.assertRaises(KeyError):
            unigram_dist(0)

    def test_unigram_entropy(self):
        tokenizer = NllbTokenizer("600M")
        token_counts = collect_unigram_counts(
            "test_files/small.txt", tokenizer, show_progress=False
        )
        unigram_dist = create_unigram_distribution_from_counts(
            token_counts,
            len(tokenizer),
            {0, 1, 3} | set(range(256001, len(tokenizer))),
            k_smoother=0,
        )
        lines = ["The cat"]
        result = compute_unigram_entropy(lines, tokenizer, unigram_dist)
        expected = -math.log2(2 / 15) - math.log2(1 / 15) - math.log2(2 / 15)
        self.assertAlmostEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
