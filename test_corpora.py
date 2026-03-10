import json
import unittest
from corpora import (
    Corpus,
    TokenizedCorpus,
    EncipheredCorpus,
    Bitext,
    MixtureOfBitexts,
    CodeswitchedBitext,
    BatchedBitext,
    TokenizedMixtureOfTextAndGoalEncoding,
)
from configure import create_bitexts
from extract_tok_utils import build_fast_align_dict_from_raw
import torch
from torch import tensor
from tokenization import NllbTokenizer
from transformers import AutoModelForSeq2SeqLM


class TestCorporaRevised(unittest.TestCase):

    def test_corpus1(self):
        corpus = Corpus("test_files/lang1.txt")
        corpus_iter = iter(corpus)
        line = next(corpus_iter)
        self.assertEqual(line, "The cat chased the mouse.")
        line = next(corpus_iter)
        self.assertEqual(line, "She reads a book.")
        line = next(corpus_iter)
        self.assertEqual(line, "They play soccer.")

    def test_corpus2(self):
        corpus = Corpus("test_files/lang1.txt")
        counter = 0
        for _ in corpus:
            counter += 1
        self.assertEqual(counter, 20)

    def test_tokenized_corpus(self):
        tokenizer = NllbTokenizer("600M")
        corpus = Corpus("test_files/lang1.txt")
        tokenized_corpus = TokenizedCorpus(corpus, tokenizer, lang_code="eng_Latn")
        corpus_iter = iter(tokenized_corpus)
        tokens = next(corpus_iter)
        self.assertEqual(
            tokens, [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2]
        )
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 11873, 272, 22665, 9, 28487, 248075, 2])
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 13710, 18379, 43583, 2299, 248075, 2])

    def test_enciphered_corpus(self):
        tokenizer = NllbTokenizer("600M")
        corpus = Corpus("test_files/lang1.txt")
        tokenized_corpus = TokenizedCorpus(corpus, tokenizer, lang_code="eng_Latn")
        encipherment_dict = {x: x + 5 for x in range(4, 256000)}
        encipherment = lambda x: encipherment_dict.get(x, x)
        enciphered_corpus = EncipheredCorpus(tokenized_corpus, encipherment)
        corpus_iter = iter(enciphered_corpus)
        tokens = next(corpus_iter)
        self.assertEqual(
            tokens, [256047, 1622, 7880, 233, 55506, 354, 227884, 248080, 2]
        )
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 11878, 277, 22670, 14, 28492, 248080, 2])
        tokens = next(corpus_iter)
        self.assertEqual(tokens, [256047, 13715, 18384, 43588, 2304, 248080, 2])

    def test_bitext(self):
        tokenizer = NllbTokenizer("600M")
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang2.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        corpus_iter = iter(bitext)
        tokens1, tokens2 = next(corpus_iter)
        self.assertEqual(
            tokens1, [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2]
        )
        self.assertEqual(
            tokens2, [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2]
        )
        tokens1, tokens2 = next(corpus_iter)
        self.assertEqual(tokens1, [256047, 11873, 272, 22665, 9, 28487, 248075, 2])
        self.assertEqual(tokens2, [256057, 19945, 6622, 159, 68078, 248075, 2])

    def test_batched_bitext(self):
        tokenizer = NllbTokenizer("600M")
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang2.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        batched_bitext = BatchedBitext(bitext, 4, src_pad_token=0, tgt_pad_token=-100)
        expected1 = tensor(
            [
                [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 0],
                [256047, 13710, 18379, 43583, 2299, 248075, 2, 0, 0],
                [256047, 117, 6337, 109233, 248075, 2, 0, 0, 0],
            ]
        )
        expected2 = tensor(
            [
                [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2],
                [256057, 19945, 6622, 159, 68078, 248075, 2, -100, -100, -100, -100],
                [256057, 21422, 5665, 138, 1166, 96236, 248075, 2, -100, -100, -100],
                [256057, 156, 3, 913, 15931, 1877, 248075, 2, -100, -100, -100],
            ]
        )
        bitext_iter = iter(batched_bitext)
        lang1, lang2 = next(bitext_iter)
        self.assertTrue(torch.equal(lang1["input_ids"], expected1))
        self.assertTrue(torch.equal(lang2["input_ids"], expected2))

    def test_mixture_of_bitexts(self):
        tokenizer = NllbTokenizer("600M")
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang2.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        batched_bitext1 = BatchedBitext(bitext, 4, src_pad_token=0, tgt_pad_token=-100)
        corpus1 = Corpus("test_files/lang1.txt")
        tokenized_corpus1 = TokenizedCorpus(corpus1, tokenizer, lang_code="eng_Latn")
        corpus2 = Corpus("test_files/lang3.txt")
        tokenized_corpus2 = TokenizedCorpus(corpus2, tokenizer, lang_code="fra_Latn")
        bitext = Bitext(tokenized_corpus1, tokenized_corpus2)
        batched_bitext2 = BatchedBitext(bitext, 4, src_pad_token=0, tgt_pad_token=-100)
        bitext1_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "fra_Latn",
        }
        bitext2_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "deu_Latn",
        }
        mix = MixtureOfBitexts(
            {("lang1", "lang2"): batched_bitext1, ("lang1", "lang3"): batched_bitext2},
            {
                ("lang1", "lang2"): bitext1_metadata,
                ("lang1", "lang3"): bitext2_metadata,
            },
            only_once_thru=True,
        )
        counter = 0
        for _ in mix:
            counter += 1
        self.assertEqual(counter, 10)

    def test_mixture_of_bitexts_from_config(self):
        with open("test_files/example_config.json") as f:
            config = json.load(f)
        mix = create_bitexts(config)
        counter = 0
        for _ in mix["dev"]:
            counter += 1
        self.assertEqual(counter, 4)

    def test_mixture_of_bitexts_from_config2(self):
        with open("test_files/example_config.json") as f:
            config = json.load(f)
        mix = create_bitexts(config)
        mix_iter = iter(mix["train"])
        batch1, batch2, metadata = next(mix_iter)
        bitext1_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "fra_Latn",
        }
        bitext2_metadata = {
            "lang1_tokenizer": "nllb",
            "lang1_encipherment": 0,
            "lang1_code": "eng_Latn",
            "lang2_tokenizer": "nllb",
            "lang2_encipherment": 0,
            "lang2_code": "deu_Latn",
        }
        if metadata["lang2_code"] == "fra_Latn":
            expected1 = tensor(
                [
                    [256047, 1617, 7875, 228, 55501, 349, 227879, 248075, 2],
                    [256047, 11873, 272, 22665, 9, 28487, 248075, 2, 0],
                ]
            )
            expected2 = tensor(
                [
                    [256057, 1181, 32779, 9, 170684, 356, 82, 324, 40284, 248075, 2],
                    [
                        256057,
                        19945,
                        6622,
                        159,
                        68078,
                        248075,
                        2,
                        -100,
                        -100,
                        -100,
                        -100,
                    ],
                ]
            )
            self.assertEqual(metadata, bitext1_metadata)
        else:
            expected1 = tensor(
                [
                    [256047, 7007, 158826, 349, 9715, 248075, 2, 0],
                    [256047, 1617, 143207, 43982, 13003, 19836, 248075, 2],
                ]
            )
            expected2 = tensor(
                [
                    [256042, 7007, 2658, 566, 39380, 664, 216976, 2799, 248075, 2],
                    [256042, 6856, 33887, 10184, 24453, 211091, 4108, 77678, 248075, 2],
                ]
            )
            self.assertEqual(metadata, bitext2_metadata)
        self.assertTrue(torch.equal(batch1["input_ids"], expected1))
        self.assertTrue(torch.equal(batch2["input_ids"], expected2))


if __name__ == "__main__":
    unittest.main()
