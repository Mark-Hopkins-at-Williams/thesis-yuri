import random
from typing import Dict, Tuple, List, Optional, Iterator
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from permutations import *
import torch

class Bitext(IterableDataset):
    def __init__(self, lang1_file: str, lang2_file: str):
        self.lang1_file = lang1_file
        self.lang2_file = lang2_file

    def line_streamer(self, file_path: str) -> Iterator[str]:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.rstrip('\n')

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        return zip(self.line_streamer(self.lang1_file), self.line_streamer(self.lang2_file))


class MixtureOfBitexts:
    def __init__(
        self,
        bitexts: Dict[Tuple[str, str], Bitext],
        batch_size: int,
        sampling_probs: Optional[List[float]] = None
    ):
        self.bitexts = bitexts
        self.keys = list(bitexts)
        self.batch_size = batch_size
        self.batch_iters = {}

        for key in self.keys:
            self.batch_iters[key] = self._create_iterator(key)

        total = sum(sampling_probs) if sampling_probs else len(bitexts)
        self.sampling_probs = [p / total for p in (sampling_probs or [1.0] * len(bitexts))]

    def _create_iterator(self, key: Tuple[str, str]) -> Iterator[Tuple[List[str], List[str]]]:
        return iter(DataLoader(
            self.bitexts[key],
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True
        ))

    def next_batch(self) -> Tuple[List[str], List[str], str, str]:
        lang_pair = random.choices(self.keys, weights=self.sampling_probs, k=1)[0]

        try:
            lang1_sents, lang2_sents = next(self.batch_iters[lang_pair])
        except StopIteration:
            self.batch_iters[lang_pair] = self._create_iterator(lang_pair)
            lang1_sents, lang2_sents = next(self.batch_iters[lang_pair])

        return lang1_sents, lang2_sents, lang_pair[0], lang_pair[1]

    @staticmethod
    def create_from_files(
        text_files: Dict[str, str],
        lps: List[Tuple[str, str]],
        batch_size: int,
        sampling_probs: Optional[List[float]] = None
    ) -> 'MixtureOfBitexts':
        bitexts = {(l1, l2): Bitext(text_files[l1], text_files[l2]) for (l1, l2) in lps}
        return MixtureOfBitexts(bitexts, batch_size, sampling_probs)

    def get_language_codes(self) -> List[str]:
        return sorted({code for pair in self.keys for code in pair})


class TokenizedMixtureOfBitexts:
    def __init__(
        self,
        mixture_of_bitexts: MixtureOfBitexts,
        tokenizer: AutoTokenizer,
        max_length: int,
        permutation_map: Optional[Dict] = None
    ):
        self.mixture_of_bitexts = mixture_of_bitexts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.permutation_map = permutation_map

    def _tokenize(self, sents: List[str], lang: str, alt_pad_token: int = None):
        self.tokenizer.src_lang = lang
        tokens = self.tokenizer(
            sents, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        )
        if alt_pad_token is not None:
            tokens.input_ids[tokens.input_ids == self.tokenizer.pad_token_id] = alt_pad_token
        if self.permutation_map is not None:
            if lang in self.permutation_map.keys():
                pmap_lang = self.permutation_map[lang]
                new_tokens = []
                for tokenized_sent in tokens['input_ids']:
                    new_sent = []
                    for token_id in tokenized_sent:
                        new_sent.append(pmap_lang(int(token_id)))
                    new_tokens.append(new_sent)
                tokens['input_ids'] = torch.tensor(new_tokens)
                
        return tokens

    def next_batch(self):
        lang1_sents, lang2_sents, lang1_code, lang2_code = self.mixture_of_bitexts.next_batch()
        lang1_tokenized = self._tokenize(lang1_sents, lang1_code)
        lang2_tokenized = self._tokenize(lang2_sents, lang2_code, alt_pad_token=-100)
        return (lang1_tokenized['input_ids'], 
                lang2_tokenized['input_ids'], 
                lang1_tokenized['attention_mask'],
                lang2_tokenized['attention_mask'])

    def get_language_codes(self) -> List[str]:
        return self.mixture_of_bitexts.get_language_codes


text_files = {"eng_Latn": "test_files/lang1.txt", "fra_Latn": "test_files/lang2.txt"}

mix = MixtureOfBitexts.create_from_files(text_files, [("eng_Latn", "fra_Latn")], 3)

base_model = "facebook/nllb-200-distilled-600M"; tokenizer = AutoTokenizer.from_pretrained(base_model)
 
tokenizer1 = AutoTokenizer.from_pretrained(base_model); tokenizer1.src_lang = "eng_Latn"; eng_vocab = tokenizer1.vocab_size

tokenizer2 = AutoTokenizer.from_pretrained(base_model); tokenizer2.src_lang = "fra_Latn"; fra_vocab = tokenizer2.vocab_size

pmap = {"eng_Latn": CreateRandomPermutationWithFixedPoints(eng_vocab, []), "fra_Latn": CreateRandomPermutationWithFixedPoints(fra_vocab, [])}

tmob = TokenizedMixtureOfBitexts(mix, tokenizer, max_length=128, permutation_map=pmap)

print(tmob.next_batch())

