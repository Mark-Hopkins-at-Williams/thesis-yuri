import random
from typing import Dict, Tuple, List, Optional, Iterator
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, NllbTokenizerFast
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
        mixbitext: MixtureOfBitexts,
        tokenizer: NllbTokenizerFast,
        max_length: int
    ):
        self.mixbitext = mixbitext
        self.tokenizer = tokenizer
        self.max_length = max_length
    def next_batch(self):
        nxtbatch = self.mixbitext.next_batch()
        tokenizer_lang1 = self.tokenizer
        tokenizer_lang1.src_lang = nxtbatch[2]
        lang1_sents = list(nxtbatch[0])
        lang2_sents = list(nxtbatch[1])
        lang1_tokenized = tokenizer_lang1(lang1_sents, padding=True, max_length=self.max_length, truncation = True)
        lang1_input_id = torch.tensor(lang1_tokenized['input_ids'])
        lang1_attention_mask = torch.tensor(lang1_tokenized['attention_mask'])
        tokenizer_lang2 = self.tokenizer
        tokenizer_lang2.src_lang = nxtbatch[3]
        lang2_tokenized = tokenizer_lang2(lang2_sents, padding=True, max_length=self.max_length, truncation = True)
        lang2_input_id = torch.where(torch.tensor(lang2_tokenized['input_ids']) == 1, torch.tensor(-100), torch.tensor(lang2_tokenized['input_ids']))
        lang2_attention_mask = torch.tensor(lang2_tokenized['attention_mask'])
        result =(lang1_input_id, lang2_input_id, lang1_attention_mask, lang2_attention_mask)
        return result
    

