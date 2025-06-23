import random
from typing import Dict, Tuple, List, Optional, Iterator
from torch.utils.data import DataLoader, IterableDataset


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
