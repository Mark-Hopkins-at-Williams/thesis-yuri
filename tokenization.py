import sys
from transformers import AutoTokenizer
from typing import Dict, Tuple, List, Optional, Iterator, Callable
import warnings
from abc import ABC
from abc import abstractmethod

import sentencepiece as spm


class Tokenizer(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __call__(self, sents: List[str]):
        pass

    @abstractmethod
    def get_special_tokens(self):
        pass

    @abstractmethod
    def batch_decode(self):
        pass


class HuggingfaceTokenizer(Tokenizer):

    def __init__(self, model_name, max_length=None):
        self.max_length = max_length
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`clean_up_tokenization_spaces` was not set.*",
                category=FutureWarning,
                module="transformers.tokenization_utils_base",
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except OSError:
                sys.stderr.write("Tokenizer not found. Using NLLB tokenizer instead.\n")
                sys.stderr.flush()
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "facebook/nllb-200-distilled-600M"
                )
        self.special_tokens = dict(
            zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)
        )

    def __len__(self):
        return len(self.tokenizer)

    def __call__(self, sents: List[str], lang_code=None):
        if lang_code is not None:
            self.tokenizer.src_lang = lang_code
        result = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length if self.max_length is not None else None,
        )
        return result["input_ids"].squeeze().tolist()

    def get_special_tokens(self):
        return self.special_tokens

    def batch_decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)


class NllbTokenizer(HuggingfaceTokenizer):
    def __init__(self, size, max_length=None):
        super().__init__(f"facebook/nllb-200-distilled-{size}", max_length=max_length)


class SentencePieceTokenizer(Tokenizer):

    def __init__(self, model_dir, max_length=None):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_dir)
        self.max_length = max_length

    def __len__(self):
        return self.sp.get_piece_size()

    def __call__(self, sent: str, lang_code=None):
        if lang_code is not None:
            ids = [self.sp.piece_to_id(lang_code)]
        else:
            ids = []
        ids.extend(self.sp.encode(sent, out_type=int))
        ids.append(self.sp.eos_id())
        return ids

    def get_special_tokens(self):
        return {
            "<unk>": self.sp.unk_id(),
            "</s>": self.sp.eos_id(),
            "<pad>": self.sp.pad_id(),
        }

    def batch_decode(self, token_ids):
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        return self.sp.decode(token_ids)

    def convert_ids_to_tokens(self, ids):
        return [self.sp.id_to_piece(id) for id in ids]
