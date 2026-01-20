import sys
from torch import tensor
from transformers import AutoTokenizer
from typing import Dict, Tuple, List, Optional, Iterator, Callable
import warnings
from abc import ABC
from abc import abstractmethod


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
        return self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length if self.max_length is not None else None,
        )

    def get_special_tokens(self):
        return self.special_tokens

    def batch_decode(self, token_ids):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


class NllbTokenizer(HuggingfaceTokenizer):
    def __init__(self, size, max_length=None):
        super().__init__(f"facebook/nllb-200-distilled-{size}", max_length=max_length)


class ByteTokenizer:
    def __init__(self, encoding="utf-8"):
        self.encoding = encoding
        self.special_tokens = {"</s>": 256, "<pad>": 257}

    def __call__(self, sents: List[str], lang_code=None):
        input_ids = []
        max_tokens = 0
        for sent in sents:
            tokens = list(sent.encode(self.encoding))
            tokens.append(self.special_tokens["</s>"])
            max_tokens = max(max_tokens, len(tokens))
            input_ids.append(tokens)
        for i in range(len(input_ids)):
            while len(input_ids[i]) < max_tokens:
                input_ids[i].append(self.special_tokens["<pad>"])
        inputs = {"input_ids": tensor(input_ids)}
        return inputs

    def __len__(self):
        return 256 + len(self.special_tokens)

    def get_special_tokens(self):
        return self.special_tokens
    
class WhiteSpaceTokenizer: # will only work on a string i guess 
    def __init__(self):
        self.special_tokens = {}

    def __call__(self, sent: str, lang_code=None):
        tokens = sent.split()
        inputs = {"input_ids": tokens}
        return inputs

    def __len__(self):
        return 256 + len(self.special_tokens) # too lazy to code this rn 

    def get_special_tokens(self):
        return self.special_tokens
