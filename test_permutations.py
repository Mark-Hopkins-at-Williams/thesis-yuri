import unittest
from corpora import Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
from finetune import tokenize, prepare_tokenizer_and_model
from transformers import AutoTokenizer
from torch import tensor
from permutations import *


#WILL ADD A FEW TESTS SOON
