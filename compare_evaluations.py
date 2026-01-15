from configure import harvest_language_codes
from configure import initialize_tokenizer
from configure import load_tokenized_bitexts
from configure import USE_CUDA
import json
import math
from myutil import cleanup
from myutil import logger
import torch
from transformers import AutoModelForSeq2SeqLM
from validate import evaluate_model

from tokenization import ByteTokenizer
from tokenization import NllbTokenizer
from unigram import compute_unigram_entropy
from unigram import load_unigram_distribution


def compute_conditional_entropy(tmob, model):
    model.eval()
    next_batch = tmob.next_batch()
    total_entropy = 0.0
    while next_batch is not None:
        cleanup()
        x, y, _, _ = next_batch
        x = x.to(model.device)
        y = y.to(model.device)
        model_output = model(**x, labels=y.input_ids)
        logits = model_output.logits
        probs = torch.softmax(logits, dim=-1)
        y_ids = y.input_ids
        y_ids = (y_ids != -100).int() * y_ids + (
            y_ids == -100
        ).int() * 2  # if -100 padding exists, turn them into regular pad tokens (id=2)
        assigned = probs.gather(
            dim=-1,
            index=y_ids.unsqueeze(-1),
        ).squeeze(-1)
        assigned[:, 0] = 1.0
        neg_log_probs = -torch.log2(assigned) * (
            y["attention_mask"].int()
        )  # remove pad tokens from consideration
        entropy = torch.sum(neg_log_probs)
        total_entropy += entropy.item()
        next_batch = tmob.next_batch()
    return total_entropy


def compute_target_side_unigram_entropy(tmob, unigram_distribution):
    entropy = 0.0
    next_batch = tmob.next_batch()
    while next_batch is not None:
        x, y, _, _ = next_batch
        token_ids, token_counts = torch.unique(y["input_ids"], return_counts=True)
        for tok, count in zip(token_ids.tolist(), token_counts.tolist()):
            try:
                token_prob = unigram_distribution(tok)
                entropy += -count * math.log2(token_prob)
            except:
                pass
        next_batch = tmob.next_batch()
    return entropy


if __name__ == "__main__":
    logger("initializing model...")
    model_name = "facebook/nllb-200-distilled-600M"
    config_file = "data/simple.json"
    with open(config_file) as reader:
        config = json.load(reader)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if USE_CUDA:
        torch.cuda.set_device(0)
        model.cuda()
    lang_codes = harvest_language_codes(config)
    tokenizer = initialize_tokenizer(config)

    logger("computing conditional entropy")
    bitexts = load_tokenized_bitexts(config, use_alt_pad_token_for_tgt_lang=False)
    ce = compute_conditional_entropy(bitexts["test"], model)
    logger("computing target side unigram entropy")
    bitexts = load_tokenized_bitexts(config, use_alt_pad_token_for_tgt_lang=False)
    nllb_dist = load_unigram_distribution(f"unigram_lms/eng_Latn.nllb.unigram_lm.json")
    nllb_entropy = compute_target_side_unigram_entropy(bitexts["test"], nllb_dist)
    byte_bitexts = load_tokenized_bitexts(
        config, ByteTokenizer(), use_alt_pad_token_for_tgt_lang=False
    )
    byte_dist = load_unigram_distribution(f"unigram_lms/eng_Latn.byte.unigram_lm.json")
    byte_entropy = compute_target_side_unigram_entropy(byte_bitexts["test"], byte_dist)

    logger(f"conditional entropy: {ce}")
    logger(f"target side unigram entropy (nllb): {nllb_entropy}")
    logger(f"target side unigram entropy (byte): {byte_entropy}")
