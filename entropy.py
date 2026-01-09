from corpora import Bitext
from corpora import MixtureOfBitexts
from corpora import TokenizedMixtureOfBitexts
from myutil import cleanup
from tokenization import HuggingfaceTokenizer
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM
from myutil import logger


def compute_conditional_entropy(
    src_lang, src_path, tgt_lang, tgt_path, base_model, batch_size=1
):
    # doesn't deal with batches (pad tokens) yet
    bitext = Bitext(src_path, tgt_path)
    mix = MixtureOfBitexts(
        {(("test", src_lang), ("test", tgt_lang)): bitext},
        batch_size=batch_size,
        only_once_thru=True,
    )
    lang_codes = {("test", src_lang): src_lang, ("test", tgt_lang): tgt_lang}
    tokenizer = HuggingfaceTokenizer(base_model)
    tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
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
        assigned = probs.gather(
            dim=-1,
            index=y.input_ids.unsqueeze(-1),
        ).squeeze(-1)
        assigned[-1][0] = 1.0
        neg_log_probs = -torch.log2(assigned)
        entropy = torch.sum(neg_log_probs)
        total_entropy += entropy.item()
        next_batch = tmob.next_batch()
    return total_entropy


def compute_unconditional_entropy(
    src_lang, src_path, tgt_lang, tgt_path, base_model, batch_size=1
):
    # doesn't deal with batches (pad tokens) yet
    bitext = Bitext(src_path, tgt_path)
    mix = MixtureOfBitexts(
        {(("test", src_lang), ("test", tgt_lang)): bitext},
        batch_size=batch_size,
        only_once_thru=True,
    )
    lang_codes = {("test", src_lang): src_lang, ("test", tgt_lang): tgt_lang}
    tokenizer = HuggingfaceTokenizer(base_model)
    tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    next_batch = tmob.next_batch()
    total_entropy = 0.0
    while next_batch is not None:
        cleanup()
        _, y, _, _ = next_batch
        y = y.to(model.device)
        x = {
            "input_ids": torch.tensor([[2]] * batch_size).to(device),
            "attention_mask": torch.tensor([[1]] * batch_size).to(device),
        }
        model_output = model(**x, labels=y.input_ids)
        logits = model_output.logits
        probs = torch.softmax(logits, dim=-1)
        assigned = probs.gather(
            dim=-1,
            index=y.input_ids.unsqueeze(-1),
        ).squeeze(-1)
        assigned[-1][0] = 1.0
        neg_log_probs = -torch.log2(assigned)
        entropy = torch.sum(neg_log_probs)
        total_entropy += entropy.item()
        next_batch = tmob.next_batch()
    return total_entropy


if __name__ == "__main__":
    base_model = "facebook/nllb-200-distilled-600M"
    with open("/mnt/storage/hopkins/data/flores/lang_codes") as reader:
        lang_codes = [line.strip() for line in reader if line.strip() != "eng_Latn"]

    lang_codes = ["zho_Hant"]

    for lang in tqdm(lang_codes):
        src_path = "/mnt/storage/hopkins/data/flores/dev.eng_Latn"
        tgt_path = f"/mnt/storage/hopkins/data/flores/dev.{lang}"

        ce = compute_conditional_entropy(
            "eng_Latn", src_path, lang, tgt_path, base_model, batch_size=1
        )
        print(f"{lang},{ce}")
