import json
import unittest
from corpora import load_tokenizer, Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
from torch import tensor
from transformers import AutoModelForSeq2SeqLM

def to_eng(lang_code):
    src_path = "/mnt/storage/hopkins/data/nllb/seed/seed/" + lang_code
    lang_prefix = lang_code[:3]
    bitext = Bitext(src_path, "/mnt/storage/hopkins/data/nllb/seed/seed/eng_Latn")
    mix = MixtureOfBitexts({(("test", lang_prefix), ("test", "eng")): bitext}, 128)

    lang_codes = {
        ("test", lang_prefix): lang_code,
        ("test", "eng"): "eng_Latn"
    }

    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = load_tokenizer(base_model)
    tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes, max_length=128)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model) 

    x, y, _, _ = tmob.next_batch()

    x = x.to(model.device)
    y = y.to(model.device)
    loss = model(**x, labels=y.input_ids).loss
    with open("seedperp.txt", "a") as file:
        file.write(lang_code + " -> " + "eng_Latn perplexity score: " + str(loss.item()) + "\n")

def from_eng(lang_code):
    tgt_path = "/mnt/storage/hopkins/data/nllb/seed/seed/" + lang_code
    lang_prefix = lang_code[:3]
    bitext = Bitext("/mnt/storage/hopkins/data/nllb/seed/seed/eng_Latn", tgt_path)
    mix = MixtureOfBitexts({(("test", "eng"), ("test", lang_prefix)): bitext}, 128)

    lang_codes = {
        ("test", "eng"): "eng_Latn",
        ("test", lang_prefix): lang_code
    }

    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = load_tokenizer(base_model)
    tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes, max_length=128)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model) 

    x, y, _, _ = tmob.next_batch()

    x = x.to(model.device)
    y = y.to(model.device)
    loss = model(**x, labels=y.input_ids).loss
    with open("seedperp.txt", "a") as file:
        file.write("eng_Latn -> " + lang_code + " perplexity score: " + str(loss.item()) + "\n")

def from_blank(lang_code):
    tgt_path = "/mnt/storage/hopkins/data/nllb/seed/seed/" + lang_code
    lang_prefix = lang_code[:3]
    bitext = Bitext("test_files/blank.txt", tgt_path)
    mix = MixtureOfBitexts({(("test", "eng"), ("test", lang_prefix)): bitext}, 128)

    lang_codes = {
        ("test", "eng"): "eng_Latn",
        ("test", lang_prefix): lang_code
    }

    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = load_tokenizer(base_model)
    tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes, max_length=128)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model) 

    x, y, _, _ = tmob.next_batch()

    x = x.to(model.device)
    y = y.to(model.device)
    loss = model(**x, labels=y.input_ids).loss
    with open("seedperp.txt", "a") as file:
        file.write("language model -> " + lang_code + " perplexity score: " + str(loss.item()) + "\n\n")

if __name__ == "__main__":
    langs = ['ace_Arab', 'bjn_Arab', 'fur_Latn', 'knc_Latn', 'mni_Beng', 'scn_Latn', 'zgh_Tfng', 'ace_Latn', 'bjn_Latn', 'fuv_Latn', 'lij_Latn', 'mri_Latn', 'shn_Mymr', 'ary_Arab', 'bug_Latn', 'gug_Latn', 'lim_Latn', 'nqo_Nkoo', 'srd_Latn', 'arz_Arab', 'crh_Latn', 'hne_Deva', 'lmo_Latn', 'nus_Latn', 'szl_Latn', 'bam_Latn', 'dik_Latn', 'kas_Arab', 'ltg_Latn', 'pbt_Arab', 'taq_Latn', 'ban_Latn', 'dzo_Tibt', 'kas_Deva', 'mag_Deva', 'prs_Arab', 'taq_Tfng', 'bho_Deva', 'eng_Latn', 'knc_Arab', 'vec_Latn']

    for lang in langs:
        to_eng(lang)
        from_eng(lang)
        from_blank(lang) 

    print("done writing data idiot")
