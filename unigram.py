from corpora import load_tokenizer, Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
import torch
from transformers import AutoModelForSeq2SeqLM
from collections import Counter
import math


def unigram_distribution(src_lang, base_model):
    src_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{src_lang}"
    tokenizer = load_tokenizer(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_freqs = Counter()
    total_tokens = 0 
    with open(src_path, 'r') as file:
        for line in file:
            # i actually don't know what .to(model.device) does lol ...
            inputs = tokenizer(line.strip(), return_tensors="pt").to(model.device)
            #print(inputs.token_to_chars(5))
            token_vals = inputs.input_ids.flatten().tolist()
            token_freqs.update(token_vals[1:len(token_vals)-1])
            total_tokens += inputs.attention_mask.sum().item() - 2
    # token_probs = Counter()
    for token in token_freqs:
        prob = token_freqs[token] / total_tokens
        token_freqs[token] = prob

    return token_freqs


def calculate_unigram_perplexity(src_lang, unigram_distribution, base_model):
    src_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{src_lang}"
    tokenizer = load_tokenizer(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perp = 0.0

    with open(src_path, 'r') as file:
        for line in file:
            # i actually don't know what .to(model.device) does lol ...
            inputs = tokenizer(line.strip(), return_tensors="pt").to(model.device)
            token_vals = inputs.input_ids.flatten().tolist()
            for token in token_vals[1:len(token_vals)-1]:
                token_prob = unigram_distribution[token]
                perp += -math.log(token_prob)
    return perp


if __name__ == "__main__":
    langs = [
        "ace_Arab",
        "bjn_Arab",
        "fur_Latn",
        "knc_Latn",
        "mni_Beng",
        "scn_Latn",
        "zgh_Tfng",
        "ace_Latn",
        "bjn_Latn",
        "fuv_Latn",
        "lij_Latn",
        "mri_Latn",
        "shn_Mymr",
        "ary_Arab",
        "bug_Latn",
        "gug_Latn",
        "lim_Latn",
        "nqo_Nkoo",
        "srd_Latn",
        "arz_Arab",
        "crh_Latn",
        "hne_Deva",
        "lmo_Latn",
        "nus_Latn",
        "szl_Latn",
        "bam_Latn",
        "dik_Latn",
        "kas_Arab",
        "ltg_Latn",
        "pbt_Arab",
        "taq_Latn",
        "ban_Latn",
        "dzo_Tibt",
        "kas_Deva",
        "mag_Deva",
        "prs_Arab",
        "taq_Tfng",
        "bho_Deva",
        "eng_Latn",
        "knc_Arab",
        "vec_Latn",
    ]

    base_model = "facebook/nllb-200-distilled-600M"

    with open("unigram_loss.txt", "a") as file:
        for lang in langs:
            unigram_probs = unigram_distribution(lang, base_model)
            perplexity = calculate_unigram_perplexity(lang, unigram_probs, base_model)
            file.write(f'{lang} unigram perplexity: {perplexity}\n')
            
    print("done writing data idiot")

    # print("done writing data idiot")
    # with open("seedperp.txt", "a") as file:
    #     file.write(
    #         f"{src_lang} -> {tgt_lang} perplexity score: {loss.item()}\n"
    #     )