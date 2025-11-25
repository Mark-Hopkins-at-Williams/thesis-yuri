from corpora import load_tokenizer, Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
import torch
from transformers import AutoModelForSeq2SeqLM
from collections import Counter
import math
from bytetokenizer import ByteTokenizer, Tokens


def compute_unigram_distribution(src_path, tokenizer, model_used=False):
    token_freqs = Counter()
    total_tokens = 0 
    with open(src_path, 'r') as file:
        for line in file:
            inputs = tokenizer(line.strip()) # returns python lists if using model
            token_vals = inputs["input_ids"]       
            if model_used:
                token_freqs.update(token_vals[1:])   
                total_tokens += len(token_vals) - 1 # subtract 1 bc language token included? 
            else:
                token_freqs.update(token_vals) 
                total_tokens += len(token_vals)
    return {token: token_freqs[token] / total_tokens for token in token_freqs}
    

def compute_unigram_perplexity(src_path, tokenizer, model_used=False):
    unigram_distribution = compute_unigram_distribution(src_path, tokenizer, model_used=model_used)
    perp = 0.0
    with open(src_path, 'r') as file:
        for line in file:
            inputs = tokenizer(line.strip())#, return_tensors="pt")
            token_vals = inputs["input_ids"]
            for token in token_vals[1:len(token_vals)]:
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
    hf_tokenizer = load_tokenizer(base_model)
    def tokenizer(s):
        return hf_tokenizer(s, return_tensor="pt")
    
    byte_tokenizer = ByteTokenizer("utf-8")

    # dist = compute_unigram_distribution(test_path, byte_tokenizer)
    # print(compute_unigram_perplexity(test_path, byte_tokenizer))

    # with open("unigram_loss.txt", "a") as file:
    #     for lang in langs:
    #         src_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{lang}"
    #         perplexity = compute_unigram_perplexity(src_path, hf_tokenizer, model_used=True)
    #         print(f'{lang} unigram perplexity: {perplexity}\n')
    #         file.write(f'{lang} unigram perplexity: {perplexity}\n')

    with open("unigram_loss.txt", "a") as file: 
        file.write("unigram perplexity for english using a gigantic corpus\n\n")
        years = ["07", "08", "09", "10", "11"]
        for year in years: 
            src_path = f"/corpus/training-monolingual/news.20{year}.en.shuffled"
            # file not found error idfk why
            perplexity = compute_unigram_perplexity(src_path, hf_tokenizer, model_used=True)
            file.write(f'20{year} english unigram perplexity: {perplexity}\n')

    print("done writing data idiot")
