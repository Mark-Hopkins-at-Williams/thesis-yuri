from corpora import load_tokenizer, Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
import torch
from transformers import AutoModelForSeq2SeqLM


def compute_conditional_perplexity(src_lang, tgt_lang, base_model):
    if src_lang is None:
        src_path = "test_files/blank.txt"
        src_lang = "eng_Latn"
    else:
        src_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{src_lang}"
    
    tgt_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{tgt_lang}"
    bitext = Bitext(src_path, tgt_path)
    mix = MixtureOfBitexts(
        {(("test", src_lang), ("test", tgt_lang)): bitext}, batch_size=32, only_once_thru=True
    )
    lang_codes = {("test", src_lang): src_lang, ("test", tgt_lang): tgt_lang}
    tokenizer = load_tokenizer(base_model)
    tmob = TokenizedMixtureOfBitexts(
        mix, tokenizer, lang_codes=lang_codes, max_length=128
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model.to(device)
    next_batch = tmob.next_batch()
    total_loss = 0.0
    while next_batch is not None:
        x, y, _, _ = next_batch        
        x = x.to(model.device)
        y = y.to(model.device)
        loss = model(**x, labels=y.input_ids).loss
        total_loss += loss.item() * y.attention_mask.sum()
        next_batch = tmob.next_batch()
    return total_loss

def compute_lm_perplexity(tgt_lang, base_model):
    compute_conditional_perplexity(None, tgt_lang, base_model)
    





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
    
    for lang in langs:
        print(f'eng_Latn -> {lang}: {compute_conditional_perplexity(lang, "eng_Latn", base_model)}')
        print(f'{lang} -> eng_Latn: {compute_conditional_perplexity("eng_Latn", lang, base_model)}')
        print(f'{lang}: {compute_lm_perplexity(lang,  base_model)}')
        
        

    print("done writing data idiot")
    # with open("seedperp.txt", "a") as file:
    #     file.write(
    #         f"{src_lang} -> {tgt_lang} perplexity score: {loss.item()}\n"
    #     )