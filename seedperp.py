from corpora import load_tokenizer, Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
import torch
from transformers import AutoModelForSeq2SeqLM
from unigram import compute_unigram_perplexity

def compute_conditional_perplexity(src_lang, src_path, tgt_lang, tgt_path, base_model, batch_size=32):
    bitext = Bitext(src_path, tgt_path)
    mix = MixtureOfBitexts(
        {(("test", src_lang), ("test", tgt_lang)): bitext}, batch_size=batch_size, only_once_thru=True
    )
    lang_codes = {("test", src_lang): src_lang, ("test", tgt_lang): tgt_lang}
    tokenizer = load_tokenizer(base_model)
    tmob = TokenizedMixtureOfBitexts(
        mix, tokenizer, lang_codes=lang_codes, max_length=128
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    return total_loss.item()





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
    
    src_path = 'test_files/lang1.txt'  # english
    tgt_path = 'test_files/lang2.txt'  # french
    p_cond = compute_conditional_perplexity('eng_Latn', src_path, "fra_Latn", tgt_path, base_model, batch_size=6)
    p_unigram = compute_unigram_perplexity(tgt_path, base_model)
    
    src_path = "test_files/blank.txt"
    p_lm = compute_conditional_perplexity("eng_Latn", src_path, "fra_Latn", tgt_path, base_model)
        
    
    print(p_cond)
    print(p_unigram)
    print(p_lm)
    exit()
        
    for lang in langs:
        src_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{lang}"
        tgt_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/eng_Latn"
        print(f'eng_Latn -> {lang}: {compute_conditional_perplexity(lang, src_path, "eng_Latn", tgt_path, base_model)}')
        
        src_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/eng_Latn"
        tgt_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{lang}"        
        print(f'{lang} -> eng_Latn: {compute_conditional_perplexity("eng_Latn", src_path, lang, tgt_path, base_model)}')
        
        src_path = "test_files/blank.txt"
        tgt_path = f"/mnt/storage/hopkins/data/nllb/seed/seed/{lang}"                
        print(f'{lang}: {compute_conditional_perplexity("eng_Latn", src_path, lang, tgt_path, base_model)}')
        
        

    print("done writing data idiot")
    # with open("seedperp.txt", "a") as file:
    #     file.write(
    #         f"{src_lang} -> {tgt_lang} perplexity score: {loss.item()}\n"
    #     )