import evaluate
from pathlib import Path
import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts
from permutations import load_permutation_map


def translate(
    src_tokenized,
    tokenizer,
    model,
    tgt_lang,
    permutation,
    a=32,
    b=3,
    num_beams=4,
    **kwargs
):
    model.eval()  # turn off training mode
    result = model.generate(
        **src_tokenized.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * src_tokenized.input_ids.shape[1]),
        num_beams=num_beams,
        **kwargs
    )
    result = result.to('cpu')
    result.apply_(permutation.get_inverse())
    return tokenizer.batch_decode(result, skip_special_tokens=True)


def translate_tokenized_mixture_of_bitexts(mix, model, tokenizer, pmap):         
    if USE_CUDA:
        model.cuda()
    batch = mix.next_batch()  
    translations = dict()
    while batch is not None:
        src, _, src_code, tgt_code = batch
        permutation = pmap[tgt_code]
        key = '->'.join([src_code, tgt_code])
        if key not in translations:
            translations[key] = []
        translated = translate(src, tokenizer, model, tgt_code, permutation)
        translations[key].extend(translated)
        batch = mix.next_batch() 
    return translations


def evaluate_translations(candidate_translations, reference_translations):
    bleu_calc = evaluate.load("sacrebleu")
    chrf_calc = evaluate.load("chrf")
    reference_translations = [[ref] for ref in reference_translations]
    bleu_result = bleu_calc.compute(predictions = candidate_translations, references = reference_translations)
    chrf_result = chrf_calc.compute(predictions = candidate_translations, references = reference_translations)
    return {'bleu': round(bleu_result["score"], 3), 'chrf': round(chrf_result["score"], 3)}


def main():
    test_mix = MixtureOfBitexts.create_from_files(
        {
            "fra_Latn": "data/test.fr",
            "eng_Latn": "data/test.en",
        },
        [("eng_Latn", "fra_Latn")],
        batch_size=32,
        only_once_thru=True
    )           
    references = dict()
    batch = test_mix.next_batch()
    while batch is not None:
        _, tgt, src_code, tgt_code = batch        
        if (src_code, tgt_code) not in references:
            references[(src_code, tgt_code)] = []        
        references[(src_code, tgt_code)].extend(tgt)
        batch = test_mix.next_batch()  
    print(references)
    
if __name__ == "__main__":
    main()