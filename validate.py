from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
    result.apply_(permutation.get_inverse())
    return tokenizer.batch_decode(result, skip_special_tokens=True)


if __name__ == "__main__":
    test_mix = MixtureOfBitexts.create_from_files(
        {
            "fra_Latn": "data/test.fr",
            "eng_Latn": "data/test.en",
        },
        [("eng_Latn", "fra_Latn")],
        batch_size=2,
    )           
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    test_data = TokenizedMixtureOfBitexts(test_mix, tokenizer, max_length=128)
    src, tgt = test_data.next_batch()
    model = AutoModelForSeq2SeqLM.from_pretrained("experiments/exp-v17/") 
    pmap = load_permutation_map(Path("experiments/exp-v17/") / "permutations.json")
    permutation = pmap["fra_Latn"]
    translated = translate(src, tokenizer, model, "fra_Latn", permutation)
    for x in translated:
        print(x)


def batched_translate(texts, batch_size=8, **kwargs):
    idxs, texts2 = zip(*sorted(enumerate(texts), key=lambda p: len(p[1]), reverse=True))
    results = []
    for i in tqdm(range(0, len(texts2), batch_size)):
        results.extend(translate(texts2[i : i + batch_size], **kwargs))
    return [p for _, p in sorted(zip(idxs, results))]
