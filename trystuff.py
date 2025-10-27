import json
import unittest
from corpora import load_tokenizer, Bitext, MixtureOfBitexts, TokenizedMixtureOfBitexts
from torch import tensor
from transformers import AutoModelForSeq2SeqLM

# i assume the first one is src, second is tgt 
bitext = Bitext("test_files/blank.txt", "/mnt/storage/hopkins/data/nllb/seed/seed/fur_Latn")
mix = MixtureOfBitexts({(("test", "eng"), ("test", "fur")): bitext}, 128)
#print(mix.next_batch())trystuff.py


lang_codes = {
    ("test", "eng"): "eng_Latn",
    ("test", "fur"): "fur_Latn"
}
base_model = "facebook/nllb-200-distilled-600M"
tokenizer = load_tokenizer(base_model)
tmob = TokenizedMixtureOfBitexts(mix, tokenizer, lang_codes=lang_codes, max_length=128)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model) 

x, y, _, _ = tmob.next_batch()

x = x.to(model.device)
y = y.to(model.device)
loss = model(**x, labels=y.input_ids).loss
print("loss: " + str(loss.item()))

