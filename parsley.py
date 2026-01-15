from datasets import load_dataset
from tqdm import tqdm
import sys

# { "src_lang": "acf", "src_text": "Jou òswè sala, sé gadyenn mouton-an kité mouton yo an savann-an", "tgt_lang": "eng", "tgt_text": "That night the shepherds le ft their sheep in the field" }

def parsley(ds, text_var, lang_code, num_lines):
    # src = ds['translation'][0]['src_lang']
    # tgt = ds['translation'][0]['tgt_lang']
    with open(f"corpus/training-monolingual/{lang_code}.corpus.txt", "w") as writer:
        for i in tqdm(range(num_lines)):
            line = ds[text_var][i]
            writer.write(line + "\n")

if __name__ == "__main__":
    # split = str(sys.argv[1])
    train_ds = load_dataset("RichNachos/georgian-corpus", split="train")
    parsley(train_ds, "doc_content", "kat_Geor", 1000000)
    # train_ds = load_dataset("daqc/wikipedia-txt-spanish", split="train")
    # parsley(train_ds, "text", "spa_Latn", 1000000)