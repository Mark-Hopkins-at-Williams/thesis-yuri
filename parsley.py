from datasets import load_dataset
from tqdm import tqdm
import sys


def parsley(ds, text_var, lang_code, num_lines):
    ds_iter = iter(ds)
    with open(f"{lang_code}.corpus.txt", "w") as writer:
        for _ in tqdm(range(num_lines)):
            line = next(ds_iter)
            writer.write(line[text_var] + "\n")


if __name__ == "__main__":
    train_ds = load_dataset("RichNachos/georgian-corpus", split="train", streaming=True)
    parsley(train_ds, "doc_content", "kat_Geor", 1000000)
