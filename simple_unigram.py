from collections import Counter
import json
import math
import subprocess
from tokenization import ByteTokenizer
from tokenization import NllbTokenizer
from tqdm import tqdm
from flores_codes import flores_codes
import sys


def collect_unigram_counts(text_file, tokenizer, batch_size=1024, show_progress=True):
    if show_progress:
        result = subprocess.run(["wc", "-l", text_file], capture_output=True, text=True)
        line_count = int(result.stdout.strip().split()[0])
    token_freqs = dict()
    with open(text_file, "r") as reader:
        line_stream = tqdm(reader, total=line_count) if show_progress else reader
        for line in line_stream:
            tokens = tokenizer(line.strip())
            for token in tokens:
                if token not in token_freqs:
                    token_freqs[token] = 0
                token_freqs[token] += 1
    return token_freqs


def create_unigram_distribution_from_counts(token_counts):

    def distribution(token):
        return token_counts.get(token, 0) / total_token_count

    total_token_count = 0
    for token in token_counts:
        total_token_count += token_counts.get(token, 0)
    return distribution


def create_unigram_dict_from_counts(token_counts):
    total_token_count = 0
    probs = dict()
    for token in token_counts:
        total_token_count += token_counts.get(token, 0)
    for token in token_counts:
        probs[token] = token_counts.get(token, 0) / total_token_count

    return probs


def compute_unigram_entropy(line, tokenizer, unigram_distribution):
    entropy = 0.0
    inputs = tokenizer([line.strip()])
    token_vals = inputs["input_ids"].squeeze().tolist()
    in_prologue = True
    in_epilogue = False
    for token in token_vals:
        try:
            token_prob = unigram_distribution(token)
            in_prologue = False
            if in_epilogue:
                raise Exception(f"Found token in the wrong place: {line}")
            entropy += -math.log2(token_prob)
        except KeyError:
            if in_prologue or in_epilogue:
                pass
            else:
                in_epilogue = True
    return entropy


def train_unigram_distribution(text_file, tokenizer):
    token_counts = collect_unigram_counts(text_file, tokenizer)
    return create_unigram_distribution_from_counts(token_counts)


def train_unigram_dict(text_file, tokenizer):
    token_counts = collect_unigram_counts(text_file, tokenizer)
    return create_unigram_dict_from_counts(token_counts)


def train_nllb_tokenizer_unigram_lm(text_file, json_file):
    tokenizer = NllbTokenizer("600M")
    return train_unigram_distribution(
        text_file,
        tokenizer,
        {-100, 0, 1} | set(range(256001, len(tokenizer))),
        k_smoother=1,
        unk_token=3,
        json_file=json_file,
    )


def train_byte_tokenizer_unigram_lm(text_file, json_file):
    return train_unigram_distribution(
        text_file,
        ByteTokenizer(),
        set(range(257, 258)),
        k_smoother=1,
        json_file=json_file,
    )


def load_unigram_distribution(json_file):
    with open(json_file) as reader:
        data = json.load(reader)
    counts = data["counts"]
    return create_unigram_distribution_from_counts(
        {int(k): counts[k] for k in counts},
        data["vocab_size"],
        set(data["tokens_to_ignore"]),
        data["k_smoother"],
    )


if __name__ == "__main__":
    codes = flores_codes().keys()
    mode = sys.argv[1]
    src_path = sys.argv[2]
    lang_code = sys.argv[3]
    years = ["07"]  # , "08", "09", "10", "11"

    (train_fn, tokenizer) = {
        "byte": (train_byte_tokenizer_unigram_lm, ByteTokenizer()),
        "nllb": (train_nllb_tokenizer_unigram_lm, NllbTokenizer("600M")),
    }.get(mode)

    if mode == "byte":
        for year in years:
            src_path = f"/mnt/storage/yuri/thesis-yuri/corpus/training-monolingual/news.20{year}.en.shuffled"
            unigram_distribution = train_byte_tokenizer_unigram_lm(
                src_path, f"unigram_lms/{lang_code}.byte.unigram_lm.json"
            )
        dist = load_unigram_distribution(
            f"unigram_lms/{lang_code}.byte.unigram_lm.json"
        )
        test_data = []
        with open("/mnt/storage/hopkins/data/flores/dev.eng_Latn") as reader:
            for line in reader:
                test_data.append(line)
        tokenizer = ByteTokenizer()
        entropy = compute_unigram_entropy(test_data, tokenizer, dist)
        print(f"entropy: {entropy}")
    elif mode == "nllb":
        for year in years:
            src_path = f"/mnt/storage/yuri/thesis-yuri/corpus/training-monolingual/news.20{year}.en.shuffled"
            unigram_distribution = train_nllb_tokenizer_unigram_lm(
                src_path, f"unigram_lms/{lang_code}.nllb.unigram_lm.json"
            )
        dist = load_unigram_distribution(
            f"unigram_lms/{lang_code}.nllb.unigram_lm.json"
        )
        test_data = []
        with open("/mnt/storage/hopkins/data/flores/dev.eng_Latn") as reader:
            for line in reader:
                test_data.append(line)
        tokenizer = NllbTokenizer("600M")
        entropy = compute_unigram_entropy(test_data, tokenizer, dist)
        print(f"entropy: {entropy}")
