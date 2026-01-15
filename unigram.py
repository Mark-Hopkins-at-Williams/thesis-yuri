from collections import Counter
import json
import math
import subprocess
from tokenization import ByteTokenizer
from tokenization import NllbTokenizer
from tqdm import tqdm
from flores_codes import flores_codes
from download import download
import sys


def collect_unigram_counts(text_file, tokenizer, num_lines=1000000, batch_size=1024, show_progress=True):
    if show_progress:
        result = subprocess.run(["wc", "-l", text_file], capture_output=True, text=True)
        line_count = int(result.stdout.strip().split()[0])
    token_freqs = Counter()
    counter = 0
    with open(text_file, "r") as reader:
        batch = []
        line_stream = tqdm(reader, total=line_count) if show_progress else reader
        for line in line_stream:
            batch.append(line.strip())
            if len(batch) >= batch_size:
                inputs = tokenizer(batch)
                for row in inputs["input_ids"]:
                    token_ids = row.tolist()
                    token_freqs.update(token_ids)
                batch = []
            counter += 1
            if counter > num_lines:
                break

    if len(batch) > 0:
        inputs = tokenizer(batch)
        for row in inputs["input_ids"]:
            token_ids = row.tolist()
            token_freqs.update(token_ids)
    return {token: token_freqs[token] for token in token_freqs}


def create_unigram_distribution_from_counts(
    token_counts, vocab_size, tokens_to_ignore, k_smoother=1
):
    def distribution(token):
        if token in tokens_to_ignore:
            raise KeyError(f"Unsupported token: {token}")
        else:
            return (token_counts.get(token, 0) + k_smoother) / total_token_count

    total_token_count = 0
    for token in set(range(vocab_size)) - tokens_to_ignore:
        if token not in tokens_to_ignore:
            total_token_count += (
                token_counts.get(token, 0) + k_smoother
            )  # add-k smoothing
    return distribution


def compute_unigram_entropy(lines, tokenizer, unigram_distribution):
    entropy = 0.0
    for line in lines:
        inputs = tokenizer(line.strip())
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


def train_unigram_distribution(
    text_file, tokenizer, num_lines, tokens_to_ignore, k_smoother, unk_token=None, json_file=None
):
    token_counts = collect_unigram_counts(text_file, tokenizer, num_lines=num_lines)
    if unk_token is not None:
        token_counts[unk_token] = 1
    data = {
        "counts": token_counts,
        "vocab_size": len(tokenizer),
        "tokens_to_ignore": sorted(tokens_to_ignore),
        "k_smoother": k_smoother,
    }
    if json_file is not None:
        with open(json_file, "w+") as writer:
            json.dump(data, writer)
    return create_unigram_distribution_from_counts(
        data["counts"],
        data["vocab_size"],
        set(data["tokens_to_ignore"]),
        data["k_smoother"],
    )


def train_nllb_tokenizer_unigram_lm(text_file, json_file):
    tokenizer = NllbTokenizer("600M")
    return train_unigram_distribution(
        text_file,
        tokenizer,
        1000000,
        {0, 1} | set(range(256001, len(tokenizer))),
        k_smoother=1,
        unk_token=3,
        json_file=json_file,
    )


def train_byte_tokenizer_unigram_lm(text_file, json_file):
    return train_unigram_distribution(
        text_file,
        ByteTokenizer(),
        100000,
        set(range(256, 258)),
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
    # years = ["07"]  # , "08", "09", "10", "11"

    (train_fn, tokenizer) = {
        "byte": (train_byte_tokenizer_unigram_lm, ByteTokenizer()),
        "nllb": (train_nllb_tokenizer_unigram_lm, NllbTokenizer("600M")),
    }.get(mode)

    try: 
        dist = load_unigram_distribution(f"unigram_lms/{lang_code}.{mode}.unigram_lm.json")
    except FileNotFoundError:
        # make unigram lm if not alr existing
        unigram_distribution = train_fn(
        src_path, f"unigram_lms/{lang_code}.{mode}.unigram_lm.json"
        )
        dist = load_unigram_distribution(f"unigram_lms/{lang_code}.{mode}.unigram_lm.json")
    
    test_data = []
    if lang_code in codes:
        try: 
            with open(f"/mnt/storage/hopkins/data/flores/dev.{lang_code}", "r") as reader:
                for line in reader:
                    test_data.append(line)
        except FileNotFoundError: 
            try: 
                with open(f"flores/dev.{lang_code}", "r") as reader:
                    for line in reader:
                        test_data.append(line)
            except: 
                download([lang_code])
                with open(f"flores/dev.{lang_code}", "r") as reader:
                    for line in reader:
                        test_data.append(line)
        except: 
            print("you fucked up somehow")
    else:
        print("AAAAAAAAAAAAAAAAAAAA")
        quit()

    entropy = compute_unigram_entropy(test_data, tokenizer, dist)
    print(f"{mode} unigram entropy for {lang_code}: {entropy}")

    # if mode == "byte":
    #     for year in years:
    #         src_path = f"/mnt/storage/yuri/thesis-yuri/corpus/training-monolingual/news.20{year}.en.shuffled"
    #         unigram_distribution = train_byte_tokenizer_unigram_lm(
    #             src_path, "unigram_lms/news.byte.unigram_lm.json"
    #         )
    #     dist = load_unigram_distribution("unigram_lms/news.byte.unigram_lm.json")
    #     test_data = []
    #     with open("/mnt/storage/hopkins/data/flores/dev.eng_Latn") as reader:
    #         for line in reader:
    #             test_data.append(line)
    #     tokenizer = ByteTokenizer()
    #     entropy = compute_unigram_entropy(test_data, tokenizer, dist)
    #     print(f"entropy: {entropy}")
    # elif mode == "nllb":
    #     for year in years:
    #         src_path = f"/mnt/storage/yuri/thesis-yuri/corpus/training-monolingual/news.20{year}.en.shuffled"
    #         unigram_distribution = train_nllb_tokenizer_unigram_lm(
    #             src_path, "unigram_lms/news.nllb.unigram_lm.json"
    #         )
    #     dist = load_unigram_distribution("unigram_lms/news.nllb.unigram_lm.json")
    #     test_data = []
    #     with open("/mnt/storage/hopkins/data/flores/dev.eng_Latn") as reader:
    #         for line in reader:
    #             test_data.append(line)
    #     tokenizer = NllbTokenizer("600M")
    #     entropy = compute_unigram_entropy(test_data, tokenizer, dist)
    #     print(f"entropy: {entropy}")
