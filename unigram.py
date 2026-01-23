from collections import Counter
import json
import math
import subprocess
from tokenization import ByteTokenizer, NllbTokenizer, WhiteSpaceTokenizer
from tqdm import tqdm
from flores_codes import flores_codes
from download import download
import argparse
import torch 

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
                inputs = tokenizer(batch) # why are we using tokenizer on a LIST here ????????? 
                for row in inputs["input_ids"]: # now we assume inputs["input_ids"] is a 2D list????? "row" should be a list??????????? 
                    if isinstance(row, list):
                        token_ids = row 
                        token_freqs.update(token_ids)
                    elif isinstance(row, torch.Tensor): # hope this is a torch tensor D:
                        token_ids = row.tolist()
                        token_freqs.update(token_ids)
                    else: 
                        print("WEEWOOOWEEWOOWEEEWOOO. expected list or tensor, got smth else")
                        exit()
                batch = []
            counter += 1
            if counter > num_lines:
                break
    if len(batch) > 0:
        inputs = tokenizer(batch)
        for row in inputs["input_ids"]:
            if isinstance(row, list):
                token_ids = row 
            else: # hope this is a torch tensor D:
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

    total_token_count = sum(k for k in token_counts.values() if k not in tokens_to_ignore)
    # for token in set(range(vocab_size)) - tokens_to_ignore:
    #     if token not in tokens_to_ignore:
    #         total_token_count += (
    #             token_counts.get(token, 0) + k_smoother
    #         )  # add-k smoothing 
    return distribution


def compute_unigram_entropy(lines, tokenizer, unigram_distribution): # lines should be a list of str?? 
    entropy = 0.0
    for line in lines:
        inputs = tokenizer(line.strip()) # why are we using tokenizer on a string here ????????? 
        # print(inputs)
        # print(f"type: {type(inputs)}")
        if isinstance(inputs["input_ids"], list):
            token_vals = inputs["input_ids"] # THIS SHOULD BE A 1D LIST???
        else: # pray that it's a tensor 
            # token_vals = inputs["input_ids"].squeeze().tolist() # why was this squeeze
            token_vals = inputs["input_ids"].flatten().tolist()
        # print(token_vals)
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

def compute_bpe_entropy(lines, tokenizer, unigram_distribution): # lines should be a list of str?? 
    entropy = 0.0
    inputs = tokenizer(lines) 
    token_vals = inputs["input_ids"] # should just be a list of all tokens 
    for token in token_vals:
        # print(f"actual token: {token}")
        token_prob = unigram_distribution(token)
        # print(f"token prob: {token_prob}")
        entropy += -math.log2(token_prob)
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
        with open(json_file, "w") as writer:
            json.dump(data, writer)
    return create_unigram_distribution_from_counts(
        data["counts"],
        data["vocab_size"],
        set(data["tokens_to_ignore"]),
        data["k_smoother"],
    )

def train_bpe_tokenizer_unigram_lm(text_file, json_file):
    return train_unigram_distribution(
        text_file,
        WhiteSpaceTokenizer(),
        1000000000, # kinda scuffed 
        {}, # no tokens to ignore?? 
        k_smoother=1,
        json_file=json_file,
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
    counts = data["counts"] # dict of each unique token -> frequency 
    return create_unigram_distribution_from_counts(
        {k: counts[k] for k in counts}, # used to cast k as an int, is that ever needed? stopped casting to int bc bpe tokens are strings, but can configure later 
        data["vocab_size"],
        set(data["tokens_to_ignore"]),
        data["k_smoother"],
    )


if __name__ == "__main__":
    codes = flores_codes().keys()
    parser = argparse.ArgumentParser(description="Calculate unigram entropy!")
    parser.add_argument(
        "mode", type=str, help="byte, bpe, or nllb unigram"
    )
    parser.add_argument(
        "--path", type=str
    )
    parser.add_argument(
        "lang_code", type=str
    )
    parser.add_argument(
        "-b", "--bpe_vocab_size", type=str, default="16k"
    )
    args = parser.parse_args()
    mode = args.mode 
    src_path = args.path 
    lang_code = args.lang_code
    bpe_vocab_size = args.bpe_vocab_size

    if mode == "bpe":
        unigram_path = f"unigram_lms/{lang_code}.{bpe_vocab_size}-{mode}.unigram_lm.json"
        bpe_dev_path = f"flores/bpe_tokenized/{bpe_vocab_size}.dev.{lang_code}"
    else: 
        unigram_path = f"unigram_lms/{lang_code}.{mode}.unigram_lm.json"

    (train_fn, tokenizer) = {
        "byte": (train_byte_tokenizer_unigram_lm, ByteTokenizer()),
        "nllb": (train_nllb_tokenizer_unigram_lm, NllbTokenizer("600M")),
        "bpe": (train_bpe_tokenizer_unigram_lm, WhiteSpaceTokenizer())
    }.get(mode)

    try: 
        dist = load_unigram_distribution(unigram_path)
    except FileNotFoundError:
        # make unigram lm if not alr existing
        unigram_distribution = train_fn(
        src_path, unigram_path
        ) 
        dist = load_unigram_distribution(unigram_path)
    
    test_data = []
    if lang_code in codes:
        if mode == "bpe":
            with open(bpe_dev_path, "r") as reader:
                for line in reader:
                    test_data.append(line)
        else:
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

    # test_data is a list of str always ?? 
    if mode == "bpe": 
        entropy = compute_bpe_entropy(test_data, tokenizer, dist)
        print(f"{bpe_vocab_size} {mode} unigram entropy for {lang_code}: {entropy}")
    else:
        entropy = compute_unigram_entropy(test_data, tokenizer, dist)
        print(f"{mode} unigram entropy for {lang_code}: {entropy}")