import math
import random
from simple_unigram import train_unigram_distribution


def generate_token():
    random_draw = random.random()
    if random_draw < 0.5:
        return "ab"
    elif random_draw < 0.75:
        return "ca"
    else:
        return "a\n"


def generate_string(num_tokens):
    tokens = [generate_token() for i in range(num_tokens)]
    return "".join(tokens)


def char_tokenizer(s):
    return list(s)


def real_tokenizer(s):
    toks = [s[i] + s[i + 1] for i in range(0, len(s) - 1, 2)]
    if len(s) % 2 == 1:
        toks.append(s[-1])
    return toks


def bad_tokenizer(s):
    toks = [s[i] + s[i + 1] + s[i + 2] for i in range(0, len(s) - 2, 3)]
    if len(s) % 3 > 0:
        toks.append("".join(s[-(len(s) % 3) :]))
    return toks


def space_tokenizer(s):
    return s.split()


def get_encoding_length(text_file, tokenizer):
    encoding_length = 0.0
    with open(text_file, "r") as reader:
        for line in reader:
            tokens = tokenizer(line.strip())
            for token in tokens:
                p = dist(token)
                if p > 0:
                    encoding_length += -math.log2(p)
                else:
                    encoding_length += -math.log2(0.0000001)
    return encoding_length


if __name__ == "__main__":
    file_prefix = "ab"
    with open(f"{file_prefix}.train", "w") as writer:
        for sent in generate_string(1000000):
            writer.write(f"{sent}")
    with open(f"{file_prefix}.test", "w") as writer:
        for sent in generate_string(1000000):
            writer.write(f"{sent}")

    tokenizer = char_tokenizer
    dist = train_unigram_distribution(f"{file_prefix}.train", tokenizer)
    encoding_len = get_encoding_length(f"{file_prefix}.test", tokenizer)
    print(f"char tokenizer: {encoding_len}")

    tokenizer = real_tokenizer
    dist = train_unigram_distribution(f"{file_prefix}.train", tokenizer)
    encoding_len = get_encoding_length(f"{file_prefix}.test", tokenizer)
    print(f"real tokenizer: {encoding_len}")

    tokenizer = bad_tokenizer
    dist = train_unigram_distribution(f"{file_prefix}.train", tokenizer)
    encoding_len = get_encoding_length(f"{file_prefix}.test", tokenizer)
    print(f"bad tokenizer: {encoding_len}")
