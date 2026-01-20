import json
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path
import sentencepiece as spm
from unigram import train_unigram_distribution
from tokenization import PretokenizedBPETokenizer

TRAIN_FILE = "data/europarl.en-lt.lt"
DEV_FILE = "data/wmt19.en-lt.600k.lt"
EXPERIMENT_DIR = "experiments/mdl-europarl-lt"
VOCAB_SIZES = [800, 1600, 3200, 6400, 12800, 25000]
CHARACTER_COVERAGE = 1.0
NUM_CODEPOINTS = 65536
CODEPOINT_ENCODING_LENGTH = math.ceil(math.log2(NUM_CODEPOINTS))

# create experiment directory
experiment_version = 0
while os.path.exists(f"{EXPERIMENT_DIR}-v{experiment_version}"):
    experiment_version += 1
experiment_dir = f"{EXPERIMENT_DIR}-v{experiment_version}"
os.makedirs(experiment_dir)

# train BPE models and tokenize corpus with each
vocab_token_maps = dict()
for vocab_size in VOCAB_SIZES:
    model_prefix = Path(experiment_dir) / f"bpe.{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=TRAIN_FILE,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        user_defined_symbols=[],
        model_type="bpe",
        character_coverage=CHARACTER_COVERAGE,
    )
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    vocab_token_maps[vocab_size] = {
        i: sp.id_to_piece(i) for i in range(sp.get_piece_size())
    }
    with open(TRAIN_FILE) as reader:
        with open(
            Path(experiment_dir) / f"tokenized.bpe.{vocab_size}.train.txt", "w"
        ) as writer:
            for line in reader:
                tokenized = " ".join([str(x) for x in sp.encode(line)])
                writer.write(tokenized + "\n")
    with open(DEV_FILE) as reader:
        with open(
            Path(experiment_dir) / f"tokenized.bpe.{vocab_size}.dev.txt", "w"
        ) as writer:
            for line in reader:
                tokenized = " ".join([str(x) for x in sp.encode(line)])
                writer.write(tokenized + "\n")


# train unigram language models
unigram_lms = dict()
for vocab_size in VOCAB_SIZES:
    tokenizer = PretokenizedBPETokenizer(vocab_size)
    tokenized_train = Path(experiment_dir) / f"tokenized.bpe.{vocab_size}.train.txt"
    dist = train_unigram_distribution(
        tokenized_train,
        tokenizer,
        [tokenizer.get_special_tokens()["<pad>"]],
        k_smoother=1,
        json_file=Path(experiment_dir) / f"unigram_lm.bpe.{vocab_size}.txt",
    )
    unigram_lms[vocab_size] = dist

# encode codebooks
codebook_encoding_lengths = dict()
for vocab_size in VOCAB_SIZES:
    total_encoding_length = 2 * vocab_size + 2
    token_map = vocab_token_maps[vocab_size]
    unigram_lm = unigram_lms[vocab_size]
    for token_id in token_map:
        token = token_map[token_id]
        encoded_token_length = 2 * math.ceil(math.log2(len(token))) + 2
        encoded_token_content = len(token) * CODEPOINT_ENCODING_LENGTH
        token_prob = unigram_lm(token_id)
        codeword_length = math.ceil(-math.log2(token_prob))
        encoded_codeword_length = 2 * math.ceil(math.log2(codeword_length)) + 2
        encoded_codeword_content = codeword_length
        total_encoding_length += (
            encoded_token_length
            + encoded_token_content
            + encoded_codeword_length
            + encoded_codeword_content
        )
    codebook_encoding_lengths[vocab_size] = total_encoding_length

# encode corpus (TODO: look into how special tokens are treated)
corpus_encoding_lengths = dict()
overall_encoding_lengths = dict()
for vocab_size in VOCAB_SIZES:
    tokenizer = PretokenizedBPETokenizer(vocab_size)
    tokenized_train = Path(experiment_dir) / f"tokenized.bpe.{vocab_size}.dev.txt"
    unigram_lm = unigram_lms[vocab_size]
    total_encoding_length = 0.0
    with open(tokenized_train) as reader:
        line_encoding_length = 0
        for line in reader:
            line = line.strip()
            if len(line) > 0:
                inputs = tokenizer([line])
                token_vals = inputs["input_ids"].squeeze().tolist()
                for token in token_vals:
                    token_prob = unigram_lm(token)
                    line_encoding_length += math.ceil(-math.log2(token_prob))
        total_encoding_length += line_encoding_length
    corpus_encoding_lengths[vocab_size] = total_encoding_length
    overall_encoding_lengths[vocab_size] = (
        codebook_encoding_lengths[vocab_size] + total_encoding_length
    )

for k in corpus_encoding_lengths:
    print(f"{k}:")
    print(f"codebook encoding: {codebook_encoding_lengths[k]}")
    print(f"corpus encoding:   {corpus_encoding_lengths[k]}")
    print(
        f"overall encoding:  {codebook_encoding_lengths[k] + corpus_encoding_lengths[k]}"
    )


with open(Path(experiment_dir) / f"results.json", "w") as writer:
    json.dump(overall_encoding_lengths, writer, indent=4)

data = {int(k): overall_encoding_lengths[k] for k in overall_encoding_lengths}
xs = sorted(data.keys())
ys = [data[x] for x in xs]
plt.plot(xs, ys)
plt.savefig(Path(experiment_dir) / "results.png")
