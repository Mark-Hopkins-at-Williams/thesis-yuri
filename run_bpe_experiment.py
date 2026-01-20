import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sentencepiece as spm
from unigram import compute_unigram_entropy
from unigram import train_unigram_distribution
from unigram import train_byte_tokenizer_unigram_lm
from tokenization import PretokenizedBPETokenizer, ByteTokenizer


TRAIN_FILE = "mycorpus.txt"
DEV_FILE = "mydev.txt"

EXPERIMENT_DIR = "experiments/bpe"
VOCAB_SIZES = [800, 1600, 3200, 6400, 12500, 25000, 50000, 100000, 200000]
CHARACTER_COVERAGE = 1.0

# create experiment directory
experiment_version = 0
while os.path.exists(f"{EXPERIMENT_DIR}-v{experiment_version}"):
    experiment_version += 1
experiment_dir = f"{EXPERIMENT_DIR}-v{experiment_version}"
os.makedirs(experiment_dir)

# train byte unigram model and compute dev entropy
dist = train_byte_tokenizer_unigram_lm(
    TRAIN_FILE,
    json_file=Path(experiment_dir) / f"unigram_lm.byte.txt",
)
byte_entropy = 0.0
with open(DEV_FILE) as reader:
    for line in reader:
        byte_entropy += compute_unigram_entropy(line, ByteTokenizer(), dist)

# train BPE models and tokenize corpus with each
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

# train unigram language models and compute entropy
entropies = dict()
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
    tokenized_dev = Path(experiment_dir) / f"tokenized.bpe.{vocab_size}.dev.txt"
    entropy = 0.0
    with open(tokenized_dev) as reader:
        for line in reader:
            entropy += compute_unigram_entropy(line, tokenizer, dist)
    entropies[vocab_size] = entropy

compression_ratios = {k: byte_entropy / entropies[k] for k in entropies}

with open(Path(experiment_dir) / f"results.json", "w") as writer:
    json.dump(compression_ratios, writer, indent=4)

data = {int(k): compression_ratios[k] for k in compression_ratios}
xs = sorted(data.keys())
ys = [data[x] for x in xs]
plt.plot(xs, ys)
plt.savefig(Path(experiment_dir) / "ratios.png")
