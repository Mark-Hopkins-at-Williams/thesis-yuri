import argparse
from pathlib import Path
import sentencepiece as spm
import os


def train_sentencepiece_tokenizer(
    train_file, vocab_size, model_dir, character_coverage=1.0
):
    model_prefix = Path(model_dir) / f"bpe.{vocab_size}"
    spm.SentencePieceTrainer.train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=0,
        unk_id=1,
        eos_id=2,
        bos_id=-1,
        pad_piece="<pad>",
        unk_piece="<unk>",
        eos_piece="</s>",
        control_symbols=["spa_Latn", "quy_Latn"],
        model_type="bpe",
        character_coverage=character_coverage,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentencepiece tokenizer.")
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory to save tokenizer files"
    )
    parser.add_argument(
        "--train", type=str, required=True, help="Training text for tokenizer"
    )
    parser.add_argument(
        "--vocab_size", type=int, required=True, help="Vocab size for tokenizer"
    )
    args = parser.parse_args()

    # make directory for tokenizer files
    base_dir = args.dir
    model_version = 0
    while os.path.exists(f"{base_dir}-{args.vocab_size}-v{model_version}"):
        model_version += 1
    model_dir = f"{base_dir}-{args.vocab_size}-v{model_version}"
    os.makedirs(model_dir)

    train_sentencepiece_tokenizer(args.train, args.vocab_size, model_dir)
