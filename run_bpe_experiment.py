from corpora import MixtureOfBitexts
import json
import sentencepiece as spm
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize data using BPE")
    parser.add_argument(
        "config_file", type=str
    )
    parser.add_argument(
        "split", type=str
    )
    parser.add_argument(
        "v_size", type=int
    )
    parser.add_argument(
        "--char_coverage", type=float, default=1.0
    )
    args = parser.parse_args()
    config_file = args.config_file
    dsplit = args.split
    v_size = args.v_size 
    char_coverage = args.char_coverage

    with open(config_file) as reader:
        config = json.load(reader)
    bitexts = {
        split: MixtureOfBitexts.create_from_config(config, split, only_once_thru=True)
        for split in ["train", "dev", "test"]
    }

    all_corpora = dict()
    for corpus in config["corpora"]:
        for key in config["corpora"][corpus]:
            for split in config["corpora"][corpus][key]:
                all_corpora[(corpus, key, split)] = config["corpora"][corpus][key][split]
    bitexts = dict()
    for bitext in config["bitexts"]:
        for split in ["train", "dev", "test"]:
            src = (bitext["corpus"], bitext["src"], split)
            tgt = (bitext["corpus"], bitext["tgt"], split)
            bitexts[(src, tgt)] = (all_corpora[src], all_corpora[tgt])

    # train a BPE model
    sp_models = dict()
    for src_key, tgt_key in bitexts:
        if src_key[2] == "train":
            src_file, tgt_file = bitexts[(src_key, tgt_key)]
            prefix = f"{'_'.join(src_key)}-{'_'.join(tgt_key)}"
            sp_models[(src_key, tgt_key)] = f"{prefix}.model"
            spm.SentencePieceTrainer.train(
                input=tgt_file,
                model_prefix=prefix,
                vocab_size=v_size, 
                user_defined_symbols=[],
                model_type="bpe",
                character_coverage=char_coverage, 
            )

    for src_key, tgt_key in sp_models:
        size = str(v_size//1000)
        str_size = f"{size}k"
        # load bpe model 
        sp = spm.SentencePieceProcessor(model_file=sp_models[(src_key, tgt_key)])
        (src_corpus, src_lang, _) = src_key
        (tgt_corpus, tgt_lang, _) = tgt_key

        if dsplit == "train":
            tokenized_target_path = f"bpe_unigram_training/{str_size}.{src_lang}-to-{tgt_lang}.tokens"
        elif dsplit == "dev": 
            tokenized_target_path = f"flores/bpe_tokenized/{str_size}.dev.{src_lang}"
        bitext_src, bitext_tgt = bitexts[
            ((src_corpus, src_lang, "dev"), (tgt_corpus, tgt_lang, "dev"))
        ]
        with open(bitext_tgt) as reader, open(tokenized_target_path, 'w') as writer:
            for line in reader:
                tokenized = sp.encode(line, out_type=str)
                processed = " ".join(tokenized)
                writer.write(f"{processed}\n")
