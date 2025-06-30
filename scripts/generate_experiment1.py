import json


def create_bituning_config1(num_train_lines):
    return {
        "model_dir": f"experiments/exp1-bi1-{num_train_lines}",
        "corpora": {
            "eng_Latn": {
                "train": "data/train.en",
                "dev": "data/dev.en",
                "test": "data/test.en",
                "permutation": 0,
            },
            "tsn_Latn": {
                "train": "data/train.de",
                "dev": "data/dev.de",
                "test": "data/test.de",
                "permutation": 1,
            },
        },
        "bitexts": [
            {"src": "eng_Latn", "tgt": "tsn_Latn", "train_lines": [0, num_train_lines]}
        ],
        "finetuning_parameters": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "batch_size": 64,
            "num_steps": 60000,
        },
    }


def create_bituning_config2(num_train_lines):
    return {
        "model_dir": f"experiments/exp1-bi2-{num_train_lines}",
        "corpora": {
            "eng_Latn": {
                "train": "data/train.en",
                "dev": "data/dev.en",
                "test": "data/test.en",
                "permutation": 0,
            },
            "tsn_Latn": {
                "train": "data/train.de",
                "dev": "data/dev.de",
                "test": "data/test.de",
                "permutation": 1,
            },
        },
        "bitexts": [
            {
                "src": "eng_Latn",
                "tgt": "tsn_Latn",
                "train_lines": [num_train_lines, 2 * num_train_lines],
            }
        ],
        "finetuning_parameters": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "batch_size": 64,
            "num_steps": 60000,
        },
    }


def create_multituning_config(num_train_lines):
    return {
        "model_dir": f"experiments/exp1-multi-{num_train_lines}",
        "corpora": {
            "eng_Latn": {
                "train": "data/train.en",
                "dev": "data/dev.en",
                "test": "data/test.en",
                "permutation": 0,
            },
            "tsn_Latn": {
                "train": "data/train.de",
                "dev": "data/dev.de",
                "test": "data/test.de",
                "permutation": 1,
            },
            "tso_Latn": {
                "train": "data/train.de",
                "dev": "data/dev.de",
                "test": "data/test.de",
                "permutation": 2,
            },
        },
        "bitexts": [
            {"src": "eng_Latn", "tgt": "tsn_Latn", "train_lines": [0, num_train_lines]},
            {
                "src": "eng_Latn",
                "tgt": "tso_Latn",
                "train_lines": [num_train_lines, 2 * num_train_lines],
            },
        ],
        "finetuning_parameters": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "batch_size": 64,
            "num_steps": 60000,
        },
    }


def create_shell_script(num_train_lines):
    preface = "\n".join(
        [
            "#!/bin/sh",
            "#SBATCH -c 1",
            "#SBATCH -t 3-12:00",
            "#SBATCH -p dl",
            "#SBATCH --mem=10G",
            "#SBATCH -o logs/log_%j.out",
            "#SBATCH -e logs/log_%j.err",
            "#SBATCH --gres=gpu:1",
        ]
    )
    calls = "\n".join([
        f"python finetune.py --config configs/experiment1.bi1.{num_train_lines}.json",
        f"python finetune.py --config configs/experiment1.bi2.{num_train_lines}.json",
        f"python finetune.py --config configs/experiment1.multi.{num_train_lines}.json",
    ])
    return preface + "\n" + calls


for num_train_lines in [256, 512, 1024, 2048, 4096, 8192, 16834]:
    config = create_bituning_config1(num_train_lines)
    with open(f"configs/experiment1.bi1.{num_train_lines}.json", "w") as writer:
        json.dump(config, writer, indent=4)
    config = create_bituning_config2(num_train_lines)
    with open(f"configs/experiment1.bi2.{num_train_lines}.json", "w") as writer:
        json.dump(config, writer, indent=4)
    config = create_multituning_config(num_train_lines)
    with open(f"configs/experiment1.multi.{num_train_lines}.json", "w") as writer:
        json.dump(config, writer, indent=4)
    shell_script = create_shell_script(num_train_lines)
    with open(f"configs/run.exp1.{num_train_lines}.json", "w") as writer:
        writer.write(shell_script)
    