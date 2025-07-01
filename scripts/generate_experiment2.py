import json

VARIANT = 2

if VARIANT == 1:
    SRC = "en"
    TGT1 = "es"
    TGT2 = "pt"
    SRC_ID = "eng_Latn"
    TGT1_ID = "tsn_Latn"
    TGT2_ID = "tso_Latn"
elif VARIANT == 2:
    SRC = "en"
    TGT1 = "de"
    TGT2 = "nl"
    SRC_ID = "eng_Latn"
    TGT1_ID = "tsn_Latn"
    TGT2_ID = "tso_Latn"

def create_bituning_config1(num_train_lines):
    return {
        "model_dir": f"experiments/exp2-{VARIANT}-bi1-{num_train_lines}",
        "corpora": {
            "eng_Latn": {
                "train": f"data/train.{SRC}",
                "dev": f"data/dev.{SRC}",
                "test": f"data/test.{SRC}",
                "permutation": 0,
            },
            "tsn_Latn": {
                "train": f"data/train.{TGT1}",
                "dev": f"data/dev.{TGT1}",
                "test": f"data/test.{TGT1}",
                "permutation": 1,
            },
        },
        "bitexts": [
            {"src": SRC_ID, "tgt": TGT1_ID, "train_lines": [0, num_train_lines]}
        ],
        "finetuning_parameters": {
            "base_model": "facebook/nllb-200-distilled-600M",
            "batch_size": 64,
            "num_steps": 60000,
        },
    }


def create_bituning_config2(num_train_lines):
    return {
        "model_dir": f"experiments/exp2-{VARIANT}-bi2-{num_train_lines}",
        "corpora": {
            "eng_Latn": {
                "train": f"data/train.{SRC}",
                "dev": f"data/dev.{SRC}",
                "test": f"data/test.{SRC}",
                "permutation": 0,
            },
            "tso_Latn": {
                "train": f"data/train.{TGT2}",
                "dev": f"data/dev.{TGT2}",
                "test": f"data/test.{TGT2}",
                "permutation": 1,
            },
        },
        "bitexts": [
            {
                "src": SRC_ID,
                "tgt": TGT2_ID,
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
        "model_dir": f"experiments/exp2-{VARIANT}-multi-{num_train_lines}",
        "corpora": {
            "eng_Latn": {
                "train": f"data/train.{SRC}",
                "dev": f"data/dev.{SRC}",
                "test": f"data/test.{SRC}",
                "permutation": 0,
            },
            "tsn_Latn": {
                "train": f"data/train.{TGT1}",
                "dev": f"data/dev.{TGT1}",
                "test": f"data/test.{TGT1}",
                "permutation": 1,
            },
            "tso_Latn": {
                "train": f"data/train.{TGT2}",
                "dev": f"data/dev.{TGT2}",
                "test": f"data/test.{TGT2}",
                "permutation": 1,
            },
        },
        "bitexts": [
            {"src": SRC_ID, "tgt": TGT1_ID, "train_lines": [0, num_train_lines]},
            {
                "src": SRC_ID,
                "tgt": TGT2_ID,
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
        f"python finetune.py --config configs/experiment2.{VARIANT}.bi1.{num_train_lines}.json",
        f"python finetune.py --config configs/experiment2.{VARIANT}.bi2.{num_train_lines}.json",
        f"python finetune.py --config configs/experiment2.{VARIANT}.multi.{num_train_lines}.json",
    ])
    return preface + "\n" + calls


for num_train_lines in [256, 512, 1024, 2048, 4096, 8192, 16834]:
    config = create_bituning_config1(num_train_lines)
    with open(f"configs/experiment2.{VARIANT}.bi1.{num_train_lines}.json", "w") as writer:
        json.dump(config, writer, indent=4)
    config = create_bituning_config2(num_train_lines)
    with open(f"configs/experiment2.{VARIANT}.bi2.{num_train_lines}.json", "w") as writer:
        json.dump(config, writer, indent=4)
    config = create_multituning_config(num_train_lines)
    with open(f"configs/experiment2.{VARIANT}.multi.{num_train_lines}.json", "w") as writer:
        json.dump(config, writer, indent=4)
    shell_script = create_shell_script(num_train_lines)
    with open(f"configs/run.exp2.{VARIANT}.{num_train_lines}.json", "w") as writer:
        writer.write(shell_script)
    