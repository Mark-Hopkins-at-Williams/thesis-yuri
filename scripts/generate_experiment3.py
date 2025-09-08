import json
from pathlib import Path

VARIANT = 5

# defaults
BASE_MODEL = "facebook/nllb-200-distilled-600M"
DATA_DIR = "data/"
FREEZE_ENCODER = False
FREEZE_DECODER = True
TGT = "en"
TGT_ID = "eng_Latn"

if VARIANT == 1:
    SRC = "es"
    SRCS = [f'{SRC}{i}' for i in range(2)]
elif VARIANT == 2:
    BASE_MODEL = "experiments/jpn-eng-pretrained-v1/"
    SRC = "es"
    SRCS = [f'{SRC}{i}' for i in range(2)]
elif VARIANT == 3:
    DATA_DIR = "/mnt/storage/hopkins/data/americasnlp2025/ST1_MachineTranslation/data/wayuu-spanish/prebatched"
    TGT = "es"
    TGT_ID = "spa_Latn"
    SRC = "guc"
    SRCS = [f'{SRC}{i}' for i in range(2)]
elif VARIANT == 4:
    SRC = "cs"
    SRCS = [f'{SRC}{i}' for i in range(2)]
elif VARIANT == 5:
    DATA_DIR = "/mnt/storage/hopkins/data/inuktitut"
    SRC = "iu"
    SRCS = [f'{SRC}{i}' for i in range(2)]
   

FT_PARAMS = {
    "base_model": BASE_MODEL,
    "freeze_encoder": FREEZE_ENCODER,
    "freeze_decoder": FREEZE_DECODER,
    "batch_size": 64,
    "num_steps": 60000,
}

SRC_IDS = [
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "umb_Latn",
    "uzn_Latn",
    "vec_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
]


def create_bituning_config(num_train_lines, src_index):
    return {
        "model_dir": f"experiments/exp3-{VARIANT}/exp3-{VARIANT}-bi{src_index}-{num_train_lines}",
        "corpora": { 
            "europarl": {
                TGT: {
                    "lang_code": TGT_ID,
                    "train": f"{DATA_DIR}/train.{TGT}",
                    "dev": f"{DATA_DIR}/dev.{TGT}",
                    "test": f"{DATA_DIR}/test.{TGT}",
                    "permutation": 0,
                },
                SRCS[src_index]: {
                    "lang_code": SRC_IDS[src_index],
                    "train": f"{DATA_DIR}/train.{SRC}",
                    "dev": f"{DATA_DIR}/dev.{SRC}",
                    "test": f"{DATA_DIR}/test.{SRC}",
                    "permutation": 1,
                },
            }
        },
        "bitexts": [
            {
                "corpus": "europarl",
                "src": SRCS[src_index],
                "tgt": TGT,
                "train_lines": [
                    src_index * num_train_lines,
                    src_index * num_train_lines + num_train_lines,
                ],
            }
        ],
        "finetuning_parameters": FT_PARAMS,
    }


def create_multituning_config(num_train_lines):
    corpora = { 
        "europarl": {
            TGT: {
                "lang_code": TGT_ID,
                "train": f"{DATA_DIR}/train.{TGT}",
                "dev": f"{DATA_DIR}/dev.{TGT}",
                "test": f"{DATA_DIR}/test.{TGT}",
                "permutation": 0,
            }
        }
    }
    for src_index, src in enumerate(SRCS):
        corpora["europarl"][SRCS[src_index]] = {
            "lang_code": SRC_IDS[src_index],
            "train": f"{DATA_DIR}/train.{SRC}",
            "dev": f"{DATA_DIR}/dev.{SRC}",
            "test": f"{DATA_DIR}/test.{SRC}",
            "permutation": 1 + src_index,
        }
    bitexts = [
        {
            "corpus": "europarl",
            "src": SRCS[src_index],
            "tgt": TGT,
            "train_lines": [
                src_index * num_train_lines,
                src_index * num_train_lines + num_train_lines,
            ],
        }
        for src_index in range(len(SRCS))
    ]
    return {
        "model_dir": f"experiments/exp3-{VARIANT}/exp3-{VARIANT}-multi-{num_train_lines}",
        "corpora": corpora,
        "bitexts": bitexts,
        "finetuning_parameters": FT_PARAMS,
    }


def create_shell_script(num_train_lines):
    preface = [
        "#!/bin/sh",
        "#SBATCH -c 1",
        "#SBATCH -t 3-12:00",
        "#SBATCH -p dl",
        "#SBATCH -o logs/log_%j.out",
        "#SBATCH -e logs/log_%j.err",
        "#SBATCH --gres=gpu:1",
    ]
    exp_config = config_dir / f"experiment3-{VARIANT}.multi.{num_train_lines}.json"
    preface.append(f"python finetune.py --config {exp_config}")
    for src_index in range(len(SRCS)):
        exp_config = (
            config_dir / f"experiment3-{VARIANT}.bi{src_index}.{num_train_lines}.json"
        )
        preface.append(
            f"python finetune.py --config {exp_config}",
        )
    return "\n".join(preface)


config_dir = Path(f"configs/exp3-{VARIANT}")
config_dir.mkdir(parents=True, exist_ok=True)
for num_train_lines in [1024, 2048, 4096, 8192, 16384]:
    for src_index in range(len(SRCS)):
        config = create_bituning_config(num_train_lines, src_index)
        with open(
            config_dir / f"experiment3-{VARIANT}.bi{src_index}.{num_train_lines}.json",
            "w",
        ) as writer:
            json.dump(config, writer, indent=4)
    config = create_multituning_config(num_train_lines)
    with open(
        config_dir / f"experiment3-{VARIANT}.multi.{num_train_lines}.json", "w"
    ) as writer:
        json.dump(config, writer, indent=4)
    shell_script = create_shell_script(num_train_lines)
    with open(config_dir / f"run.exp3-{VARIANT}.{num_train_lines}.sh", "w") as writer:
        writer.write(shell_script)
