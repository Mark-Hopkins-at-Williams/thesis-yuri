import json
from pathlib import Path

EXP_NUM = 5
BASE_MODEL = "facebook/nllb-200-distilled-600M"
TGT = "en"
TGT_ID = "eng_Latn"
FREEZE_ENCODER = False
FREEZE_DECODER = True

variants = []
langs = ['cs', 'da', 'de', 'es', 'fi', 
         'et', 'fr', 'it', 'nl', 'pl', 
         'pt', 'ro', 'sk', 'sl', 'sv']
for i in range(len(langs)):
    for j in range(i+1, len(langs)):
        variants.append([langs[i], langs[j]])
    


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
        "model_dir": f"experiments/exp{EXP_NUM}-{VARIANT}/exp{EXP_NUM}-{VARIANT}-bi{src_index}-{num_train_lines}",
        "corpora": { 
            "europarl": {
                SRCS[src_index]: {
                    "lang_code": SRC_IDS[src_index],
                    "train": f"data/train.{SRCS[src_index]}",
                    "dev": f"data/dev.{SRCS[src_index]}",
                    "test": f"data/test.{SRCS[src_index]}",
                    "permutation": 1,
                },
                TGT: {
                    "lang_code": TGT_ID,
                    "train": f"data/train.{TGT}",
                    "dev": f"data/dev.{TGT}",
                    "test": f"data/test.{TGT}",
                    "permutation": 0,
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
        "finetuning_parameters": {
            "base_model": BASE_MODEL,
            "batch_size": 64,
            "num_steps": 60000,
            "freeze_encoder": FREEZE_ENCODER,
            "freeze_decoder": FREEZE_DECODER
        },
    }


def create_multituning_config(num_train_lines):
    corpora = { 
            "europarl": {
                TGT: {
                    "lang_code": TGT_ID,
                    "train": f"data/train.{TGT}",
                    "dev": f"data/dev.{TGT}",
                    "test": f"data/test.{TGT}",
                    "permutation": 0,
                }
            }
        }
    for src_index, src in enumerate(SRCS):
        corpora["europarl"][SRCS[src_index]] = {
            "lang_code": SRC_IDS[src_index],
            "train": f"data/train.{src}",
            "dev": f"data/dev.{src}",
            "test": f"data/test.{src}",
            "permutation": 1,
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
        "model_dir": f"experiments/exp{EXP_NUM}-{VARIANT}/exp{EXP_NUM}-{VARIANT}-multi-{num_train_lines}",
        "corpora": corpora,
        "bitexts": bitexts,
        "finetuning_parameters": {
            "base_model": BASE_MODEL,
            "batch_size": 64,
            "num_steps": 60000,
            "freeze_encoder": FREEZE_ENCODER,
            "freeze_decoder": FREEZE_DECODER
        },
    }


def create_shell_script_lines(num_train_lines):
    lines = []
    exp_config = config_dir / f"experiment{EXP_NUM}-{VARIANT}.multi.{num_train_lines}.json"
    lines.append(f"python finetune.py --config {exp_config}")
    for src_index in range(len(SRCS)):
        exp_config = config_dir / f"experiment{EXP_NUM}-{VARIANT}.bi{src_index}.{num_train_lines}.json"
        lines.append(f"python finetune.py --config {exp_config}",)
    return lines

preface = [
            "#!/bin/sh",
            "#SBATCH -c 1",
            "#SBATCH -t 3-12:00",
            "#SBATCH -p dl",
            "#SBATCH -o logs/log_%j.out",
            "#SBATCH -e logs/log_%j.err",
            "#SBATCH --gres=gpu:1",
        ]

shell_script_lines = []
for num_train_lines in [4096]:
    for VARIANT, SRCS in enumerate(variants):
        config_dir = Path(f"configs/exp{EXP_NUM}-{VARIANT}")
        config_dir.mkdir(parents=True, exist_ok=True)
        for src_index in range(len(SRCS)):
            config = create_bituning_config(num_train_lines, src_index)
            with open(
                config_dir / f"experiment{EXP_NUM}-{VARIANT}.bi{src_index}.{num_train_lines}.json", "w"
            ) as writer:
                json.dump(config, writer, indent=4)
        config = create_multituning_config(num_train_lines)
        with open(
            config_dir / f"experiment{EXP_NUM}-{VARIANT}.multi.{num_train_lines}.json", "w"
        ) as writer:
            json.dump(config, writer, indent=4)
        shell_script_lines.extend(create_shell_script_lines(num_train_lines))

with open(Path(f"configs") / f"run.exp{EXP_NUM}-all.{num_train_lines}.sh", "w") as writer:
    writer.write("\n".join(preface + shell_script_lines))
