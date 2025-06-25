import os
import gc
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

from configure import USE_CUDA
from corpora import MixtureOfBitexts, TokenizedMixtureOfBitexts


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def tokenize(sents, lang: str, tokenizer: AutoTokenizer, max_length: int, alt_pad_token: int = None):
    tokenizer.src_lang = lang
    tokens = tokenizer(
        sents, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    if alt_pad_token is not None:
        tokens.input_ids[tokens.input_ids == tokenizer.pad_token_id] = alt_pad_token
    return tokens


def prepare_tokenizer_and_model(
    base_model: str, new_lang_codes, freeze_encoder: bool, model_dir: str
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    if freeze_encoder:
        print("--> ENCODER FROZEN <--")
        for param in model.get_encoder().parameters():
            param.requires_grad = False
    else:
        print("--> encoder NOT frozen <--")

    missing = [code for code in new_lang_codes if code not in tokenizer.get_vocab()]
    if missing:
        print("Augmenting vocabulary with:", missing)
        tokenizer.add_special_tokens({"additional_special_tokens": missing})
        model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(model_dir)

    if USE_CUDA:
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        model.cuda()

    return tokenizer, model


def evaluate(model, dev_data, tokenizer, max_length, batches: int = 100):
    model.eval()
    dev_losses = []
    with torch.no_grad():
        for _ in range(batches):
            x_input_ids, y_input_ids, x_mask, y_mask = dev_data.next_batch()
            x = {
                "input_ids": x_input_ids.to(model.device),
                "attention_mask": x_mask.to(model.device) 
                }
            y = {
                "input_ids": y_input_ids.to(model.device),
                "attention_mask": y_mask.to(model.device) 
                }
            loss = model(**x, labels=y["input_ids"]).loss
            dev_losses.append(loss.item())
    return np.mean(dev_losses)


def plot_losses(train_x, train_y, dev_x, dev_y, out_path: str):
    plt.clf()
    plt.plot(train_x, train_y, label="train", color="blue", linewidth=2)
    plt.plot(dev_x, dev_y, label="dev", color="red", linewidth=2)
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)


def finetune(
    model,
    tokenizer,
    train_data,
    dev_data,
    base_model: str,
    model_dir: str,
    training_steps: int,
    max_length: int = 128,
    report_every: int = 500,
    validate_every: int = 500,
    patience: int = 5,
    freeze_encoder: bool = False,
):
    print(f"Training {model_dir}")

    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

    cleanup()
    train_losses, train_plot_x, train_plot_y = [], [], []
    dev_plot_x, dev_plot_y = [], []
    best_dev_loss, steps_since_best = None, 0

    for i in tqdm(range(training_steps)):
        try:
            model.train()
            x_input_ids, y_input_ids, x_mask, y_mask = train_data.next_batch()
            x = {
                "input_ids": x_input_ids.to(model.device),
                "attention_mask": x_mask.to(model.device) 
                }
            y = {
                "input_ids": y_input_ids.to(model.device),
                "attention_mask": y_mask.to(model.device) 
                }
            loss = model(**x, labels=y["input_ids"]).loss
            loss.backward()
            train_losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM. Cleaning up.")
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise

        if i > 0 and i % report_every == 0:
            avg_train_loss = np.mean(train_losses[-report_every:])
            print(f"Step {i} (train): {avg_train_loss:.4f}")
            train_plot_x.append(i)
            train_plot_y.append(avg_train_loss)
            sys.stdout.flush()

        if i % validate_every == 0:
            print("Validating...")
            dev_loss = evaluate(model, dev_data, tokenizer, max_length)
            print(f"Dev loss: {dev_loss:.4f}")
            dev_plot_x.append(i)
            dev_plot_y.append(dev_loss)
            sys.stdout.flush()

            plot_losses(
                train_plot_x,
                train_plot_y,
                dev_plot_x,
                dev_plot_y,
                os.path.join(model_dir, "training.png"),
            )

            if best_dev_loss is None or dev_loss < best_dev_loss:
                print("Saving new best model.")
                best_dev_loss = dev_loss
                steps_since_best = 0
                model.save_pretrained(model_dir)
            else:
                steps_since_best += 1
                print(f"No improvement. Patience: {patience - steps_since_best}")
                if steps_since_best >= patience:
                    print("Early stopping.")
                    break


def main():
    parser = argparse.ArgumentParser(description="Finetune NLLB model.")
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory to save finetuned model"
    )
    parser.add_argument("--steps", type=int, default=60_000, help="Training steps")
    args = parser.parse_args()

    # Create unique model directory
    base_dir = args.model_dir
    model_version = 0
    while os.path.exists(f"{base_dir}-v{model_version}"):
        model_version += 1
    model_dir = f"{base_dir}-v{model_version}"
    os.makedirs(model_dir)

    raw_train_data = MixtureOfBitexts.create_from_files(
        {
            "en": "/mnt/storage/hopkins/data/europarl/preprocessed/train.en",
            "pl": "/mnt/storage/hopkins/data/europarl/preprocessed/train.pl",
        },
        [("en", "pl")],
        batch_size=32,
    )
    raw_dev_data = MixtureOfBitexts.create_from_files(
        {
            "en": "/mnt/storage/hopkins/data/europarl/preprocessed/dev.en",
            "pl": "/mnt/storage/hopkins/data/europarl/preprocessed/dev.pl",
        },
        [("en", "pl")],
        batch_size=32,
    )
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer, model = prepare_tokenizer_and_model(
    base_model=model_name,
    new_lang_codes=raw_train_data.get_language_codes(),
    freeze_encoder=False,
    model_dir=model_dir,
    )
    train_data = TokenizedMixtureOfBitexts(raw_train_data, tokenizer, max_length=128)
    dev_data   = TokenizedMixtureOfBitexts(raw_dev_data, tokenizer, max_length=128)
    finetune(
        model, tokenizer, train_data, dev_data, model_name, model_dir, args.steps, freeze_encoder=False
    )


if __name__ == "__main__":
    main()
