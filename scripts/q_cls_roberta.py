"""
This RoBERTa model classifies sentences into interrogative and
non-interrogative sentences. The following dataset is used for training and
testing. IntVsDecl dataset: https://tinyurl.com/4ah7w3u6
https://tinyurl.com/25eecdyr
"""

import code
import os
import argparse
import random
import logging
import numpy as np
import dotenv

from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)
from openai import OpenAI

DATA_DIR = "data/IntVsDecl"
dotenv.load_dotenv()

logger = logging.getLogger(__name__)


class IntVsDeclDataset(Dataset):
    def __init__(self, df, tokenizer, args, ds_type="train"):
        self.tokenizer = tokenizer
        self.args = args
        self.examples = []
        for i in tqdm(range(len(df)), desc=f"Loading {ds_type} dataset"):
            text = df.iloc[i]["doc"]
            label = df.iloc[i]["target"]
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=args.seq_length,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
            )
            self.examples.append(
                {
                    "text": text,
                    "input_ids": encoding["input_ids"].flatten(),
                    "attention_mask": encoding["attention_mask"].flatten(),
                    "label": torch.tensor(label, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def train(args, model, train_ds, val_ds):
    sampler = RandomSampler(train_ds)
    tr_loader = DataLoader(
        train_ds, sampler=sampler, batch_size=args.train_batch_size, num_workers=0
    )

    args.max_steps = args.epochs * len(tr_loader)
    args.save_steps = len(tr_loader) // 2
    args.warmup_steps = args.max_steps // 10
    model.to(args.device)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_ds))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    loss_stats = {"loss": [], "avg_loss": []}
    best_f1 = 0.0

    model.zero_grad()
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(tr_loader):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            label = batch["label"].to(args.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            loss_stats["avg_loss"].append(loss.item())

            if global_step % args.log_interval == 0:
                avg_loss = sum(loss_stats["avg_loss"]) / len(loss_stats["avg_loss"])
                loss_stats["avg_loss"] = []
                loss_stats["loss"].append(avg_loss)
                logger.info("Step: %d, Loss: %.4f", global_step, avg_loss)

            if global_step != 0 and global_step % args.save_steps == 0:
                logger.info("Step: %d, Loss: %.4f", global_step, loss.item())
                result = evaluate(args, model, val_ds)

                if result["f1-score"] > best_f1:
                    best_f1 = result["f1-score"]
                    file_path = os.path.join(
                        args.output_dir, "checkpoints/best_model.pt"
                    )
                    model_to_save = model.module if hasattr(model, "module") else model
                    torch.save(model_to_save.state_dict(), file_path)
                    logger.info("Model saved at %s", file_path)

            global_step += 1


def evaluate(args, model, val_ds):
    model.eval()
    val_sampler = SequentialSampler(val_ds)
    val_loader = DataLoader(
        val_ds, sampler=val_sampler, batch_size=args.eval_batch_size, num_workers=0
    )

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(val_ds))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    logits = []
    labels = []

    for batch in tqdm(val_loader, desc="validation"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        label = batch["label"].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            eval_loss += outputs.loss.item()
            logits.append(outputs.logits.cpu().numpy())
            labels.append(label.cpu().numpy())

    # validation loss
    eval_loss = eval_loss / len(val_loader)
    logger.info("Validation Loss: %.4f", eval_loss)

    # classification report
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.argmax(logits, axis=1)

    result = classification_report(labels, preds, output_dict=True)
    print(result)

    return result["weighted avg"]


def test(args, model, test_ds):
    model.eval()
    test_sampler = SequentialSampler(test_ds)
    test_loader = DataLoader(
        test_ds, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0
    )

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_ds))
    logger.info("  Batch size = %d", args.eval_batch_size)

    logits = []
    labels = []

    for batch in tqdm(test_loader, desc="testing"):
        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        label = batch["label"].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            logits.append(outputs.logits.cpu().numpy())
            labels.append(label.cpu().numpy())

    # classification report
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.argmax(logits, axis=1)

    result = classification_report(labels, preds, output_dict=True)
    print(result)

    return


def set_seed(seed, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    # data parameters
    # fmt: off
    parser.add_argument("--input_dir", type=str, default="data/IntVsDecl")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--val_file", type=str, default="val.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--chkpt_path", type=str, 
                        default="data/output/checkpoints/best_model.pt",
                        help="Path to the model checkpoint")
    # model parameters
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Model architecture to be fine-tuned")
    # runtime parameters
    parser.add_argument("--seq_length", type=int, default=512,
                        help="Maximum sequence length after tokenization."
                        "Default to the max input length for the model")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of epochs for training")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Interval for logging")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for training")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for training")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for training")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Warmup steps for training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training")
    parser.add_argument("--do_test", action="store_true",
                        help="Whether to run evaluation")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run prediction")

    # RQ0: Efficacy of Knowledge Distillation
    parser.add_argument("--enable_kd", action="store_true", 
                        help="Enable knowledge distillation")
    # fmt: on

    args = parser.parse_args()

    # setup the device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.enable_kd:
        args.train_file = "q_cls_gpt_annotated.csv"
    # TODO. set filename for the model to be saved

    # setup logging
    logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
    logger.info(f"Using device: {args.device}")

    # set seed for everything
    set_seed(args.seed)

    # Load the model and tokenizer
    model_config = RobertaConfig.from_pretrained(args.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name, config=model_config
    )
    logger.info("Configurations: %s", args)

    # Training
    if args.do_train:
        # Load the datasets
        train_df = pd.read_csv(f"{args.input_dir}/{args.train_file}")
        val_df = pd.read_csv(f"{args.input_dir}/{args.val_file}")

        train_dataset = IntVsDeclDataset(train_df, tokenizer, args, ds_type="train")
        val_dataset = IntVsDeclDataset(val_df, tokenizer, args, ds_type="val")
        train(args, model, train_dataset, val_dataset)
    if args.do_test:
        if not os.path.exists(args.chkpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {args.chkpt_path}")

        model.load_state_dict(torch.load(args.chkpt_path, map_location=args.device))
        model.to(args.device)
        logging.info("Model loaded from %s", args.chkpt_path)

        test_df = pd.read_csv(f"{args.input_dir}/{args.test_file}")
        test_dataset = IntVsDeclDataset(test_df, tokenizer, args, ds_type="test")
        test(args, model, test_dataset)


if __name__ == "__main__":
    main()
