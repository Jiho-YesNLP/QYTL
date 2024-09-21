"""
A RoBERTa model for BT-level classification of questions.
Using the {} dataset. This dataset is unbalanced, so we will need to do
something about that. Perhaps, WeightedRandomSampler in PyTorch.

"""

import code
import os
import argparse
import logging
import random
import csv

from tqdm import tqdm, trange

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report

import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
    SequentialSampler,
)
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    RobertaConfig,
    RobertaTokenizer,
    RobertaModel,
)

# Logger both to file and console
logger = logging.getLogger("bt_cls")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class QBTDataset(Dataset):
    def __init__(self, df, tokenizer, args, ds_type="train"):
        self.tokenizer = tokenizer
        self.args = args
        self.examples = []
        for i in tqdm(df.index):  # , desc=f"Loading {ds_type} dataset"):
            q = df.iloc[i]["QUESTION"] if ds_type != "yt" else df.iloc[i]["text"]
            label = df.iloc[i]["BT LEVEL"] if ds_type != "yt" else -1
            encoding = tokenizer.encode_plus(
                q,
                add_special_tokens=True,
                max_length=args.seq_length,
                return_token_type_ids=False,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                truncation=True,
            )
            entry = {
                "sid": i,
                "text": q,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "label": torch.tensor(label, dtype=torch.long),
            }
            self.examples.append(entry)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class BTClassifier(torch.nn.Module):
    def __init__(self, model_name, num_classes=6):
        super(BTClassifier, self).__init__()
        self.mdl_config = RobertaConfig.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name, config=self.mdl_config)
        self.pre_classifier = torch.nn.Linear(
            self.model.config.hidden_size, self.model.config.hidden_size
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        pooler = hidden_states[:, 0]
        pooler = torch.nn.ReLU()(self.pre_classifier(pooler))
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        else:
            return logits


def train(args, model, train_ds, val_ds, sampler):
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

            loss, _ = model(input_ids, attention_mask=attention_mask, labels=label)
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
                        args.output_dir, "checkpoints/best_bt_cls_model.pt"
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
            eval_loss, outputs = model(
                input_ids, attention_mask=attention_mask, labels=label
            )
            logits.append(outputs.cpu().numpy())
            labels.append(label.cpu().numpy())

    # validation loss
    eval_loss = eval_loss / len(val_loader)
    logger.info("Validation Loss: %.4f", eval_loss)

    # classification report
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    preds = np.argmax(logits, axis=1)

    result = classification_report(labels, preds, output_dict=True)
    print(result["weighted avg"])

    return result["weighted avg"]


def set_seed(seed, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    # data parameters
    parser.add_argument("--input_dir", type=str, default="data/q_bt")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--test_file", type=str, default="test.csv")
    parser.add_argument("--output_dir", type=str, default="data/output")
    parser.add_argument("--chkpt_path", type=str, 
                        default="data/output/checkpoints/btcls_best.pt",
                        help="Path to the model checkpoint")
    # model parameters
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Model architecture to be fine-tuned")
    # runtime parameters
    parser.add_argument("--seq_length", type=int, default=126,
                        help="Maximum sequence length after tokenization."
                        "Default to the max input length for the model")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=6,
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
    # fmt: on

    args = parser.parse_args()

    # setup the device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the data
    logger.info("Loading training data")
    train_df = pd.read_csv(os.path.join(args.input_dir, args.train_file))
    test_df = pd.read_csv(os.path.join(args.input_dir, args.test_file))

    # describe the data
    print(train_df.head())
    print(train_df.groupby("BT LEVEL").size())

    # mapping categories to integers
    bt_levels = [
        "Knowledge",
        "Comprehension",
        "Application",
        "Analysis",
        "Synthesis",
        "Evaluation",
    ]
    train_df["BT LEVEL"] = train_df["BT LEVEL"].apply(lambda x: bt_levels.index(x))
    test_df["BT LEVEL"] = test_df["BT LEVEL"].apply(lambda x: bt_levels.index(x))

    # split the training data into training and validation
    train_df = train_df.sample(frac=0.9).reset_index(drop=True)
    val_df = train_df.sample(frac=0.1).reset_index(drop=True)

    # set seed for everything
    set_seed(args.seed)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = BTClassifier(args.model_name)
    logger.info("Configurations: %s", args)

    if args.do_train:
        train_dataset = QBTDataset(train_df, tokenizer, args)
        val_dataset = QBTDataset(val_df, tokenizer, args)

        # random weight setup
        class_counts = train_df["BT LEVEL"].value_counts()
        sample_weights = [1 / class_counts[i].item() for i in train_df["BT LEVEL"]]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )

        train(args, model, train_dataset, val_dataset, sampler)
    if args.do_predict:
        """
        Load the model from the checkpoint and run predictions on YouTube
        question files.
        """
        if args.chkpt_path is None or not os.path.exists(args.chkpt_path):
            logger.error("Checkpoint path not provided")
            raise FileNotFoundError("Checkpoint path not provided")
        # read files and create a dataset
        cols = ["subject", "vid", "text", "votes", "replies", "is_reply"]
        yt_df = pd.DataFrame(columns=cols)
        entries = []
        for file in tqdm(os.listdir("data/output/ext_questions")):
            if file.endswith(".csv"):
                df = pd.read_csv(f"data/output/ext_questions/{file}")
                df["vid"] = "-".join(file.split("-")[:-2])
                if len(df) == 0:
                    continue
                entries.append(df[cols])
            yt_df = pd.concat(entries, ignore_index=True)

        yt_df["BT LEVEL"] = "UNK"
        yt_df["PROB"] = 0.0
        yt_dataset = QBTDataset(yt_df, tokenizer, args, ds_type="yt")

        # load the model
        model.load_state_dict(torch.load(args.chkpt_path, map_location=args.device))
        model.to(args.device)
        logging.info("Model loaded from %s", args.chkpt_path)

        # run predictions
        model.eval()
        inf_sampler = SequentialSampler(yt_dataset)
        yt_loader = DataLoader(
            yt_dataset,
            sampler=inf_sampler,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )

        logger.info("***** Running predictions *****")
        logger.info("  Num examples = %d", len(yt_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        results = []
        for batch in tqdm(yt_loader, desc="predicting"):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            # get softmax probabilities
            probs = torch.nn.Softmax(dim=1)(outputs)
            # take the max probability
            max_probs = torch.max(probs, dim=1)
            for i in range(len(input_ids)):
                results.append(
                    (
                        batch["sid"][i].item(),
                        max_probs.indices[i].item(),
                        max_probs.values[i].item(),
                    )
                )

        # add results to the dataframe and save to a csv file
        for sid, pred, prob in results:
            yt_df.loc[sid, "BT LEVEL"] = bt_levels[pred]
            yt_df.loc[sid, "PROB"] = prob

        yt_df.to_csv("data/output/q_bt_pred.csv", index=False)


if __name__ == "__main__":
    main()
