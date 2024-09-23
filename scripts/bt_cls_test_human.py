"""
Run Bloom Taxonomy classification on human annotated data.
Find the best probability threshold for 'irrelevant' class to maximize F1
score. The best threshold is used to classify the test data.

Test with two models (trained with raw data and augmented data) and compare.
"""

import code
import os
import argparse
import logging
import random
from tqdm import tqdm

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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

# logger setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BT_LEVELS = [
    "knowledge",
    "comprehension",
    "application",
    "analysis",
    "evaluation",
    "synthesis",
]
BT_LEVEL_MAP = {l: i for i, l in enumerate(BT_LEVELS)}


class QBTDataset(Dataset):  # modified for using human data
    def __init__(self, df, tokenizer, args, ds_type="train"):
        self.tokenizer = tokenizer
        self.args = args
        self.examples = []
        for i in tqdm(df.index):  # , desc=f"Loading {ds_type} dataset"):
            q = df.iloc[i]["text"]
            label = -1
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
                "qid": df.iloc[i]["qid"],
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
    parser.add_argument("--input_dir", type=str, default="data")
    parser.add_argument("--test_file", type=str, default="q_bt_human_label.csv")
    parser.add_argument("--chkpt_path", type=str, 
                        default="data/output/checkpoints/best_bt_cls_model.pt",
                        help="Path to the model checkpoint")
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
    parser.add_argument("--log_interval", type=int, default=64,
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
    # fmt: on

    args = parser.parse_args()

    # set seed for everything
    set_seed(args.seed)

    # setup the device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read test data file
    df = pd.read_csv(os.path.join(args.input_dir, args.test_file))
    # give a unique id to each row
    df["qid"] = range(len(df))
    logger.info("Read %d rows from %s", len(df), args.test_file)

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    test_ds = QBTDataset(df, tokenizer, args, ds_type="human")

    # load model
    model = BTClassifier(args.model_name)
    logger.info("Configurations: %s", args)

    # load the model
    model.load_state_dict(torch.load(args.chkpt_path, map_location=args.device))
    model.to(args.device)
    logging.info("Model loaded from %s", args.chkpt_path)

    # run predictions
    model.eval()
    inf_sampler = SequentialSampler(test_ds)
    loader = DataLoader(
        test_ds,
        sampler=inf_sampler,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    logger.info("***** Running predictions *****")
    logger.info("  Num examples = %d", len(test_ds))
    logger.info("  Batch size = %d", args.eval_batch_size)

    results = []

    for batch in tqdm(loader, desc="predicting"):
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
                    batch["qid"][i].item(),
                    max_probs.indices[i].item(),
                    max_probs.values[i].item(),
                )
            )
    results = {r[0]: (r[1], r[2]) for r in results}

    pred = []
    labels = []
    for i, row in df[df["label"] != "irrelevant"].iterrows():
        pred.append(results[row["qid"]][0])
        labels.append(BT_LEVEL_MAP[row["label"]])

    print(classification_report(labels, pred))
    code.interact(local=locals())


if __name__ == "__main__":
    main()
