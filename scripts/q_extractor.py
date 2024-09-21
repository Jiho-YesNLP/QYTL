"""
This script use a fine-tuned RoBERta q classification model to identify and
extract questions from the YouTube comments.
We apply a minimal text processing pipeline to the comments before feeding them
into the model:
    - remove line breaks
    - remove emojis and Unicode characters
    - split the comments into sentences

It returns a csv file in the following format:
    vidid, cid, subject, text, logprob, #replies, #votes, is_reply
"""

import code
import os
import argparse
import json
import csv
import logging

import nltk
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

# Logger both to file and console
logger = logging.getLogger("q_extractor")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("log/q_extractor.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class CommentsDataset(Dataset):
    def __init__(self, args, tokenizer, comment_file_path):
        self.args = args
        self.tokenizer = tokenizer
        self.comments = []
        self.load_comments(comment_file_path)

    def load_comments(self, filename):
        with open(os.path.join(self.args.input_dir, filename), "r") as f:
            comments_ = json.load(f)
            subject = filename.split("-")[-2]
            for c in comments_:
                cmmt = c["comment"].replace("\n", " ")
                cmmt = cmmt.encode("ascii", "ignore").decode()
                cmmt = nltk.sent_tokenize(cmmt)
                for sent in cmmt:
                    enc = self.tokenizer.encode_plus(
                        sent,
                        add_special_tokens=True,
                        max_length=512,
                        return_token_type_ids=False,
                        padding="max_length",
                        return_tensors="pt",
                        truncation=True,
                    )
                    self.comments.append(
                        {
                            "text": sent,
                            "input_ids": enc["input_ids"],
                            "attention_mask": enc["attention_mask"],
                            "vid": c["vid"],
                            "cid": c["cid"],
                            "subject": subject,
                            "votes": c["votes"],
                            "replies": c["replies"],
                            "is_reply": c["reply"],
                        }
                    )

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.comments[idx]


def extract_questions(args, model, cmmt_ds, filename="inference"):
    if len(cmmt_ds) == 0:
        return None

    questions = []
    model.eval()
    logits = []
    cmmt_sampler = SequentialSampler(cmmt_ds)
    cmmt_loader = DataLoader(
        cmmt_ds, sampler=cmmt_sampler, batch_size=32, num_workers=0
    )

    for batch in cmmt_loader:
        input_ids = batch["input_ids"].squeeze(1).to(args.device)
        attention_mask = batch["attention_mask"].squeeze(1).to(args.device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits.append(outputs.logits.cpu().numpy())

    logits = np.concatenate(logits, axis=0)
    preds = np.argmax(logits, axis=1)
    for i, p in enumerate(preds):
        if p == 1:
            questions.append(cmmt_ds[i])
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/comments",
        help="Directory containing the comments",
    )
    parser.add_argument(
        "--output", type=str, default="data/output", help="Directory to save the output"
    )
    parser.add_argument(
        "--chkpt_path",
        type=str,
        required=True,
        default="data/output/checkputs/best_model.pt",
        help="Path to the model checkpoint",
    )
    parser.add_argument("--model_name", type=str, default="roberta-base")
    args = parser.parse_args()

    # Load the model
    if not os.path.exists(args.chkpt_path):
        raise ValueError(f"Model checkpoint not found at {args.chkpt_path}")

    # setup the device (GPU)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_config = RobertaConfig.from_pretrained(args.model_name)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name, config=model_config
    )

    # Load the model checkpoint
    model.load_state_dict(torch.load(args.chkpt_path, map_location=args.device))
    model.to(args.device)
    logger.info("Optimized q classifier model loaded successfully")

    for filename in os.listdir(args.input_dir):
        cmmt_ds = CommentsDataset(args, tokenizer, filename)
        # Extract questions
        questions = extract_questions(args, model, cmmt_ds, filename)
        if questions is None:
            logger.info(f"No questions found in {filename}.")
            continue

        # save the output
        csv_filename = filename.replace(".json", ".csv")
        with open(os.path.join(args.output, csv_filename), "w") as f:
            writer = csv.writer(f)
            header = ["vid", "cid", "subject", "text", "votes", "replies", "is_reply"]
            writer.writerow(header)
            for q in questions:
                writer.writerow(
                    [
                        q["vid"],
                        q["cid"],
                        q["subject"],
                        q["text"],
                        q["votes"],
                        q["replies"],
                        q["is_reply"],
                    ]
                )

        # log stats
        logger.info(
            f"\nExtracted questions {len(questions)} / {len(cmmt_ds)} "
            f"from {filename}."
        )


if __name__ == "__main__":
    main()
