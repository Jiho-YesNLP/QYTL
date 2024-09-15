"""
This RoBERTa model classifies sentences into interrogative and non-interrogative sentences. The following dataset is used for training and testing.
IntVsDecl dataset: https://www.kaggle.com/datasets/shahrukhkhan/questions-vs-statementsclassificationdataset?resource=download
"""

import os

import dotenv
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from openai import OpenAI

DATA_DIR = "data/IntVsDecl"
dotenv.load_dotenv()


def get_teacher_logits_batch(texts_batch, teacher):
    """
    Get teacher logits for a batch of texts
    """
    inst_str = (
        "Classify the following sentence into interrogative or "
        "non-interrogative. (Use labels '0' for non-interative and '1' for "
        "question)"
    )
    examples_str = [
        (("Alongside the capital, the most popular tourist destinations are "
         "Isfahan, Mashhad and Shiraz. What was Iran's rank in the top 10 "
         "Middle East destinations according to UNESCO"), 1),
        ("They merged into InBev, becoming the largest brewery.", 0),
        ("Were any terrorist groups involved in the Burmese conflicts ?", 1),
        ("Maxx, Costco, Sam's Club and others", 0)]

    messages = [{"role": "system", "content": inst_str}]
    messages.extend(
        [{"role": "user", "content": ex[0]},
         {"role": "assistant", "content" for ex in examples_str]
        {"role": "user", "content": },

    prompts = [
        {
            "prompt": prompt_str.format(text),
            "max_tokens": 1,
            "logprobs": 5,
            "n": 1,
            "stop": None,
        }
        for text in texts_batch
    ]
    responses = teacher.chat.completions.create(
        engine="text-davinci-003",  # Use the appropriate engine for GPT-3.5-turbo-0125
        batch_size=len(prompts),
        prompt=[prompt["prompt"] for prompt in prompts],
        max_tokens=1,
        logprobs=2,
        n=1,
        stop=None,
    )

    teacher_logits_batch = []
    for response in responses["choices"]:
        logprobs = response["logprobs"]["top_logprobs"][0]
        logits = [logprobs[token] for token in logprobs]
        teacher_logits_batch.append(logits)

    return torch.tensor(teacher_logits_batch)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    valid_df = pd.read_csv(f"{DATA_DIR}/val.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

    # Load RoBERTa Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("roberta-base")
    # teacher model
    gpt = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load datasets
    train_ds = TextDataset(
        train_df["doc"], train_df["target"].values, tokenizer, max_len=128
    )
    test_ds = TextDataset(
        test_df["doc"], test_df["target"].values, tokenizer, max_len=128
    )

    # GPU setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # train
    for epoch in range(3):
        model.train()
        for batch in DataLoader(train_ds, batch_size=32, shuffle=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label = batch["label"].to(device)

            # teacher inferences
            texts = batch["text"]
            teacher_logits = get_teacher_logits_batch(texts, gpt).to(device)

            # student model
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=label)
            student_logits = outputs.logits

            loss = outputs.loss
            loss.backward()
            optimizer.step()

        # print(f"Epoch: {epoch}, Loss: {loss}")
