"""
The exam question dataset contains 2534 examples. We can obtain random samples (approximately 20%) from this dataset for test and validation data and use the rest to generate augmented data for each subject of STEM. We give a topic in a particular subject (e.g., binary tree in CS) and select a Bloom cog level (e.g., analyize), provide some examples questions that belong to the same category, and ask LLM for generating similar type of questinos. Finally, we can run a predictive model on the test data to see how much improvement we can achieve by the data augmentation process.

class distribution:
    Knowledge: 344, Comprehension: 961, Application: 316, Analysis: 304, Synthesis: 279
"""

import code
import os
import random
import argparse
import json
import csv
import re

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

BT_LEVELS = [
    "Knowledge",
    "Comprehension",
    "Application",
    "Analysis",
    "Evaluation",
    "Synthesis",
]
SUBJECTS = ["cs", "math", "physics", "chemistry", "biology"]

# GPT Prompt
SYS = (
    "Bloom's Taxonomy categorizes cognitive levels into six levels, each described by a specific action verb:\n"
    "- Knowledge: define, match, recall, state, list, label\n"
    "- Comprehension: discuss, review, paraphrase, describe, explain\n"
    "- Application: apply, demonstrate, illustrate, solve, use\n"
    "- Analysis: analyze, compare, contrast, differentiate, distinguish\n"
    "- Evaluation: argue, conclude, critique, evaluate, justify, verify\n"
    "- Synthesis: create, design, develop, formulate, organize, plan\n\n"
    "The following are examples of questions that correspond to one of the Bloom's Cognitive Levels. Please write a question that aligns with the given Bloom's Cognitive Level and pertains to a STEM topic.\n\n"
)


def read_subject_topics():
    topics = dict.fromkeys(SUBJECTS)
    # Read the topics for each subject from the files
    for subject in topics:
        with open(f"data/stem_topics/{subject}_topics.txt", "r") as f:
            topics[subject] = f.read().splitlines()
        print(f"{subject}: {len(topics[subject])} topics read")
    return topics


def data_augment(questions, topics, gpt, n_aug=3):
    """
    For each question in the training set, generate a question given the
    type of BT cognitive level and the topic using an LLM model.
    """
    inst1_tpl = "Generate a question that belongs to the {} level."
    inst2_tpl = "Generate a {} question on {} in {}."

    # Preparing a batch file
    # Per each question, generate n_aug questions on a random topic with the same BT Level
    topic_list = [(subj, topic) for subj in topics for topic in topics[subj]]
    sample_topics = random.choices(topic_list, k=n_aug * len(questions))
    batch = []
    for i, row in tqdm(questions.iterrows(), total=len(questions)):
        for _ in range(n_aug):
            messages = [{"role": "system", "content": SYS}]
            # add few-shot examples
            for bt in BT_LEVELS:
                sample = questions[questions["BT LEVEL"] == bt].sample(1)
                messages.append(
                    {
                        "role": "user",
                        "content": inst1_tpl.format(sample["BT LEVEL"].iloc[0]),
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": " ".join(sample["QUESTION"].iloc[0].split()),
                    }
                )
            j = len(batch)
            messages.append(
                {
                    "role": "user",
                    "content": inst2_tpl.format(
                        row["BT LEVEL"], sample_topics[j][1], sample_topics[j][0]
                    ),
                }
            )
            batch.append(messages)
    with open("data/exam_questions_aug/batch.jsonl", "w") as f:
        for i, row in enumerate(batch):
            entry = {
                "custom_id": "req-{}".format(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-3.5-turbo-0125",
                    "messages": row,
                    "max_tokens": 100,
                },
            }
            f.write(json.dumps(entry) + "\n")

    # Uploading batch input file
    print("Submitting OpenAI batch request...")
    batch_file = gpt.files.create(
        file=open("data/exam_questions_aug/batch.jsonl", "rb"), purpose="batch"
    )
    # Creating the batch request
    batch_resp = gpt.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "Question generation for data augmentation"},
    )
    print(f"batch chat completion requested: {batch_resp.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_aug", type=int, default=3)
    parser.add_argument(
        "--batch_id", type=str, help="batch id for OpenAI reponse retrieval"
    )
    args = parser.parse_args()

    load_dotenv()
    gpt = OpenAI(api_key=os.environ.get("GPT_API_KEY"))

    if args.batch_id:
        # check the status and retrieve response if ready
        resp = gpt.batches.retrieve(args.batch_id)

        request_fp = f"data/exam_questions_aug/batch.jsonl"
        results_fp = f"data/exam_questions_aug/{args.batch_id}-results.jsonl"
        if resp.status == "completed":
            print("Batch completed")
            # download the results
            results = gpt.files.content(resp.output_file_id)
            with open(results_fp, "wb") as f:
                f.write(results.read())
            print(f"Results saved to {results_fp}")

        # read batch
        batch_req = []
        batch_resp = []
        for line in open(request_fp, "r"):
            batch_req.append(json.loads(line))
        for line in open(results_fp, "r"):
            batch_resp.append(json.loads(line))

        assert len(batch_req) == len(batch_resp), "Batch request and response mismatch"
        # save a csv file with the generated questions
        p = re.compile(r"^Generate a (\w+) question on (.+) in (.+).$")
        with open("data/exam_questions_aug/train_aug.csv", "w") as f:
            writer = csv.writer(f)
            # write header
            writer.writerow(["QUESTION", "BT LEVEL"])
            for req, resp in zip(batch_req, batch_resp):
                # I need to extract the BT level from the content manually.
                # I could not find a way to pass this information through the request API.
                m = p.match(req["body"]["messages"][-1]["content"])
                if m:
                    bt_level = m.group(1)
                else:
                    bt_level = "Knowledge"
                writer.writerow(
                    [
                        resp["response"]["body"]["choices"][0]["message"]["content"],
                        bt_level,
                    ]
                )
        print("Generated questions saved to data/exam_questions_aug/train_aug.csv")
    else:
        # Load the original exam question dataset
        df_exam_questions = pd.read_csv(
            "data/exam_questions_aug/exam_questions_original.csv"
        )

        # shuffle
        df_exam_questions = df_exam_questions.sample(frac=1).reset_index(drop=True)

        # Save 20% of the data for validation and test data
        n = len(df_exam_questions) // 5
        # save valid.csv and test.csv
        df_exam_questions[:n].to_csv("data/exam_questions_aug/valid.csv", index=False)
        df_exam_questions[n : 2 * n].to_csv(
            "data/exam_questions_aug/test.csv", index=False
        )
        df_exam_questions[2 * n :].to_csv(
            "data/exam_questions_aug/train.csv", index=False
        )

        topics = read_subject_topics()
        data_augment(df_exam_questions[2 * n :], topics, gpt, n_aug=3)
