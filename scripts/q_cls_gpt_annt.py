"""
This script uses a GPT model to annotate the IntVsDecl dataset with two
classes: Interrogative and Declarative. This will ask whether a given
sentence is a question or not, and write the results with their logprobs that
can be used for the model's confidence level with the prediction.

Steps:
    1. prepare .jsonl batch file with IntVsDecl/train.cvs
    2. create a batch resquest
    3. when the request is completed, calculate the accuracy and save the
    annotated dataset file for future training.

"""

import code
import os
import argparse
import json

from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


SYS = """
You will be asked to classify the following sentence into 
interrogative (question) or non-interrogative (statement). Use the labels
'0' for non-interrogative and '1' for interrogative.\n"""

# fmt: off
FEWSHOT_EXAMPLES = [
    ( "In 1952, Thomas Watson, Sr. In what year did IBM open its first office in Poughkeepsie", 1,),
    ( "What often lacks in software developed when its released that can eventually lead to errors?", 1,),
    ("there's nothing really that gets in that early", 0), 
    ( "The President of BYU, currently Kevin J Worthen, reports to the Board, through the Commissioner of Education.", 0,),
]
# fmt: on


def batch_request(args):
    """
    Prepare the batch file and create a batch request.
    """
    print("Preparing the batch file...")
    # Read the test dataset
    train_df = pd.read_csv(args.train_file)
    if args.num_examples > 0:
        train_df = train_df.head(args.num_examples)
    # chunking the dataframe
    max_req = 40000  # batch file can contain up to 50,000 requests
    list_df = [train_df[i : i + max_req] for i in range(0, train_df.shape[0], max_req)]

    for i, df in enumerate(list_df):
        jsonl_file = os.path.join(
            args.data_dir, "IntVsDecl/q_cls_gpt_batch{}.jsonl".format(i)
        )
        with open(os.path.join(jsonl_file), "w") as f:
            for _, row in tqdm(df.iterrows(), total=len(df)):
                msg = [{"role": "system", "content": SYS}]
                if args.fewshot:
                    for s, t in FEWSHOT_EXAMPLES:
                        msg.append({"role": "user", "content": s})
                        msg.append({"role": "assistant", "content": t})
                msg.append({"role": "user", "content": row["doc"]})
                entry = {
                    "custom_id": "{}".format(row.iloc[0]),
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": args.model_name,
                        "messages": msg,
                        "max_tokens": 1,
                        "logprobs": True,
                    },
                }
                f.write(json.dumps(entry) + "\n")

        # Uploading batch input file
        batch_file = model.files.create(
            file=open(jsonl_file, "rb"),
            purpose="batch",
        )
        # Creating the batch
        batch_resp = model.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": "Question classification task"},
        )
        print(f"batch requested {batch_resp.id}")
    return


def retrieve_eval_save(args):
    """
    Retrieve the results of a batch request and evaluate the model.
    """
    print("Retrieving the batch results...")
    # Read the train dataset
    train_df = pd.read_csv(args.train_file)
    train_df.rename(columns={"Unnamed: 0": "tr_qid"}, inplace=True)
    # Add new columns for the predictions and logprobs
    train_df[["pred", "logprobs"]] = [None, 0.0]

    # Retrieve the results
    # - status check; all the batches should be completed
    files = []
    for batch_id in args.batch_ids:
        batch = model.batches.retrieve(batch_id)
        print("Batch {}: {}".format(batch_id, batch.status))
        if batch.status != "completed":
            print("Not all batches are completed. Exiting...")
            return
        else:
            files.append(batch.output_file_id)

    prediction_error = 0
    for file_id in files:
        resp = model.files.content(file_id)
        # parse jsonl in text
        for line in tqdm(resp.text.split("\n")):
            try:
                gpt_label = json.loads(line)
            except json.JSONDecodeError:  # skip empty lines
                continue
            choice0 = gpt_label["response"]["body"]["choices"][0]
            qid = gpt_label["custom_id"]
            try:
                pred = int(choice0["message"]["content"])
            except ValueError:
                pred = 0  # default to non-interrogative
                prediction_error += 1
            logprob = float(choice0["logprobs"]["content"][0]["logprob"])

            train_df.loc[train_df["tr_qid"] == int(qid), "pred"] = pred
            train_df.loc[train_df["tr_qid"] == int(qid), "logprobs"] = logprob
    print("{} pred. errors found that are neighther 0 or 1: ", prediction_error)
    code.interact(local=dict(globals(), **locals()))

    # save the annotated dataset
    if args.save_preds:
        train_df.to_csv("data/IntVsDecl/q_cls_gpt_annotated.csv", index=False)

    # calculate the accuracy, precision and recall with gpt predictions
    acc = (train_df["pred"] == train_df["target"]).mean()
    prec = (train_df["pred"] & train_df["target"]).sum() / train_df["pred"].sum()
    recall = (train_df["pred"] & train_df["target"]).sum() / train_df["target"].sum()

    print(f"{args.model_name} agreement with IntVsDecl dataset")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {recall:.2f}")

    # print some examples with high logprobs
    print("Examples with high logprobs")
    for _, row in train_df[train_df["logprobs"] > 0.00005].iterrows():
        print(row["doc"], row["logprobs"])


if __name__ == "__main__":
    # Parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument(
        "-b",
        "--batch_ids",
        nargs="*",
        help="Unique batch id(s) returned when a request is made. Use this"
        + " to retrieve the results of a batch requests.",
    )
    parser.add_argument("--fewshot", action="store_true")
    parser.add_argument("--test_file", type=str, default="data/IntVsDecl/test.csv")
    parser.add_argument("--train_file", type=str, default="data/IntVsDecl/train.csv")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--save_preds", action="store_true")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args()

    # Load the OpenAI API key
    load_dotenv()
    model = OpenAI(api_key=os.environ.get("GPT_API_KEY"))

    if args.batch_ids is not None:
        retrieve_eval_save(args)
    else:
        batch_request(args)
