"""
This script reads sentences from the following sources, calculate important
verbs in each BT category using point-wise mutual information (PMI) and show
different uses of the verbs in the BT categories and sources.

if you have issue with nltk and ubuntu, use python 3.8.13
"""

import code
import argparse
import math
import os

import pandas as pd
from tqdm import tqdm
import nltk

stopwords = set(nltk.corpus.stopwords.words("english"))

bt_levels = [
    "Knowledge",
    "Comprehension",
    "Application",
    "Analysis",
    "Synthesis",
    "Evaluation",
]


def extract_verbs(pairs):
    """
    Extract verbs from the sentences in the pairs
    """
    lemmer = nltk.WordNetLemmatizer()
    verbs = {}  # {verb: [cnt1, cnt2, ...]}
    for pair in tqdm(pairs):
        sentence = pair[0]

        # tokenize the sentences and remove stop words
        tokens = nltk.word_tokenize(sentence)
        # POS tagging: verb only
        tagged = nltk.pos_tag(tokens)
        for word, pos in tagged:
            if pos.startswith("VB"):
                # lemmatize the verb
                word = lemmer.lemmatize(word, "v").lower()
                if word in verbs:
                    verbs[word][bt_levels.index(pair[1])] += 1
                else:
                    verbs[word] = [0] * len(bt_levels)
                    verbs[word][bt_levels.index(pair[1])] = 1
        # remove short words (<= 3 in length) and non-alphabetic words
        verbs = {
            verb: cnts
            for verb, cnts in verbs.items()
            if len(verb) > 3 and verb.isalpha()
        }
    return verbs


def calculate_pmi(verbs):
    """
    Calculate PMI for the verbs
    """
    total = sum(sum(cnts) for cnts in verbs.values())
    subtotals = [sum(cnts) for cnts in zip(*verbs.values())]
    p_bt = [cnt / total for cnt in subtotals]
    for verb, cnts in verbs.items():
        p_verb = sum(cnts) / total
        pmi = [0.0] * len(bt_levels)
        for i in range(len(bt_levels)):
            if cnts[i] > 0:
                try:
                    pmi[i] = math.log(cnts[i] / total / (p_verb * p_bt[i]))
                except ValueError:
                    code.interact(local=dict(globals(), **locals()))
        verbs[verb] = pmi
    return verbs, subtotals


def print_top_verbs(verbs_pmi, subtotals, args):
    for i in range(len(bt_levels)):
        sorted_verbs = sorted(verbs_pmi.items(), key=lambda x: x[1][i], reverse=True)
        print(f"BT level - {bt_levels[i]}, Support: {subtotals[i]}")
        for verb, pmi in sorted_verbs[:10]:
            print(f"{verb}: {pmi[i]:.2f}")
        print()


def run(args):
    # Source 1: Question-BT level dataset used for training BT classifier
    # ---------------------------------------------------------------
    # Read the sentences from the source
    df = pd.read_csv(args.source_filepath, lineterminator="\n")
    # drop short sentences
    if args.drop_short:
        df = df[df["text"].apply(lambda x: len(x.split()) >= 5)]

    if args.with_probs:
        # set probability threshold, if prob less than the threshold assign UNK
        pairs = [
            (row["text"], row["BT LEVEL"])
            if row["PROB"] > args.prob_threshold
            else (row["text"], "UNK")
            for _, row in df.iterrows()
        ]
    else:
        try:
            pairs = [(row["QUESTION"], row["BT LEVEL"]) for _, row in df.iterrows()]
        except KeyError as e:
            code.interact(local=dict(globals(), **locals()))
            err = (
                "Check if the source file has prob column. If so, use --with_probs flag"
            )
            raise KeyError(f"KeyError: {e}. {err}")

    # extract verbs
    verbs = extract_verbs(pairs)
    # # sort and print the top 10 verbs per BT level using raw counts
    # for i in range(6):
    #     sorted_verbs = sorted(verbs.items(), key=lambda x: x[1][i], reverse=True)
    #     print(bt_levels[i])
    #     for verb, cnts in sorted_verbs[:10]:
    #         print(f"{verb}: {cnts[i]}")
    #     print()

    # delete verbs that appear less than 3 time
    verbs = {verb: cnts for verb, cnts in verbs.items() if sum(cnts) >= 3}

    # calculate PMI and print the top 10 verbs per BT level
    verbs_pmi, subtotals = calculate_pmi(verbs.copy())
    # show the top 10 verbs per BT level
    print_top_verbs(verbs_pmi, subtotals, args)
    code.interact(local=dict(globals(), **locals()))


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    # data parameters
    parser.add_argument("--source_filepath", type=str, required=True,
                        help="Path to the source file")
    parser.add_argument("--with_probs", action="store_true",
                        help="Use the probabilities in the source file")
    parser.add_argument("--prob_threshold", type=float, default=0.97)
    parser.add_argument("--drop_short", action="store_true",
                        help="Drop sentences with less than 5 words")
    args = parser.parse_args()
    # fmt: on

    if args.with_probs:
        bt_levels.append("UNK")
    args.num_classes = len(bt_levels)

    run(args)


if __name__ == "__main__":
    # file_source1 = "data/q_bt/combined.csv"
    # file_source2 = "data/q_bt/train_aug.csv"
    # file_source3 = "data/output/q_bt_pred.csv"
    main()
