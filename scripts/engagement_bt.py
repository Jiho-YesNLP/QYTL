"""
Examine if there's a relationship between engagement and BT.
# or repies and likes represent engagement. Compute correlation between BT and engagement.
"""

import code
import os
from dotenv import load_dotenv
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
import pandas as pd
import googleapiclient
import googleapiclient.discovery

# file_source3 = "data/output/q_bt_pred.csv"

if os.path.exists("data/output/bt_engagement.csv"):
    df = pd.read_csv("data/output/bt_engagement.csv", lineterminator="\n")
else:
    # Read the data
    df = pd.read_csv("data/output/q_bt_pred.csv", lineterminator="\n")

    df.fillna(0, inplace=True)
    df["votes"] = (
        df["votes"].replace({"K": "e+03"}, regex=True).astype(float).astype(int)
    )

    # viewcounts per video is missing

    # Note: there is a quota limit of 10,000 units per day
    # YouTube API client
    load_dotenv()
    gac = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=os.environ["YT_API_KEY"]
    )
    # get viewcounts using google api
    vid_stats = {}
    print("Retrieving video stats...")
    for vid in tqdm(df["vid"].unique()):
        stats = gac.videos().list(part="statistics", id=vid).execute()
        vid_stats[vid] = {
            k: stats["items"][0]["statistics"][k]
            for k in ["viewCount", "likeCount", "commentCount"]
        }

    # Normalize votes and replies by viewcounts
    df["commentCount"] = df["vid"].map(lambda x: int(vid_stats[x]["commentCount"]))
    # log max
    df["votes_"] = df["votes"] / df["commentCount"]
    df["replies_"] = df["replies"] / df["commentCount"]

    # save
    df.to_csv("data/output/bt_engagement.csv", index=False)


# plot bt_level vs replies
BT_LEVELS = [
    "IRREL",
    "Knowledge",
    "Comprehension",
    "Application",
    "Analysis",
    "Evaluation",
    "Synthesis",
]

# assign IRREL class of the ones with PROB < 0.97
df["BT LEVEL"] = df.apply(
    lambda x: x["BT LEVEL"] if x["PROB"] >= 0.99 else "IRREL", axis=1
)
# replace categorical values with numerical values for correlation
df["BT LEVEL"] = df["BT LEVEL"].replace(
    {
        "Knowledge": 1,
        "Comprehension": 2,
        "Application": 3,
        "Analysis": 4,
        "Evaluation": 5,
        "Synthesis": 6,
        "IRREL": 0,
    }
)

# print correlation
# print(df[["BT LEVEL", "votes_", "replies_"]].corr())
code.interact(local={**locals(), **globals()})
# revert to categorical values
df["BT LEVEL"] = df["BT LEVEL"].replace(
    {
        1: "Knowledge",
        2: "Comprehension",
        3: "Application",
        4: "Analysis",
        5: "Synthesis",
        6: "Evaluation",
        0: "IRREL",
    }
)

# Draw a boxenplot to show BT level vs replies
sns.boxenplot(
    df,
    x="BT LEVEL",
    y="replies_",
    color="b",
    order=BT_LEVELS,
    width_method="linear",
)
plt.savefig("data/output/bt_level_vs_replies.png")

# Draw a boxenplot to show BT level vs votes
sns.boxenplot(
    df,
    x="BT LEVEL",
    y="votes_",
    color="b",
    order=BT_LEVELS,
    width_method="linear",
)
plt.savefig("data/output/bt_level_vs_votes.png")

code.interact(local={**locals(), **globals()})
