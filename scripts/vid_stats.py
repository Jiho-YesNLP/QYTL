import json
import sys
import glob
import code

from tqdm import tqdm
from prettytable import PrettyTable


def count_stats():
    # subject keys: Biology, Chemistry, Mathematics, Physics, Computer Science
    subjects = ["Biology", "Chemistry", "Mathematics", "Physics", "Computer Science"]

    # read data
    # entries are in the form of: [viewcounts, likecounts, commentcounts]
    with open(sys.argv[1], "r") as file:
        data = json.load(file)

    # print the stats table with the following columns:
    # - subject, video_count, # comments (avg/min/max), avg views, avg likes
    pt = PrettyTable()
    pt.field_names = ["Subject", "Video Count", "Comments", "Views", "Likes"]

    for s in subjects:
        video_count = len(data[s])
        views = [int(video[0]) for video in data[s]]
        likes = [int(video[1]) for video in data[s]]
        comments = [int(video[2]) for video in data[s]]

        avg_views = sum(views) / video_count
        avg_likes = sum(likes) / video_count
        avg_comments = sum(comments) / video_count
        min_comments = min(comments)
        max_comments = max(comments)

        pt.add_row(
            [
                s,
                video_count,
                f"{avg_comments:.2f} ({min_comments}-{max_comments})",
                f"{avg_views:.2f}",
                f"{avg_likes:.2f}",
            ]
        )
    print(pt)


def length_stats():
    # read transcript/comments json files and calculate the average lengths
    # of the comments/transcripts for each subject

    # subjects are indicated by the filenames: {}-{subject}-{type}.json
    trans_lengths = {k: [] for k in ["bio", "chem", "math", "phy", "cs"]}
    cmt_lengths = {k: [] for k in ["bio", "chem", "math", "phy", "cs"]}

    # transcripts
    # read json files in data/transcripts
    files = glob.glob("data/transcripts/*.json")
    print("Reading transcript files...")
    for file in tqdm(files):
        tlen = 0
        with open(file, "r") as f:
            data = json.load(f)
            subj = file.split("-")[-2]
            for entry in data:
                tlen += len(entry["text"].split())
            trans_lengths[subj].append(tlen)

    # comments
    # read json files in data/comments
    files = glob.glob("data/comments/*.json")
    print("Reading comment files...")
    for file in tqdm(files):
        with open(file, "r") as f:
            data = json.load(f)
            subj = file.split("-")[-2]
            for entry in data:
                clen = len(entry["comment"].split())
                cmt_lengths[subj].append(clen)

    # print the average lengths
    pt = PrettyTable()
    pt.field_names = ["Subject", "Transcripts", "Comments"]
    for s in ["bio", "chem", "math", "phy", "cs"]:
        pt.add_row(
            [
                s,
                sum(trans_lengths[s]) / len(trans_lengths[s]),
                sum(cmt_lengths[s]) / len(cmt_lengths[s]),
            ]
        )
    print(pt)


if __name__ == "__main__":
    count_stats()
    length_stats()
