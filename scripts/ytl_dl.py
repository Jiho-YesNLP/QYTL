"""
A shared google spreadsheet is used to store the youtube education playlist links. This script reads videos of the playlists and download transcripts and comments of the videos.
"""

import code
import os
from dotenv import load_dotenv
import argparse
import json
from itertools import islice
from tqdm import tqdm
import time


# import gspread
import googleapiclient
import googleapiclient.discovery

# https://github.com/jdepoix/youtube-transcript-api
from youtube_transcript_api import YouTubeTranscriptApi as YTTA
from youtube_transcript_api.formatters import JSONFormatter
import youtube_comment_downloader as ytc

SUBJ = {
    "Biology": "bio",
    "Chemistry": "chem",
    "Mathematics": "math",
    "Physics": "phy",
    "Computer Science": "cs",
}
VID_STATS = {k: [] for k in SUBJ.keys()}


def download_trans_comm(args, gac, ytc):
    # Read the playlist file in JSON format
    with open(args.pl_file, "r") as f:
        playlists = json.load(f)

    # read list of videos per playlist using the youtube api
    vidlist = []
    for pl in tqdm(playlists):
        # playlistitems doc  https://tinyurl.com/z5ebnpb9
        nextpage = "firstpage"
        while nextpage:
            req = gac.playlistItems().list(
                part="snippet",
                playlistId=pl["listID"],
                maxResults=50,
                pageToken=nextpage if nextpage != "firstpage" else None,
            )
            resp = req.execute()

            nextpage = resp.get("nextPageToken")
            for item in resp["items"]:
                entry = {
                    "pid": item["snippet"]["playlistId"],
                    "vid": item["snippet"]["resourceId"]["videoId"],
                    "title": item["snippet"]["title"],
                    "subject": pl["subject"],
                }
                vidlist.append(entry)
    print(f"{len(vidlist)} videos found in the {len(playlists)} playlists")

    # create a dictionary to store the transcripts and comments;
    #    .data/transcripts/ and .data/comments/
    os.makedirs(f"{args.output_dir}/transcripts", exist_ok=True)
    os.makedirs(f"{args.output_dir}/comments", exist_ok=True)

    # YTTA JSON formatter
    formatter = JSONFormatter()
    # get stats, download transcripts and comments
    for vid in vidlist:
        # get stats
        stats = gac.videos().list(part="statistics", id=vid["vid"]).execute()
        try:
            VID_STATS[vid["subject"]].append(
                [
                    stats["items"][0]["statistics"][k]
                    for k in ["viewCount", "likeCount", "commentCount"]
                ]
            )
        except:
            pass

        # if file exists, skip
        tf = "{}/transcripts/{}-{}-trans.json" "".format(
            args.output_dir, vid["vid"], SUBJ[vid["subject"]]
        )
        cf = "{}/comments/{}-{}-cmmt.json" "".format(
            args.output_dir, vid["vid"], SUBJ[vid["subject"]]
        )

        if os.path.exists(tf):
            print(f"Skipping {vid['vid']}")
        else:
            # download transcript
            print(f"Downloading transcript for {vid['vid']}")
            try:
                transcript = YTTA.get_transcript(vid["vid"])
            except:
                print(f"Failed to download transcript for {vid['vid']}")
            else:  # if no exception
                # save transcript
                with open(tf, "w", encoding="utf-8") as f:
                    f.write(formatter.format_transcript(transcript))

        # download comments
        if os.path.exists(cf):
            print(f"Skipping comments for {vid['vid']}")
        else:
            print(f"Downloading comments for {vid['vid']}")
            comments = []
            try:
                resp = ytc.get_comments(vid["vid"])
            except:
                print(f"Failed to download comments for {vid['vid']}")
            else:
                for c in islice(resp, 1000):
                    entry = {
                        "vid": vid["vid"],
                        "cid": c["cid"],
                        "comment": c["text"].replace("\n", " "),
                        "votes": c["votes"],
                        "replies": c["replies"],
                        "reply": c["reply"],
                    }
                    comments.append(entry)
                # save comments
                with open(cf, "w", encoding="utf-8") as f:
                    json.dump(comments, f)

    # save stats with unix timestamp
    statfile = "{}/stats-{}.json".format(args.output_dir, int(time.time()))
    with open(statfile, "w") as f:
        json.dump(VID_STATS, f)


if __name__ == "__main__":
    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--pl_file", type=str, required=True, help="Playlist file")
    arg_parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )
    args = arg_parser.parse_args()

    # load env variables
    load_dotenv()

    # gs_client = gspread.service_account()
    # YouTube API client
    ga_client = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=os.environ["YT_API_KEY"]
    )
    ytc_client = ytc.YoutubeCommentDownloader()
    download_trans_comm(args, ga_client, ytc_client)
