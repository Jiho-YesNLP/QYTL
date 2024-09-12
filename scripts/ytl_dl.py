"""
A shared google spreadsheet is used to store the youtube education playlist links. This script reads videos of the playlists and download transcripts and comments of the videos.
"""

import code
import os
from dotenv import load_dotenv
import argparse
from tqdm import tqdm
import json
from itertools import islice

import gspread
import googleapiclient
import googleapiclient.discovery

from youtube_transcript_api import YouTubeTranscriptApi as YTTA
from youtube_transcript_api.formatters import JSONFormatter
import youtube_comment_downloader as ytc


def read_playlists(
    client: gspread.Client,
    sh_id: str,
    sheet_name: str = "Sheet1",
):
    sh = client.open_by_key(sh_id)  # open the spreadsheet
    # sh = client.open_by_url(
    #     "https://docs.google.com/spreadsheets/d/1QAVrbPnitRay5iqdkrJ8Mk1jM1OUktPitVkXRi5tERk/edit#gid=0"
    # )
    worksheet = sh.worksheet(sheet_name)  # open the worksheet
    # columns = ['playlistname', 'number-of_vds', 'link', 'listID']
    playlists = worksheet.get_all_values()[1:]  # get all values except the header

    return playlists


def download_trans_comm(args, gsc, gac, ytc):
    playlists = read_playlists(gsc, args.sheet_id, args.sheet_name)
    vid_metadata = []
    for pl in tqdm(playlists):
        # playlistitems doc  https://tinyurl.com/z5ebnpb9
        nextpage = "firstpage"
        while nextpage:
            req = gac.playlistItems().list(
                part="snippet",
                playlistId=pl[3],
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
                }
                vid_metadata.append(entry)
    print(f"{len(vid_metadata)} videos found in the {len(playlists)} playlists")

    # create a dictionary to store the transcripts and comments; .data/transcripts/ and .data/comments/
    os.makedirs(f"{args.output_dir}/transcripts", exist_ok=True)
    os.makedirs(f"{args.output_dir}/comments", exist_ok=True)

    # JSON formatter
    formatter = JSONFormatter()
    # download transcripts and comments
    for vid in vid_metadata:
        # if file exists, skip
        skip = False
        if os.path.exists(f"{args.output_dir}/transcripts/{vid['vid']}.json"):
            print(f"Skipping {vid['vid']}")
            skip = True
        if not skip:
            # download transcript
            print(f"Downloading transcript for {vid['vid']}")
            try:
                transcript = YTTA.get_transcript(vid["vid"])
            except:
                print(f"Transcripts disabled for {vid['vid']}")
            else:  # if no exception
                json_formatted = formatter.format_transcript(transcript)
                # save transcript
                with open(
                    f"{args.output_dir}/transcripts/{vid['vid']}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(json_formatted)

        # download comments
        skip = False
        if os.path.exists(f"{args.output_dir}/comments/{vid['vid']}.json"):
            print(f"Skipping comments for {vid['vid']}")
            skip = True
        if not skip:
            print(f"Downloading comments for {vid['vid']}")
            comments = []
            for c in islice(ytc.get_comments(vid["vid"]), 1000):
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
            with open(
                f"{args.output_dir}/comments/{vid['vid']}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(comments, f)


if __name__ == "__main__":
    # args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--sheet_id", type=str, required=True, help="Google spreadsheet id"
    )
    arg_parser.add_argument(
        "--sheet_name", type=str, default="Sheet1", help="Google spreadsheet name"
    )
    arg_parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )
    args = arg_parser.parse_args()

    # load env variables
    load_dotenv()

    # gspread client
    gs_client = gspread.service_account()
    ga_client = googleapiclient.discovery.build(
        "youtube", "v3", developerKey=os.environ["YT_API_KEY"]
    )
    ytc_client = ytc.YoutubeCommentDownloader()

    download_trans_comm(args, gs_client, ga_client, ytc_client)
