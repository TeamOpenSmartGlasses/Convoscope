import os
import glob
import webvtt
import sys
from datetime import datetime
import random
from test_helper import *

def load_lex_transcripts_for_record(transcript_folder = "./lex_whisper_transcripts", chunk_time_seconds=120):
    dt_format = "%H:%M:%S.%f"

    file = sys.argv[1]
    print("File: {}".format(str(file)))

    podcast_transcript_chunks = list()
    convo_name = os.path.splitext(os.path.basename(file))[0]
    print("Processing {}...".format(convo_name))
    for caption in webvtt.read(file):
        time_start_time = datetime(1900, 1, 1)
        caption_start_time = datetime.strptime(caption.start, dt_format)
        seconds_since_start = (caption_start_time - time_start_time).total_seconds()
        cap_obj = {'timestamp': int(seconds_since_start), 'text': caption.text.strip()}
        # print(str(cap_obj))
        podcast_transcript_chunks.append(cap_obj)
    return {'name': convo_name, 'chunks': podcast_transcript_chunks}

def test_and_record_using_lex_transcript():
    # start_recording()
    start_time = time.time()

    print("=== Begin Transcript Uploading ===")
    lex = load_lex_transcripts_for_record()
    name = lex['name']
    caption_objs = lex['chunks']

    while len(caption_objs) > 0:
        if time.time() - start_time > caption_objs[0]['timestamp']:
            obj = caption_objs.pop(0)
            chat(obj['text'], True)
            print(f"Uploaded transcript: {str(obj)}")

    print("=== Transcript Uploading Complete ===")
    # ui_poll_loop_with_timeout()
    print("Waiting 15 seconds, then stopping recording")
    time.sleep(15)
    # recording = save_recording(name)
    
    # # Save recording to separate recording folder in tests/ dir
    # file_path = f'recordings/{name}.json'
    # with open(file_path, 'w') as file:
    #     json.dump(recording, file)
    # print(f"Lex recording saved to: {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: `python record_on_lex.py filename`")
        print("EX:    `python record_on_lex.py lex_whisper_transcripts/episode_001_large.vtt`")
        sys.exit(1)

    if ".vtt" not in sys.argv[1]:
        print("File must be a .vtt transcript")
        sys.exit(1)

    test_and_record_using_lex_transcript()