import time
import traceback
import asyncio

# custom
from DatabaseHandler import DatabaseHandler
from agents.adhd_stmb_agent import run_adhd_stmb_agent
from server_config import openai_api_key
from logger_config import logger
from constants import ADHD_STMB_AGENT

run_period = 20
transcript_back_time = run_period * 1.1 #The amount of time to look back for new transcripts
total_transcript_context_time = 3.5 * 60 #The total amount of time to consider for the context of the transcript to summarize
minimum_topic_time = 20 #the minimum amount of time since last topic change for a new topic chnage
max_same_topic_time = 0.75 * total_transcript_context_time #Max amount of time for recent transcript if topic hasn't changed
maximum_topic_time = 120

def adhd_stmb_agent_processing_loop():
    print("START ADHD STMB AGENT PROCESSING LOOP")
    dbHandler = DatabaseHandler(parent_handler=False)
    loop = asyncio.get_event_loop()

    while True:
        if not dbHandler.ready:
            print("dbHandler not ready")
            time.sleep(0.1)
            continue

        # wait for some transcripts to load in
        time.sleep(5)

        try:
            pLoopStartTime = time.time()
            # Check for new transcripts
            # print("RUNNING ADHD STMB LOOP")
            newTranscripts = dbHandler.get_recent_transcripts_from_last_nseconds_for_all_users(
                n=transcript_back_time)

            words_to_show = None
            for transcript in newTranscripts:
                if not dbHandler.get_user_feature_enabled(transcript['user_id'], ADHD_STMB_AGENT): 
                    continue #skip if the user is not in ADHD mode
                ctime = time.time()
                user_id = transcript['user_id']

                #get the timestamp of latest topic change of the user
                latest_topic_shifts = dbHandler.get_latest_topic_shift_for_user(user_id, true_shift=True)
                latest_topic_shift_time = None
                if latest_topic_shifts != []:
                    latest_topic_shift_time = latest_topic_shifts[0]["time_of_shift"]

                #get the last dynamic summary of the current topic of the user
                latest_dynamic_topic = dbHandler.get_latest_topic_shift_for_user(user_id, true_shift=False)
                previous_summary = ""
                if latest_dynamic_topic != []:
                    previous_summary = latest_dynamic_topic[0]["summary"]

                print("---------------------------PREVIOUS SUMMARY---------------------------")
                print(previous_summary)

                #get the whole context transcript for the user
                context_transcript, _, _ = dbHandler.get_transcripts_from_last_nseconds_for_user_as_string(user_id, n=total_transcript_context_time)

                #if there is no previous topic change, just feed in the last n minutes of transcript
                #if there was a topic change, use all the transcripts since then
                summarize_transcript_back_time = None
                if latest_topic_shift_time is None:
                    to_summarize_transcript = context_transcript
                    summarize_transcript_back_time = transcript_back_time
                else:
                    summarize_transcript_back_time = time.time() - latest_topic_shift_time
                    # summarize_transcript_back_time = min(max_same_topic_time, summarize_transcript_back_time)
                    to_summarize_transcript, _, _ = dbHandler.get_transcripts_from_last_nseconds_for_user_as_string(user_id, n=summarize_transcript_back_time)

                print("---------------------------Transcript to Summarize---------------------------")
                print(to_summarize_transcript)

                print("-------------------------------------")
                print(f"To summarize = context" if to_summarize_transcript == context_transcript else f"To summarize != context")
                #run the ADHD STMB agent
                summary, new_topic_shift_words = run_adhd_stmb_agent(to_summarize_transcript, context_transcript, previous_summary)
                print(f"-------------------------------------")
                print(f"Summary: {summary}")
                print(f"New Topic Shift Words: {new_topic_shift_words}")

                #add a new dynamic topic, not a true shift
                dbHandler.add_topic_shift_for_user(user_id, pLoopStartTime, summary, true_shift=False)

                #maybe add a new true_shift topic shift
                if summarize_transcript_back_time > minimum_topic_time: #don't allow topic shift if the last topic shift was too recent
                    if new_topic_shift_words != None:
                        #find the time stamp of the topic shift
                        #guess the timestamp of these words
                        idx = to_summarize_transcript.find(new_topic_shift_words)
                        print("idx: {}".format(idx))
                        shift_fraction = (idx + (idx + len(new_topic_shift_words))) / (2 * len(to_summarize_transcript)) #find roughly where in the transcript the topic shift was located
                        print("shift fraction: {}".format(shift_fraction))
                        transcript_start_time = pLoopStartTime - summarize_transcript_back_time
                        print("transciprt start time: {}".format(transcript_start_time))
                        topic_shift_in = summarize_transcript_back_time * shift_fraction
                        topic_shift_time = transcript_start_time + topic_shift_in + 3 # add a few seconds so no overlap in the shift phase
                        print("topic shift in: {}".format(topic_shift_in))
                        print("topic shift time: {}".format(topic_shift_time))
                        #add a new shift, save previous summary as this true shift
                        dbHandler.add_topic_shift_for_user(user_id, topic_shift_time, previous_summary, true_shift=True)

                    elif summarize_transcript_back_time > maximum_topic_time:
                        #add a new shift, save previous summary as this true shift
                        dbHandler.add_topic_shift_for_user(user_id, time.time(), previous_summary, true_shift=True)

                print("=========== ADHD AGENT SUMMARY: {}".format(summary))
                loop_time = time.time() - ctime
                print(f"RAN ADHD STMB AGENT IN : {loop_time}")

        except Exception as e:
            print("Exception in ADHD STMB loop...:")
            print(e)
            traceback.print_exc()

        finally:
            # lock.release()
            pLoopEndTime = time.time()
            #print("=== language leanring loop completed in {} seconds overall ===".format(
            #    round(pLoopEndTime - pLoopStartTime, 2)))

        #run again after delay for run_period
        time.sleep(max(0, run_period - (pLoopEndTime - pLoopStartTime)))