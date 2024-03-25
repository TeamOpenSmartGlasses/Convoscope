import time
import traceback
import asyncio

# custom
from DatabaseHandler import DatabaseHandler
from agents.speech_coach_agent import run_speech_coach_agent, run_egometer_agent, run_understandability_agent
from server_config import openai_api_key
from logger_config import logger
from constants import SPEECH_AGENT

run_period = 10 # seconds
num_periods = 6
calc_period = run_period * num_periods # to calculate the number of filler words every 2 minutes
transcript_back_time = run_period * 1.1
total_transcript_context_time = 5 * 60
minimum_topic_time = 20 #the minimum amount of time since last topic change for a new topic chnage

def speech_coach_agent_processing_loop():
    sliding_percentage_filler_words = 0
    total_percentage_filler_words = 0
    list_num_filler_words = [] 
    list_num_total_words = [] 
    print("START SPEECH COACH AGENT PROCESSING LOOP")
    dbHandler = DatabaseHandler(parent_handler=False)
    loop = asyncio.get_event_loop()

    while True:
        if not dbHandler.ready:
            print("dbHandler not ready")
            time.sleep(0.1)
            continue

        # wait for some transcripts to load in
        time.sleep(5)

        ctime = time.time()
        try:
            pLoopStartTime = time.time()
            # Check for new transcripts
            # print("RUNNING ADHD STMB LOOP")
            newTranscripts = dbHandler.get_recent_transcripts_from_last_nseconds_for_all_users(
                n=transcript_back_time, stringify = False)

            # words_to_show = None
            for raw_transcript in newTranscripts:
                # raw_transcript = {'user_id': user_id, 'transcripts': transcripts[]}
                # transcripts:[] =
                #
                # {
                # "device_id": "alexs phone",
                # "timestamp": 69420, 
                # "text": "i love poop", 
                # "speaker": 0,
                # "transcribe_language": "english"
                # }
                print("RAW TRANSCRIPT")
                print(raw_transcript)
                for t in raw_transcript['transcripts']:
                    if 'speaker' in t:
                        print(str(t['speaker']) +": " + t['text'])
                
                if not dbHandler.get_user_feature_enabled(raw_transcript['user_id'], SPEECH_AGENT): continue
                # ADD FOLLOWING LINE LATER                
                user_id = raw_transcript['user_id']

                # run filler word logic
                process_filler_words(raw_transcript, dbHandler, user_id, 
                                     sliding_percentage_filler_words, total_percentage_filler_words, list_num_filler_words, list_num_total_words)
                process_proportions(raw_transcript, dbHandler, user_id)

            newTranscripts = dbHandler.get_recent_transcripts_from_last_nseconds_for_all_users(n=30)
            for transcript in newTranscripts:
                #run egometer
                egometer_score = run_egometer_agent(transcript['text'])  
                print("EGOMETER SCORE: {}".format(egometer_score))

                #run understandability
                understandability_score = run_understandability_agent(transcript['text'])  
                print("UNDERSTANDABILITY SCORE: {}".format(understandability_score))
                dbHandler.add_understandability_score_for_user(transcript['user_id'], understandability_score)

            loop_time = time.time() - ctime
            print(f"RAN SPEECH COACH AGENT IN : {loop_time}")

        except Exception as e:
            print("Exception in SPEECH COACH loop...:")
            print(e)
            traceback.print_exc()

        finally:
            # lock.release()
            pLoopEndTime = time.time()
            #print("=== language leanring loop completed in {} seconds overall ===".format(
            #    round(pLoopEndTime - pLoopStartTime, 2)))

        #run again after delay for run_period
        time.sleep(max(0, run_period - (pLoopEndTime - pLoopStartTime)))

def process_proportions(raw_transcript, dbHandler, user_id):
        user_transcripts = [transcript['text'] for transcript in raw_transcript['transcripts'] if ('speaker' in transcript and transcript['speaker'] == 0)]
        merged_user_transcript = ('. ').join(user_transcripts) #merged_transcript
        num_user_words = len(merged_user_transcript.split(' ')) if merged_user_transcript != "" else 0

        others_transcripts = [transcript['text'] for transcript in raw_transcript['transcripts'] if ('speaker' in transcript and transcript['speaker'] != 0)]
        merged_others_transcript = ('. ').join(others_transcripts) #merged_transcript
        num_others_words = len(merged_others_transcript.split(' ')) if merged_others_transcript != "" else 0

        print("NUM USER WORDS: ", num_user_words)
        print("NUM OTHERS WORDS: ", num_others_words)
        proportion_user = 100 * num_user_words / (num_user_words + num_others_words) if num_user_words + num_others_words != 0 else 0
        proportion_user = round(proportion_user, 2)
        dbHandler.add_percent_proportion_user_for_user(user_id, proportion_user)
        print("PROPORTION USER: ", proportion_user)

def process_filler_words(raw_transcript, dbHandler, user_id, sliding_percentage_filler_words, total_percentage_filler_words, list_num_filler_words, list_num_total_words):
    print("process filler words")
    num_filler_words = 0
    num_total_words = 0
    try:
        user_transcripts = [transcript['text'] for transcript in raw_transcript['transcripts'] if ('speaker' in transcript and transcript['speaker'] == 0)]
        transcript = ('. ').join(user_transcripts) #merged_transcript
        # number_of_filler_words = int(number_of_filler_words)
        curr_num_total_words = len(transcript.split(' ')) if transcript != "" else 0
        curr_num_filler_words = run_speech_coach_agent(transcript)  

        print("-"*20)
        print(transcript)
        print("".split(' '))
        print("CURR NUM OF FILLER WORDS: ", curr_num_filler_words)
        print("CURR NUM OF TOTAL WORDS: ", curr_num_total_words)
        if curr_num_total_words == 0:
            print("CURR PERCENTAGE OF FILLER WORDS: ", curr_num_filler_words/curr_num_total_words)

        list_num_filler_words.append(curr_num_filler_words)
        list_num_total_words.append(curr_num_total_words)
        print(list_num_filler_words)
        print(list_num_total_words)
        if len(list_num_filler_words) > num_periods:
            del list_num_filler_words[0]
            del list_num_total_words[0]
        sliding_percentage_filler_words = sum(list_num_filler_words) / sum(list_num_total_words) * 100 if sum(list_num_total_words) != 0 else 0
        sliding_percentage_filler_words = round(sliding_percentage_filler_words, 1)
        num_filler_words += curr_num_filler_words
        num_total_words += curr_num_total_words
        total_percentage_filler_words = num_filler_words / num_total_words * 100 if num_total_words != 0 else 0

        # print("NUMBER OF FILLER WORDS SO FAR: ", num_filler_words)
        print("SLIDING PERCENTAGE OF FILLER WORDS: ", sliding_percentage_filler_words)
        print("-"*20)
        dbHandler.add_percent_filler_words_for_user(user_id, sliding_percentage_filler_words)
    except Exception as e:
        print("Failed to convert number of filler words to int")
        print(e)
        print(num_filler_words)
