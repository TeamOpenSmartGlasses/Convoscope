import time
import traceback
import asyncio

# custom
from DatabaseHandler import DatabaseHandler
from agents.language_learning_agent import run_language_learning_agent
from agents.helpers.word_frequency_percentiles import get_word_frequency_percentiles

from constants import LANGUAGE_LEARNING_AGENT

db_handler = DatabaseHandler(parent_handler=False)

time_between_iterations = 0.5
timelength_of_usable_transcripts = time_between_iterations * 4

def start_language_learning_agent_processing_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(language_learning_agent_processing_loop())
    loop.close()

async def language_learning_agent_processing_loop():
    print("START LANGUAGE LEARNING PROCESSING LOOP")

    while True:
        if not db_handler.ready:
            print("dbHandler not ready")
            await time.sleep(0.1)
            continue
        
        #wait for some transcripts to load in
        await asyncio.sleep(time_between_iterations)

        try:
            newTranscripts = db_handler.get_recent_transcripts_from_last_nseconds_for_all_users(n=timelength_of_usable_transcripts)
            tasks = [process_transcript(transcript) for transcript in newTranscripts]
            await asyncio.gather(*tasks)
                
        except Exception as e:
            print("Exception in ll generator...:")
            print(e)
            traceback.print_exc()

async def process_transcript(transcript: str):
    words_to_show = None
    if not db_handler.get_user_feature_enabled(transcript['user_id'], LANGUAGE_LEARNING_AGENT): return

    ctime = time.time()
    
    if db_handler.get_user_settings_value(transcript['user_id'], "is_having_language_learning_contextual_convo"):
        # print("User is having a conversation, skipping language translation")
        return
    else:
        pass
        #print("User is not having a conversation, running language translation")

    #get users target language
    target_language = db_handler.get_user_settings_value(transcript['user_id'], "target_language")
    #print("GOT TARGET LANGUAGE: " + target_language)
    source_language = db_handler.get_user_settings_value(transcript['user_id'], "source_language")
    #get the transcription language
    #print(transcript)
    transcribe_language = transcript["transcribe_language"]
    #print("GOT TRANSCIBE LANGUAGE: " + transcribe_language)

    #get word frequencies to represent how common words are
    word_frequency_percentiles = get_word_frequency_percentiles(transcript['text'], transcribe_language)

    #get previous word definitons for last few minutes
    live_translate_word_history = db_handler.get_recent_nminutes_language_learning_words_defined_history_for_user(transcript['user_id'], n_minutes=4)
    #print("GOT live translate word history: ")
    #print(live_translate_word_history)

    #run the language learning agent
    words_to_show = await run_language_learning_agent(transcript['text'], word_frequency_percentiles, target_language, transcribe_language, source_language, live_translate_word_history)
    #print("transcript is: ")
    #print(transcript)

    loop_time = time.time() - ctime
    #print(f"RAN LL IN : {loop_time}")
    #print(words_to_show)

    if words_to_show:
        final_words_to_show = list(filter(None, words_to_show))
        #print("WORDS TO SHOW")
        #print(final_words_to_show)
        db_handler.add_language_learning_words_to_show_for_user(transcript['user_id'], final_words_to_show)



        #run again after delay for run_period
        # time.sleep(max(0, time_between_iterations - (pLoopEndTime - pLoopStartTime)))
