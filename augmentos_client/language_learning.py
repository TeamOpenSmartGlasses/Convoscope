# AugmentOS
import asyncio
import json
from augmentos_client import TPAClient

# Custom
import traceback
import asyncio
import time
from llsg.language_learning_agent import run_language_learning_agent
from llsg.Modules.DatabaseHandler import DatabaseHandler
from llsg.helpers.word_frequency_percentiles import get_word_frequency_percentiles

db_handler = DatabaseHandler(parent_handler=True)

class ExampleTPA:
    def __init__(self):
        self.client = TPAClient(
            app_id="com.teamopensmartglasses.languagelearning",
            app_name="Language Learning",
            app_description="We gettin educated",
            server_url="http://localhost:8080",
            subscriptions=["*"]
        )
        # Register callbacks
        self.client.on_transcript_received(self.on_transcript)
        self.client.on_location_received(self.on_location)
        self.client.on_camera_received(self.on_camera)
        self.client.on_other_received(self.on_other)

    async def on_transcript(self, data):
        # print(f"[ExampleTPA on_transcript] Data received in callback: {data}")
        user_id = data['user_id']
        user = db_handler.get_user(user_id) # Make sure user created in db
        transcript = data['data']

        words_to_show = None

        ctime = time.time()
        
        if db_handler.get_user_settings_value(transcript['user_id'], "is_having_language_learning_contextual_convo"):
            return

        #get users target language
        target_language = db_handler.get_user_settings_value(transcript['user_id'], "target_language")
        source_language = db_handler.get_user_settings_value(transcript['user_id'], "source_language")

        transcribe_language = transcript["transcribe_language"]
        #print("GOT TRANSCIBE LANGUAGE: " + transcribe_language)

        #get word frequencies to represent how common words are
        word_frequency_percentiles = get_word_frequency_percentiles(transcript['text'], transcribe_language)

        #get previous word definitons for last few minutes
        live_translate_word_history = db_handler.get_recent_language_learning_words_defined_history_for_user(transcript['user_id'], n_seconds=30)

        #run the language learning agent
        words_to_show = await run_language_learning_agent(transcript['text'], word_frequency_percentiles, target_language, transcribe_language, source_language, live_translate_word_history)
        #print("transcript is: ")
        #print(transcript)

        loop_time = time.time() - ctime
        #print(f"RAN LL IN : {loop_time}")
        #print(words_to_show)

        if words_to_show:
            final_words_to_show = list(filter(None, words_to_show))
            print("WORDS TO SHOW")
            print(json.dumps(final_words_to_show, indent=4))
            
            ### TODO: Pull out the recent word logic from edge to here
            displayList = [f'{item["in_word"]} -> {item["in_word_translation"]}' for item in final_words_to_show]
            await self.client.send_rows_card(user_id, displayList)

    async def on_location(self, data):
        pass

    async def on_camera(self, data):
        pass

    async def on_other(self, data):
        pass

    def start(self):
        # Start the FastAPI server in a separate thread
        self.client.start()

async def main():
    example_tpa = ExampleTPA()
    example_tpa.start()
    # Run indefinitely to keep the FastAPI app running
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
