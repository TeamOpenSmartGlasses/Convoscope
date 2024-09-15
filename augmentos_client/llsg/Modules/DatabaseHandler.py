from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import time
import math
from hashlib import sha256
from llsg.Modules.server_config import database_uri, clear_users_on_start, clear_cache_on_start
import uuid
from llsg.Modules.constants import TESTING_LL_CONTEXT_CONVO_AGENT, MODES_FEATURES_MAP
from llsg.helpers.time_function_decorator import time_function


class DatabaseHandler:
    def __init__(self, parent_handler=True):
        print("INITTING DB HANDLER")
        self.uri = database_uri
        self.min_transcript_word_length = 5
        self.wake_word_min_update_time = 2  # 2 seconds
        self.user_collection = None
        self.cache_collection = None
        self.ready = False
        self.backslide = 4
        self.intermediate_transcript_validity_time = 0  # .3 # 300 ms in seconds
        self.final_transcript_validity_time = 0  # .3 # 300 ms in seconds
        self.transcript_expiration_time = 600 * 6  # 60 minutes in seconds
        self.parent_handler = parent_handler
        self.empty_transcript = {
                "text": "", "device_id": "None", "timestamp": -1, "is_final": False, "uuid": -1, "transcribe_language" : "English"
                }

        # Create a new client and connect to the server
        self.client = MongoClient(self.uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")

            self.init_users_collection()
            self.init_transcripts_collection()
            self.init_cache_collection()
            self.init_insights_collections()
            self.init_ratings_collection()
            self.init_language_learning_collection()
            self.init_gps_location_collection()
            self.init_topic_shifts_collection()
            self.ready = True
        except Exception as e:
            print(e)

    ### INIT ###
    def set_is_having_language_learning_contextual_convo_flag_for_all_users(self, flag_value=False):
        self.user_collection.update_many({}, {"$set": {"settings.is_having_language_learning_contextual_convo": flag_value}})

    def set_vocabulary_upgrade_enabled_flag_for_all_users(self, flag_value=False):
         self.user_collection.update_many({}, {"$set": {"settings.vocabulary_upgrade_enabled": flag_value}})

    def set_command_start_language_learning_contextual_convo_flag_for_all_users(self, flag_value=False):
        self.user_collection.update_many({}, {"$set": {"settings.command_start_language_learning_contextual_convo": flag_value}})

    def init_users_collection(self):
        self.user_db = self.client['users']
        self.user_collection = self.get_collection(
            self.user_db, 'users', wipe=clear_users_on_start)

        self.active_user_db = self.client['active_users']
        self.active_user_collection = self.get_collection(
        self.active_user_db, 'active_users', wipe=clear_users_on_start)
        self.set_is_having_language_learning_contextual_convo_flag_for_all_users()
        self.set_vocabulary_upgrade_enabled_flag_for_all_users()
        self.set_command_start_language_learning_contextual_convo_flag_for_all_users()

    def init_transcripts_collection(self):
        self.transcripts_db = self.client['transcripts']
        self.transcripts_collection = self.get_collection(
            self.transcripts_db, 'transcripts', wipe=clear_users_on_start)

    def init_cache_collection(self):
        self.cache_db = self.client['cache']
        self.cache_collection = self.get_collection(
            self.cache_db, 'cache', wipe=clear_cache_on_start)

    def init_insights_collections(self):
        self.results_db = self.client['results']
        self.display_requests_collection = self.get_collection(
            self.results_db, 'display_requests', wipe=clear_cache_on_start)
        self.agent_explicit_queries_collection = self.get_collection(
            self.results_db, 'agent_explicit_queries', wipe=clear_cache_on_start)
        self.agent_explicit_insights_results_collection = self.get_collection(
            self.results_db, 'agent_explicit_insights_results', wipe=clear_cache_on_start)

        self.agent_insights_results_collection = self.get_collection(
            self.results_db, 'agent_insights_results', wipe=clear_cache_on_start)
        self.agent_insights_queries_collection = self.get_collection(
            self.results_db, 'agent_insights_queries', wipe=clear_cache_on_start)
        
        self.agent_proactive_definer_collection = self.get_collection(
            self.results_db, 'agent_proactive_definer_results', wipe=clear_cache_on_start)

    def init_ratings_collection(self):
        self.ratings_db = self.client['ratings']
        self.ratings_collection = self.get_collection(
            self.ratings_db, 'ratings')

    def init_language_learning_collection(self):
        self.language_learning_db = self.client['language_learning']
        self.language_learning_collection = self.get_collection(
            self.language_learning_db, 'language_learning_results', wipe=clear_cache_on_start)
        self.ll_word_suggest_upgrade_db = self.client['ll_word_suggest_upgrade']
        self.ll_word_suggest_upgrade_collection = self.get_collection(
            self.ll_word_suggest_upgrade_db, 'll_word_suggest_upgrade_results', wipe=clear_cache_on_start)
        self.ll_context_convo_db = self.client['ll_context_convo']
        self.ll_context_convo_collection = self.get_collection(
            self.ll_context_convo_db, 'll_context_convo_results', wipe=clear_cache_on_start)

    def init_gps_location_collection(self):
        self.gps_location_db = self.client['gps_location']
        self.gps_location_collection = self.get_collection(
            self.gps_location_db, 'gps_location_result_ids', wipe=True) # When the server restarts, we don't want to keep old locations, therefore we reset the locations for all users.

    def init_topic_shifts_collection(self):
        self.topic_shifts_db = self.client['topic_shifts']
        self.topic_shifts_collection = self.get_collection(
            self.topic_shifts_db, 'topic_shifts_results', wipe=clear_cache_on_start)

    def get_collection(self, db, collection_name, wipe=False):
        if collection_name in db.list_collection_names():
            collection = db.get_collection(collection_name)
            if wipe and self.parent_handler:
                collection.drop()
                return db.create_collection(collection_name)
            return collection
        else:
            return db.create_collection(collection_name)

    ### MISC ###

    # Returns the index of the nearest beginning of a word before "curr_index"
    # EX: find_closest_start_word_index('Zimbabwe', 4) => 0
    # EX: find_closest_start_word_index('hello world, my name is alex!', 5) => 6
    # EX: ...
    # EX: find_closest_start_word_index('hello world, my name is alex!', 11) => 6
    # EX: find_closest_start_word_index('hello world, my name is alex!', 12) => 13
    def find_closest_start_word_index(self, text, curr_index):
        # print("YO! text: `{}`, indx: `{}`".format(text, (str(curr_index))))

        if curr_index > len(text):
            return len(text)
        if " " not in text:
            return 0

        latest_stop_index = 0
        for i, c in enumerate(text):
            if c == " ":
                if (i > curr_index):
                    return latest_stop_index
                latest_stop_index = i + 1
        return curr_index

    def create_user_if_not_exists(self, user_id):
        users = self.user_collection.find()
        need_create = True
        for u in users:
            if user_id == u['user_id']:
                need_create = False
        if need_create:
            print('Creating new user: ' + user_id)
            self.user_collection.insert_one(
                {"user_id": user_id,
                 "last_wake_word_time": -1,
                 "last_recording_start_time": -1,
                 "settings": {
                     "enable_agent_proactive_definer_images": True,
                     "should_update_settings": False,
                     "target_language": "Russian",
                     "source_language": "English",
                     "transcribe_language": "English",
                     "dynamic_transcribe_language": "English", #the current dynamic transcribe language that we set momentarily
                     "use_dynamic_transcribe_language": False,
                     "is_having_language_learning_contextual_convo": False,
                     "command_start_language_learning_contextual_convo": False,
                     "current_mode": "Language Learning",
                     "vocabulary_upgrade_enabled": False, 
                     #"current_mode": "Proactive Agents",
                     "enabled_proactive_agents": ["QuestionAnswerer"]
                 },
                 "transcripts": [],
                 "ui_list": [],
                 "rating_ids": [],
                 "display_request_ids": [],
                 "agent_explicit_query_ids": [],
                 "agent_explicit_insights_result_ids": [],
                 "agent_proactive_definer_result_ids": [],
                 "agent_insights_result_ids": [],
                 "agent_insights_query_ids": [],
                 "language_learning_result_ids": [],
                 "ll_context_convo_result_ids": [],
                 "ll_word_suggest_upgrade_result_ids":[],
                 "gps_location_result_ids": [],
                 "topic_shift_result_ids": [],
                 "agent_proactive_definer_irrelevant_terms": []})

    ### CACHE ###

    def find_cached_summary(self, long_description):
        description_hash = sha256(long_description.encode("utf-8")).hexdigest()
        filter = {"description": description_hash}
        item = self.cache_collection.find_one(filter)
        if item and 'summary' in item:
            return item['summary']
        else:
            return None

    def save_cached_summary(self, long_description, summary):
        description_hash = sha256(long_description.encode("utf-8")).hexdigest()
        item = {"description": description_hash, "summary": summary}
        self.cache_collection.insert_one(item)
        
    ## ACTIVE USERS ##

    def update_active_user(self, user_id, device_id):
        current_time = int(time.time())
        self.active_user_collection.update_one(
            {"user_id": user_id, "device_id": device_id},
            {"$set": {"last_active": current_time}},
            upsert=True
        )

    def get_active_users(self, active_threshold=10):
        # if TESTING_LL_CONTEXT_CONVO_AGENT:
        #     warnings.warn("TESTING MODE: Returning test user. Please remove this warning when not testing.")
        #     return [{"user_id": "oO4QvMJELYM6jEYtLDbo1LRFLPO2", "device_id": "android"}]

        current_time = int(time.time())
        query = {"last_active": {"$gte": current_time - active_threshold}}
        active_users = self.active_user_collection.find(query)
        return [{"user_id": user['user_id'], "device_id": user['device_id']} for user in active_users]

    ### OPTIONS ###

    def get_user_settings(self, user_id):
        filter = {"user_id": user_id}
        doc = self.user_collection.find_one(filter, {'settings': 1, '_id': 0})
        return doc.get('settings', None) if doc else None


    def update_user_settings(self, user_id, settings_update):
        filter = {"user_id": user_id}
        update = {"$set": {}}
        for key, value in settings_update.items():
            update["$set"][f"settings.{key}"] = value
        result = self.user_collection.update_one(filter, update)
        if result.modified_count > 0:
            print(f'Options updated for user {user_id}.')
        else:
            print(f'No updates made for user {user_id}. Either user does not exist or no changes were necessary.')

    def update_single_user_setting(self, user_id, setting_key, setting_value):
        filter = {"user_id": user_id}
        update = {"$set": {f"settings.{setting_key}": setting_value}}
        result = self.user_collection.update_one(filter, update)
        if result.modified_count > 0:
            print(f'Setting "{setting_key}" updated for user {user_id}.')
        else:
            print(f'No updates made for user {user_id}. Either user does not exist or no changes were necessary.')


    def get_user_settings_value(self, user_id, option_key):
        filter = {"user_id": user_id}
        projection = {'settings': 1, '_id': 0}
        doc = self.user_collection.find_one(filter, projection)
        if doc and 'settings' in doc and option_key in doc['settings']:
            return doc['settings'][option_key]
        else:
            return None
        
    def get_user_feature_enabled(self, user_id, feature_name):
        current_mode = self.get_user_settings_value(user_id, "current_mode")
        return current_mode in MODES_FEATURES_MAP and feature_name in MODES_FEATURES_MAP[current_mode]

    def get_should_update_settings(self, user_id):
        should_update_settings = self.get_user_settings_value(user_id, "should_update_settings")
        if should_update_settings:
            print("should update settings was true, changing to false")
            self.update_single_user_setting(user_id, "should_update_settings", False)
        return should_update_settings


    ### TRANSCRIPTS ###

    # Returns True if we can save new transcripts, False if we are getting too many transcripts too quickly
    def transcript_rate_limiter(self, user_id, new_text):
#        max_wpm = 320  # Human speech is 100-200 WPM, use 320 for safe margin
#        transcripts, transcribe_language , device_id = self.get_transcripts_from_last_nseconds_for_user_as_string(user_id, 60) + " " + new_text
#        return len(transcripts.split()) < max_wpm
        return True

    def get_latest_transcript_from_user_obj(self, user_obj):
        user_transcripts = self.transcripts_collection.find_one({"user_id": user_obj['user_id']})
        if user_transcripts["latest_intermediate_transcript"]["timestamp"] != -1:
            return user_transcripts["latest_intermediate_transcript"]
        
        elif user_transcripts["final_transcripts"]:
            return user_transcripts["final_transcripts"][-1]
        
        else:
            return None

    def save_deepgram_transcript_for_user(self, user_id, device_id, deepgram_obj, transcribe_language):
        if not deepgram_obj:
            return
        
        timestamp = time.time()

        # Format deepgram object for db
        print("SET USER DEEPGRAM TRANSCRIPT")
        transcript_objs = []
        for word_info in deepgram_obj["words"]:
            if (len(transcript_objs) == 0) or (word_info["speaker"] != transcript_objs[-1]["speaker"]):
                new_transcript_obj = {
                                    "device_id": device_id,
                                    "timestamp": timestamp, 
                                    "text": word_info["word"], 
                                    "speaker": word_info["speaker"],
                                    "transcribe_language": transcribe_language}
                transcript_objs.append(new_transcript_obj)
            else:
                # print("PREV TRANSCRIPT: " + str(transcript_objs[-1]))
                # print("NEW WORD: " + str(word_info))
                transcript_objs[-1]["text"] += " " + word_info["word"]

        filter = {"user_id": user_id}
        update = {"$push": {"final_transcripts": { "$each": transcript_objs }}}
        res = self.transcripts_collection.update_one(filter=filter, update=update, upsert=True)
        return res

    def save_transcript_for_user(self, user_id, device_id, text, is_final, transcribe_language):
        if text == "":
            return

        text = text.strip()

        transcript = {"user_id": user_id, "device_id": device_id, "text": text,
                "timestamp": time.time(), "is_final": is_final, "uuid": str(uuid.uuid4()), "transcribe_language" : transcribe_language}

        user = self.get_user(user_id)

        if not self.transcript_rate_limiter(user_id, text):
            return False

        self.purge_old_transcripts_for_user_id(user_id)

        if is_final:
            filter = {"user_id": user_id}
            update = {"$push": {"final_transcripts": transcript}}
            self.transcripts_collection.update_one(filter=filter, update=update, upsert=True)

            update = {"$set": {"latest_intermediate_transcript": self.empty_transcript}}
            self.transcripts_collection.update_one(filter=filter, update=update, upsert=True)
        else:
            filter = {"user_id": user_id}
            update = {"$set": {"latest_intermediate_transcript": transcript}}
            self.transcripts_collection.update_one(filter=filter, update=update, upsert=True)

        return transcript

    def get_user(self, user_id):
        user = self.user_collection.find_one({"user_id": user_id})
        if not user:
            self.create_user_if_not_exists(user_id)
            return self.get_user(user_id)
        return user

    def get_all_transcripts_for_user(self, user_id, delete_after=False):
        transcripts = []
        user_transcripts = self.transcripts_collection.find_one({"user_id": user_id})
        if not user_transcripts: return []
        
        if "final_transcripts" in user_transcripts:
            transcripts = user_transcripts["final_transcripts"]

        if "latest_intermediate_transcript" in user_transcripts and user_transcripts["latest_intermediate_transcript"]["text"]:
            transcripts.append(user_transcripts["latest_intermediate_transcript"])

        return transcripts

    def combine_text_from_transcripts(self, transcripts, recent_transcripts_only=True):
        curr_time = time.time()
        last_index = len(transcripts) - 1
        text = ""

        for index, t in enumerate(transcripts):
            if recent_transcripts_only and (curr_time - t['timestamp'] < self.final_transcript_validity_time):
                text += t['text']
            elif not recent_transcripts_only:
                text += t['text']

            if index != last_index:
                text += " "

        return text

    def get_recent_transcripts_from_last_nseconds_for_all_users(self, n=30, users_list=None, stringify = True):
        users = self.user_collection.find() if users_list is None else users_list
        transcripts = []
        for user in users:
            user_id = user['user_id']
            if stringify:
                transcript_string, transcribe_language, device_id = self.get_transcripts_from_last_nseconds_for_user_as_string(
                    user_id, n)
                if transcript_string:
                    transcripts.append(
                            {'user_id': user_id, 'device_id': device_id, 'text': transcript_string, 'transcribe_language': transcribe_language})
            else:
                transcript_list = self.get_transcripts_from_last_nseconds_for_user(
                    user_id, n)
                if transcript_list:
                    transcripts.append(
                            {'user_id': user_id, 'transcripts': transcript_list})
        return transcripts

    def get_transcripts_from_last_nseconds_for_user(self, user_id, n=30, transcript_list=None):
        all_transcripts = transcript_list if transcript_list else self.get_all_transcripts_for_user(
            user_id)

        recent_transcripts = []
        current_time = time.time()
        for transcript in all_transcripts:
            if current_time - transcript['timestamp'] < n:
                recent_transcripts.append(transcript)
        return recent_transcripts

    def get_transcripts_from_last_nseconds_for_user_as_string(self, user_id, n=30, transcript_list=None):
        transcripts = self.get_transcripts_from_last_nseconds_for_user(
            user_id, n, transcript_list)
        return self.stringify_transcripts(transcript_list=transcripts)

    def purge_old_transcripts_for_user_id(self, user_id):
        transcript_expiration_date = time.time() - self.transcript_expiration_time
        filter = {'user_id': user_id}
        condition = {'$pull': {'transcripts': {
            'timestamp': {'$lt': transcript_expiration_date}}}}
        self.user_collection.update_many(filter, condition)

    ### TRANSCRIPT FORMATTING ###

    def stringify_transcripts(self, transcript_list):
        output = ""
        if len(transcript_list) == 0:
            return output, "English", None

        for index, t in enumerate(transcript_list):
            output = output + t['text'] + ' '

        return output.strip(), transcript_list[0]["transcribe_language"], transcript_list[0]["device_id"]

    ### RECORDING ###

    def update_recording_time_for_user(self, user_id):
        self.create_user_if_not_exists(user_id)
        current_time = time.time()

        # Only update if the time is -1
        query_condition = {
            "user_id": user_id,
            '$or': [{'last_recording_start_time': -1}]
        }

        update = {"$set": {"last_recording_start_time": current_time}}

        result = self.user_collection.update_one(query_condition, update)
        return True if result.modified_count else False

    def reset_recording_time_for_user(self, user_id):
        old_recording_time = self.get_user(
            user_id)['last_recording_start_time']
        filter = {"user_id": user_id}
        update = {"$set": {"last_recording_start_time": -1}}
        self.user_collection.update_one(filter=filter, update=update)
        return old_recording_time

    def save_recording(self, user_id, recording_name):
        print("Saving recording")
        old_recording_time = self.reset_recording_time_for_user(user_id)
        if old_recording_time == -1:
            return []

        results_timeframe = time.time() - old_recording_time
        results = self.get_defined_terms_from_last_nseconds_for_user_device(
            user_id, results_timeframe)
        for r in results:
            time_since_recording_start = r['timestamp'] - old_recording_time
            r['time_since_recording_start'] = time_since_recording_start
        return results

        # TODO: Save to database here?
        print("Recording saved: " + recording_name)

    ### WAKE WORDS ###

    def update_wake_word_time_for_user(self, user_id):
        self.create_user_if_not_exists(user_id)
        current_time = time.time()

        # Only update if we haven't already noted a wake word within the last 2 seconds, OR if the time is -1
        query_condition = {
            "user_id": user_id,
            '$or': [
                # {'last_wake_word_time': {'$lt': (current_time - self.wake_word_min_update_time)}},
                {'last_wake_word_time': -1}
            ]
        }

        update = {"$set": {"last_wake_word_time": current_time}}

        result = self.user_collection.update_one(query_condition, update)
        return True if result.modified_count else False

    def get_users_with_recent_wake_words(self):
        filter = {"last_wake_word_time": {'$ne': -1}}
        relevant_users = self.user_collection.find(filter)
        return relevant_users

    def get_wake_word_time_for_user(self, user_id):
        user = self.get_user(user_id)
        return user["last_wake_word_time"]

    def reset_wake_word_time_for_user(self, user_id):
        self.create_user_if_not_exists(user_id)
        filter = {"user_id": user_id}
        update = {"$set": {"last_wake_word_time": -1}}
        self.user_collection.update_one(filter=filter, update=update)

    def check_for_wake_words_in_transcript_text(self, user_id, text):
        # if agents.wake_words.does_text_contain_wake_word(text):
        #   return self.update_wake_word_time_for_user(user_id)
        return False

    ### Explicit Queries ###

    def add_explicit_query_for_user(self, user_id, query):
        query_time = math.trunc(time.time())
        query_uuid = str(uuid.uuid4())
        query_obj = {'timestamp': query_time,
                     'uuid': query_uuid, 'query': query}
        self.agent_explicit_queries_collection.insert_one(query_obj)

        filter = {"user_id": user_id}
        update = {"$push": {"agent_explicit_query_ids": query_uuid}}
        self.user_collection.update_one(filter=filter, update=update)

        return query_uuid

    def add_explicit_insight_result_for_user(self, user_id, query, insight):
        insight_time = math.trunc(time.time())
        insight_uuid = str(uuid.uuid4())
        insight_obj = {'timestamp': insight_time,
                       'uuid': insight_uuid, 'query': query, 'insight': insight}
        self.agent_explicit_insights_results_collection.insert_one(insight_obj)

        filter = {"user_id": user_id}
        update = {"$push": {"agent_explicit_insights_result_ids": insight_uuid}}
        self.user_collection.update_one(filter=filter, update=update)

    def get_explicit_query_history_for_user(self, user_id, device_id=None, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("agent_explicit_query_ids", user_id, device_id, should_consume, include_consumed)

    def get_recent_n_seconds_explicit_query_history_for_user(
        self, user_id, n_seconds=10
    ):
        uuid_list = self.get_user(user_id)["agent_explicit_query_ids"]
        current_time = math.trunc(time.time())
        timestamp_threshold = current_time - n_seconds
        pipeline = [
            {
                "$match": {
                    "uuid": {"$in": uuid_list},
                    "timestamp": {"$gte": timestamp_threshold},
                }
            },
            {"$sort": {"timestamp": -1}},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        return list(
            self.agent_explicit_queries_collection.aggregate(pipeline))

    def get_explicit_insights_history_for_user(self, user_id, device_id=None, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("agent_explicit_insights_result_ids", user_id, device_id, should_consume, include_consumed)

    ### display_requests ###

    def add_display_request_for_user(self, user_id, data):
        message_uuid = str(uuid.uuid4())
        message_obj = {'timestamp': time.time(),
                       'uuid': message_uuid, 'data': data}
        self.display_requests_collection.insert_one(message_obj)

        filter = {"user_id": user_id}
        update = {"$push": {"display_request_ids": message_uuid}}
        self.user_collection.update_one(filter=filter, update=update)

    def delete_display_request_ids_for_user(self, user_id):
        filter = {"user_id": user_id}
        update = {"$set": {"display_request_ids": []}}
        self.user_collection.update_one(filter=filter, update=update)

    def get_display_requests_for_user_device(self, user_id, device_id, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("display_request_ids", user_id, device_id, should_consume, include_consumed)

    ### PROACTIVE INSIGHTS ###

    # TODO: consult kenji here // test this more
    def get_agent_insights_history_for_user(self, user_id, top=10):
        uuid_list = self.get_user(user_id)["agent_insights_result_ids"]
        pipeline = [
            {"$match": {"uuid": {"$in": uuid_list}}},
            # { "$unwind": "$agent_insights_result_ids" },
            {"$sort": {"timestamp": -1}},
            {"$limit": top},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(
            self.agent_insights_results_collection.aggregate(pipeline))

        # logger.log(logging.DEBUG, "{}: Insights history RESULTS: {}".format("get_agent_insights_history_for_user", results))

        return results
    
    def get_recent_n_seconds_agent_insights_query_history_for_user(
            self, user_id, n_seconds=10
        ):
            uuid_list = self.get_user(user_id)["agent_insights_query_ids"]
            current_time = math.trunc(time.time())
            timestamp_threshold = current_time - n_seconds
            pipeline = [
                {
                    "$match": {
                        "uuid": {"$in": uuid_list},
                        "timestamp": {"$gte": timestamp_threshold},
                    }
                },
                {"$sort": {"timestamp": -1}},
                {
                    "$project": {
                        "_id": 0,
                    }
                },
            ]
            return list(
                self.agent_insights_queries_collection.aggregate(pipeline))

    def get_recent_nminutes_agent_insights_history_for_user(
        self, user_id, n_minutes=10
    ):
        uuid_list = self.get_user(user_id)["agent_insights_result_ids"]
        current_time = math.trunc(time.time())
        n_seconds = n_minutes * 60
        timestamp_threshold = current_time - n_seconds

        pipeline = [
            {
                "$match": {
                    "uuid": {"$in": uuid_list},
                    "timestamp": {"$gte": timestamp_threshold},
                }
            },
            {"$sort": {"timestamp": -1}},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        return list(
            self.agent_insights_results_collection.aggregate(pipeline))

    def get_proactive_agents_insights_results_for_user_device(self, user_id, device_id, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("agent_insights_result_ids", user_id, device_id, should_consume, include_consumed)

    def get_defined_terms_from_last_nseconds_for_user_device(self, user_id, n=300):
        consumed_results = self.get_display_requests_for_user_device(
            user_id=user_id, device_id="", should_consume=False, include_consumed=True)

        previously_defined_terms = []
        current_time = math.trunc(time.time())
        for result in consumed_results:
            if current_time - result['timestamp'] < n:
                previously_defined_terms.append(result)
        return previously_defined_terms

    def add_agent_insight_query_for_user(self, user_id, query):
        query_time = math.trunc(time.time())
        query_uuid = str(uuid.uuid4())
        query_obj = {'timestamp': query_time,
                        'uuid': query_uuid, 'query': query}
        self.agent_insights_queries_collection.insert_one(query_obj)

        filter = {"user_id": user_id}
        update = {"$push": {"agent_insights_query_ids": query_uuid}}
        self.user_collection.update_one(filter=filter, update=update)

    def add_agent_insight_result_for_user(self, user_id, agent_name, agent_insight, agent_references=None, agent_motive=None):
        insight_time = math.trunc(time.time())
        insight_uuid = str(uuid.uuid4())
        if agent_references == "":
            agent_references = None
        insight_obj = {'timestamp': insight_time, 'uuid': insight_uuid, 'agent_name': agent_name,
                       'agent_insight': agent_insight, 'url': agent_references, 'agent_motive': agent_motive}
        self.agent_insights_results_collection.insert_one(insight_obj)

        filter = {"user_id": user_id}
        update = {"$push": {"agent_insights_result_ids": insight_uuid}}
        self.user_collection.update_one(filter=filter, update=update)

    ### INTELLIGENT ENTITY DEFINITIONS ###

    def get_definer_history_for_user(self, user_id, top=5):
        uuid_list = self.get_user(
            user_id)["agent_proactive_definer_result_ids"]
        pipeline = [
            {"$match": {"uuid": {"$in": uuid_list}}},
            # { "$match": { "user_id": user_id } },
            {"$sort": {"timestamp": -1}},
            {"$limit": top},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(
            self.agent_proactive_definer_collection.aggregate(pipeline))

        # logger.log(logging.DEBUG, "{}: Definer history RESULTS: {}".format("get_definer_history_for_user", results))

        return results

    def get_recent_nminutes_definer_history_for_user(self, user_id, n_minutes=10):
        uuid_list = self.get_user(
            user_id)["agent_proactive_definer_result_ids"]
        current_time = math.trunc(time.time())
        n_seconds = n_minutes * 60
        timestamp_threshold = current_time - n_seconds

        pipeline = [
            {
                "$match": {
                    "uuid": {"$in": uuid_list},
                    "timestamp": {"$gte": timestamp_threshold},
                }
            },
            {"$sort": {"timestamp": -1}},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(
            self.agent_proactive_definer_collection.aggregate(pipeline))

        # Extracting only names from results
        names = [result["name"] for result in results]

        # logger.log(logging.DEBUG, "{}: Definer history RESULTS: {}".format("get_recent_nminutes_definer_history_for_user", names))

        return names

    def get_language_learning_words_defined_history_for_user(self, user_id, top=5):
        uuid_list = self.get_user(user_id)["language_learning_result_ids"]
        pipeline = [
            {"$match": {"uuid": {"$in": uuid_list}}},
            {"$sort": {"timestamp": -1}},
            {"$limit": top},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(self.language_learning_collection.aggregate(pipeline))

        return results

    def get_ll_context_convo_history_for_user(self, user_id, top=2):
        uuid_list = self.get_user(user_id)["ll_context_convo_result_ids"]
        pipeline = [
            {"$match": {"uuid": {"$in": uuid_list}}},
            {"$sort": {"timestamp": -1}},
            {"$limit": top},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(self.ll_context_convo_collection.aggregate(pipeline))

        return results
    
    def get_ll_word_suggest_upgrade_histroy_for_user(self, user_id, top=2):
        uuid_list = self.get_user(user_id)["ll_word_suggest_upgrade_result_ids"]
        pipeline = [
            {"$match": {"uuid": {"$in": uuid_list}}},
            {"$sort": {"timestamp": -1}},
            {"$limit": top},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(self.ll_word_suggest_upgrade_collection.aggregate(pipeline))

        return results

    def get_gps_location_for_user(self, user_id, top=1):
        uuid_list = self.get_user(user_id)["gps_location_result_ids"]
        pipeline = [
            {"$match": {"uuid": {"$in": uuid_list}}},
            {"$sort": {"timestamp": -1}},
            {"$limit": top},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]

        results = list(self.gps_location_collection.aggregate(pipeline))

        return results

    def get_latest_topic_shift_for_user(self, user_id, top=1, true_shift=True):
        uuid_list = self.get_user(user_id)["topic_shift_result_ids"]
        match_condition = {"uuid": {"$in": uuid_list}}
        if true_shift is not None:
            match_condition["true_shift"] = true_shift
        pipeline = [
            {"$match": match_condition},
            {"$sort": {"timestamp": -1}},
            {"$limit": top},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]

        results = list(self.topic_shifts_collection.aggregate(pipeline))
        return results

    def get_recent_language_learning_words_defined_history_for_user(self, user_id, n_seconds=10):
        uuid_list = self.get_user(user_id)["language_learning_result_ids"]
        current_time = math.trunc(time.time())
        timestamp_threshold = current_time - n_seconds

        pipeline = [
            {
                "$match": {
                    "uuid": {"$in": uuid_list},
                    "timestamp": {"$gte": timestamp_threshold},
                }
            },
            {"$sort": {"timestamp": -1}},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(
            self.language_learning_collection.aggregate(pipeline))

        names = [result["in_word"] for result in results]

        return names

    def get_recent_nminutes_ll_word_suggest_upgrade_history_for_user(self, user_id, n_minutes=10):
        uuid_list = self.get_user(user_id)["ll_word_suggest_upgrade_result_ids"]
        current_time = math.trunc(time.time())
        n_seconds = n_minutes * 60
        timestamp_threshold = current_time - n_seconds

        pipeline = [
            {
                "$match": {
                    "uuid": {"$in": uuid_list},
                    "timestamp": {"$gte": timestamp_threshold},
                }
            },
            {"$sort": {"timestamp": -1}},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(
            self.ll_word_suggest_upgrade_collection.aggregate(pipeline))

        names = [result["in_upgrade"] for result in results]

        return names


    def get_recent_nminutes_ll_context_convo_history_for_user(self, user_id, n_minutes=10):
        uuid_list = self.get_user(user_id)["ll_context_convo_result_ids"]
        current_time = math.trunc(time.time())
        n_seconds = n_minutes * 60
        timestamp_threshold = current_time - n_seconds

        pipeline = [
            {
                "$match": {
                    "uuid": {"$in": uuid_list},
                    "timestamp": {"$gte": timestamp_threshold},
                }
            },
            {"$sort": {"timestamp": -1}},
            {
                "$project": {
                    "_id": 0,
                }
            },
        ]
        results = list(
            self.ll_context_convo_collection.aggregate(pipeline))

        names = [result["in_word"] for result in results]

        return names

    def add_agent_proactive_definition_results_for_user(self, user_id, entities):
        if not entities:
            return

        for entity in entities:
            if entity is None:
                continue

            entity['timestamp'] = int(time.time())
            entity['uuid'] = str(uuid.uuid4())

        self.agent_proactive_definer_collection.insert_many(entities)

        result_ids = []
        for e in entities:
            result_ids.append(e['uuid'])

        filter = {"user_id": user_id}
        update = {
            "$push": {"agent_proactive_definer_result_ids": {'$each': result_ids}}}
        self.user_collection.update_one(filter=filter, update=update)

    def get_agent_proactive_definer_results_for_user_device(self, user_id, device_id, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("agent_proactive_definer_result_ids", user_id, device_id, should_consume, include_consumed)

    def get_agent_proactive_definer_irrelevant_terms(self, user_id):
        user = self.get_user(user_id)
        return user["agent_proactive_definer_irrelevant_terms"]

    def push_agent_proactive_definer_irrelevant_term(self, user_id, new_term):
        filter = {"user_id": user_id}
        update = {
            "$push": {"agent_proactive_definer_irrelevant_terms": {"$each": [new_term]}}}
        self.user_collection.update_one(filter=filter, update=update)

        # Cut off start of list if too large
        max_size = 40
        self.user_collection.update_one(
            {'user_id': user_id},
            [
                {'$set': {
                    'agent_proactive_definer_irrelevant_terms': {
                        '$cond': {
                            'if': {'$gt': [{'$size': "$agent_proactive_definer_irrelevant_terms"}, max_size]},
                            'then': {'$slice': ["$agent_proactive_definer_irrelevant_terms", 1, {'$size': "$agent_proactive_definer_irrelevant_terms"}]},
                            'else': "$agent_proactive_definer_irrelevant_terms"
                        }
                    }
                }}
            ]
        )

    ### UI DEVICE ###

    def get_all_ui_devices_for_user(self, user_id):
        user = self.user_collection.find_one({"user_id": user_id})
        ui_list = user['ui_list']
        ui_list_ids = []
        for ui in ui_list:
            ui_list_ids.append(ui['device_id'])
        return ui_list_ids

    def add_ui_device_to_user_if_not_exists(self, user_id, device_id):
        self.create_user_if_not_exists(user_id)
        user = self.user_collection.find_one({"user_id": user_id})

        need_add = True
        if user['ui_list'] != None:
            for ui in user['ui_list']:
                if ui['device_id'] == device_id:
                    need_add = False

        if need_add:
            print("Creating device for user '{}': {}".format(user_id, device_id))
            ui_object = {"device_id": device_id, "consumed_result_ids": [
            ]}
            filter = {"user_id": user_id}
            update = {"$addToSet": {"ui_list": ui_object}}
            self.user_collection.update_one(filter=filter, update=update)

    ### INSIGHT RATING ###

    # Rating should be an integer between 1-10, with 0 being lame and 10 being super not lame
    def rate_result_by_uuid(self, user_id, result_uuid, rating):
        if not isinstance(rating, (int)):
            return "Rating must be an integer"
        if rating < 0 or rating > 10:
            return "Rating must be an integer 0 - 10"

        rating_time = math.trunc(time.time())
        rating_uuid = str(uuid.uuid4())
        rating_context, transcribe_language, device_id = self.get_transcripts_from_last_nseconds_for_user_as_string(
            user_id, n=240)
        rating_obj = {"uuid": rating_uuid, "timestamp": rating_time,
                      "result_uuid": result_uuid, "rating": rating, "context": rating_context}
        self.ratings_collection.insert_one(rating_obj)

        filter = {"user_id": user_id}
        update = {"$push": {"rating_ids":  rating_uuid}}
        self.user_collection.update_one(filter=filter, update=update)

    def get_result_ratings_for_user(self, user_id):
        return self.get_results_for_user_device("rating_ids", user_id, device_id=None, should_consume=False, include_consumed=True)

    def get_result_rating_from_result_uuid(self, insight_uuid):
        filter = {"result_uuid": insight_uuid}
        return self.ratings_collection.find_many(filter=filter)

    ### GENERIC ###

    # Search all results collections for a specific UUID
    def get_result_from_uuid(self, uuid):
        filter = {"uuid": uuid}
        res = self.display_requests_collection.find_one(filter, {'_id': 0})
        if res:
            return res
        res = self.agent_explicit_queries_collection.find_one(filter, {
                                                              '_id': 0})
        if res:
            return res
        res = self.agent_explicit_insights_results_collection.find_one(filter, {
                                                                       '_id': 0})
        if res:
            return res
        res = self.agent_insights_queries_collection.find_one(filter, {
                                                              '_id': 0})
        
        if res:
            return res
        res = self.agent_insights_results_collection.find_one(filter, {
                                                              '_id': 0})
        if res:
            return res
        res = self.agent_proactive_definer_collection.find_one(filter, {
                                                               '_id': 0})
        if res:
            return res
        res = self.language_learning_collection.find_one(filter, {'_id': 0})
        if res:
            return res
        res = self.ll_word_suggest_upgrade_collection.find_one(filter, {'_id': 0})
       
        if res:
            return res
        res = self.ll_context_convo_collection.find_one(filter, {'_id': 0})
        if res:
            return res
        res = self.gps_location_collection.find_one(filter, {'_id': 0})
        if res:
            return res
        res = self.topic_shifts_collection.find_one(filter, {'_id': 0})
        if res:
            return res

        return None

    def get_results_for_user_device(self, result_type, user_id, device_id, should_consume=True, include_consumed=False):
        self.add_ui_device_to_user_if_not_exists(user_id, device_id)

        user = self.user_collection.find_one({"user_id": user_id})

        if result_type not in user:
            raise Exception("Invalid result type: `{}`".format(str(result_type)))

        result_ids = user[result_type] if user != None else []

        # if result_ids != [] and result_type == "topic_shift_result_ids":
        #     print("running get results for user with result_type as " + result_type)
        #     print(result_ids)

        already_consumed_ids = [
        ] if include_consumed else self.get_consumed_result_ids_for_user_device(user_id, device_id)
        new_results = []

        for uuid in result_ids:
            if uuid not in already_consumed_ids:
                if should_consume:
                    self.add_consumed_result_id_for_user_device(
                        user_id, device_id, uuid)
                result = self.get_result_from_uuid(uuid)

                if result is not None:
                    new_results.append(result)

        return new_results

    def get_consumed_result_ids_for_user_device(self, user_id, device_id):
        filter = {"user_id": user_id, "ui_list.device_id": device_id}
        user = self.user_collection.find_one(filter=filter)
        if user == None or user['ui_list'] == None or user['ui_list'][0] == None:
            return []
        to_return = user['ui_list'][0]['consumed_result_ids']
        return to_return if to_return != None else []

    def add_consumed_result_id_for_user_device(self, user_id, device_id, consumed_result_uuid):
        filter = {"user_id": user_id, "ui_list.device_id": device_id}
        update = {"$addToSet": {
            "ui_list.$.consumed_result_ids": consumed_result_uuid}}
        # "$add_to_set": {"ui_list": device_id}}
        self.user_collection.update_many(filter=filter, update=update)


    def add_language_learning_words_to_show_for_user(self, user_id, words):
        for word in words:
            if word is None:
                continue

            word['timestamp'] = int(time.time())
            word['uuid'] = str(uuid.uuid4())

        self.language_learning_collection.insert_many(words)

        result_ids = []
        for e in words:
            result_ids.append(e['uuid'])

        filter = {"user_id": user_id}
        update = {"$push": {"language_learning_result_ids": {'$each': result_ids}}}
        self.user_collection.update_one(filter=filter, update=update)

    def add_ll_word_suggest_upgrade_to_show_for_user(self, user_id, words):
        for word in words:
            if word is None:
                continue

            word['timestamp'] = int(time.time())
            word['uuid'] = str(uuid.uuid4())

        self.ll_word_suggest_upgrade_collection.insert_many(words)

        result_ids = []
        for e in words:
            result_ids.append(e['uuid'])

        filter = {"user_id": user_id}
        update = {"$push": {"ll_word_suggest_upgrade_result_ids": {'$each': result_ids}}}
        self.user_collection.update_one(filter=filter, update=update)

    def get_language_learning_results_for_user_device(self, user_id, device_id, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("language_learning_result_ids", user_id, device_id, should_consume, include_consumed)

    def add_ll_context_convo_results_for_user(self, user_id, reponse):
        if not reponse:
            return
        reponse['timestamp'] = int(time.time())
        reponse['uuid'] = str(uuid.uuid4())

        print("INSERTING reponse: " + str(reponse))
        self.ll_context_convo_collection.insert_one(reponse)

        filter = {"user_id": user_id}
        update = {"$push": {"ll_context_convo_result_ids": reponse['uuid']}}
        self.user_collection.update_one(filter=filter, update=update)

    def get_ll_context_convo_results_for_user_device(self, user_id, device_id, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("ll_context_convo_result_ids", user_id, device_id, should_consume, include_consumed)

    def get_ll_word_suggest_upgrade_results_for_user_device(self, user_id, device_id, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("ll_word_suggest_upgrade_result_ids", user_id, device_id, should_consume, include_consumed)

    def get_adhd_stmb_results_for_user_device(self, user_id, device_id, should_consume=True, include_consumed=False):
        return self.get_results_for_user_device("topic_shift_result_ids", user_id, device_id, should_consume, include_consumed)

    def add_gps_location_for_user(self, user_id, location):
        if not location:
            print("No location to add")
            return
        
        location['timestamp'] = int(time.time())
        location['uuid'] = str(uuid.uuid4())

        # print("INSERTING location: " + str(location))
        self.gps_location_collection.insert_one(location)

        filter = {"user_id": user_id}
        update = {
            "$push": {
                "gps_location_result_ids": {
                    "$each": [location['uuid']],
                    "$slice": -10  # Keep only the latest 10 entries
                }
            }
        }
        self.user_collection.update_one(filter=filter, update=update)

    def add_topic_shift_for_user(self, user_id, time_of_shift, summary, true_shift=False):
        if not time_of_shift:
            print("No topic shift timestamp to add")
            return
        
        topic_shift = dict()
        topic_shift['time_of_shift'] = time_of_shift
        topic_shift['summary'] = summary
        topic_shift['true_shift'] = true_shift
        topic_shift['timestamp'] = int(time.time())
        topic_shift['uuid'] = str(uuid.uuid4())

        print("INSERTING TOPIC SHIFT: " + str(topic_shift))
        self.topic_shifts_collection.insert_one(topic_shift)

        filter = {"user_id": user_id}
        update = {"$push": {"topic_shift_result_ids": topic_shift['uuid']}}
        self.user_collection.update_one(filter=filter, update=update)

    def get_gps_location_results_for_user_device(self, user_id, device_id, should_consume=False, include_consumed=False):
        return self.get_results_for_user_device("gps_location_result_ids", user_id, None, should_consume, include_consumed) # TODO: chek if device_id can be None
