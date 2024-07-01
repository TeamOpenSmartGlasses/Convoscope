import time
import traceback
import asyncio
from math import radians, cos, sin, sqrt, atan2, ceil
import warnings

# custom
from DatabaseHandler import DatabaseHandler
from agents.ll_context_convo_agent import run_ll_context_convo_agent
from agents.helpers.get_nearby_places import get_user_location, get_nearby_places
from constants import TESTING_LL_CONTEXT_CONVO_AGENT, LL_CONTEXT_CONVO_AGENT, SKIP_AUTO_LL_CONTEXT_CONVO_AGENT


response_period = 2 # how long of a pause to wait before we send the user's response to the agent
break_convo_time_limit = 20 # how long of silence/no response before breaking out of the conversation
wpm_threshold = 10 # wpm, if greater than this, don't run a contextual convo
speed_threshold = 0.62 # speed of movement, if less than this, don't run a contextual convo

if TESTING_LL_CONTEXT_CONVO_AGENT:
    run_period = 0.25
    cooldown_period = 20
else:
    run_period = 0.2
    cooldown_period = 60

transcript_period = 40


def lat_lng_to_meters(lat1, lng1, lat2, lng2):
    # Radius of the Earth in km
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lng1_rad = radians(lng1)
    lat2_rad = radians(lat2)
    lng2_rad = radians(lng2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlng / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in kilometers
    distance_km = R * c

    # Convert km to meters
    distance_meters = distance_km * 1000

    return distance_meters


async def cleanup_conversation(user_id, db_handler, cooldown=True):
    print("Ending ll context convo with user: ", user_id)
    db_handler.update_single_user_setting(user_id, "use_dynamic_transcribe_language", False) # conversation ending, so stop using dynamic transcribe language
    db_handler.update_single_user_setting(user_id, "should_update_settings", True)
    db_handler.update_single_user_setting(user_id, "is_having_language_learning_contextual_convo", False)
    db_handler.update_single_user_setting(user_id, "command_start_language_learning_contextual_convo", False)
    print("Ended ll context convo with user: ", user_id)

    if cooldown:
        time.sleep(cooldown_period) # sleep for a while after the conversation ends

    return


async def handle_user_conversation(user_id, device_id, db_handler, force_conversation=False):
    # print("CHECKING IF WE SHOULD RUN A CONTEXTUAL CONVO FOR USER: ", user_id, device_id)
    target_language = db_handler.get_user_settings_value(user_id, "target_language")
    locations = db_handler.get_gps_location_results_for_user_device(user_id, None)
    transcripts = db_handler.get_transcripts_from_last_nseconds_for_user_as_string(user_id, n=transcript_period)
    # print("Locations: ", locations)

    if force_conversation and locations: # force the conversation to start
        pass
    elif len(locations) > 2: # this block checks if the user is moving or talking, if so, it skips the conversation
        location_1 = locations[-3]
        location_2 = locations[-2]
        location_3 = locations[-1]

        # Calculate the displacement
        displacement_1 = lat_lng_to_meters(lat1=location_1['lat'], lng1=location_1['lng'], lat2=location_2['lat'], lng2=location_2['lng'])
        displacement_2 = lat_lng_to_meters(lat1=location_2['lat'], lng1=location_2['lng'], lat2=location_3['lat'], lng2=location_3['lng'])

        delta_time_1 = location_2['timestamp'] - location_1['timestamp']
        delta_time_2 = location_3['timestamp'] - location_2['timestamp']

        speed_1 = displacement_1 / delta_time_1 if delta_time_1 > 0 else 0
        speed_2 = displacement_2 / delta_time_2 if delta_time_2 > 0 else 0

        # Get the minimum speed to ensure we are not overestimating the user's speed
        speed = min(speed_1, speed_2)

        current_wpm = len(transcripts[0].split(" ")) / (transcript_period / 60)
        # print("Current user WPM is: ", current_wpm)
        # print("Current user SPEED is: ", speed)

        if TESTING_LL_CONTEXT_CONVO_AGENT:
            warnings.warn("Currently in testing mode, skipping speed and WPM checks, please set the TESTING flag to False to run normally.")
        elif SKIP_AUTO_LL_CONTEXT_CONVO_AGENT:
            return
        elif current_wpm > wpm_threshold: # compute words per minute
            print("User is talking, skipping.")
            return
        elif speed < speed_threshold:
            print("User is not moving, skipping")
            return
    else:
        print("Not enough locations, please wait")
        await cleanup_conversation(user_id, db_handler)
        return

    conversation_history = []

    # conversation starting, so change the user's transcribe language to the target language
    db_handler.update_single_user_setting(user_id, "dynamic_transcribe_language", target_language)
    db_handler.update_single_user_setting(user_id, "use_dynamic_transcribe_language", True)
    db_handler.update_single_user_setting(user_id, "is_having_language_learning_contextual_convo", True)
    db_handler.update_single_user_setting(user_id, "should_update_settings", True)

    user_location = locations[-1] # get the most recent location
    # this block runs the contextual conversation agent until the conversation ends
    print("NOW RUNNING CONTEXTUAL CONVO FOR USER: ", user_id)
    while True:
        places = get_nearby_places(user_location)
        places = places[:min(len(places), 3)] # get max 3 nearby places
        print("LOOP ll contextual conversation")
        response = run_ll_context_convo_agent(places=places, target_language=target_language, fluency_level=8, conversation_history=conversation_history)

        # get the response for the agent
        response_context_for_agent = response['ll_context_for_agent']

        if response:
            db_handler.add_ll_context_convo_results_for_user(
                user_id, response)
        else:
            await cleanup_conversation(user_id, db_handler)
            return

        conversation_history.append(
            {"role": "agent", "content": response_context_for_agent})

        start_conversation_time = time.time()
        user_has_responded = False

        # This block waits for the user to respond
        while True: # or not user_reponse:
            # print(db_handler.get_user_settings(user_id)['is_having_language_learning_contextual_convo'])

            if not db_handler.get_user_settings(user_id)['is_having_language_learning_contextual_convo']:
                print("LL CONTEXT CONVO - User has ended the conversation, so exiting")
                await cleanup_conversation(user_id, db_handler, cooldown=False)
                return

            if not user_has_responded: # at the start of the conversation, we don't want to break out of the conversation too soon
                if time.time() - start_conversation_time > break_convo_time_limit:
                    print("LL CONTEXT CONVO - No new user response and we passed the conversational break time, so exiting")
                    await cleanup_conversation(user_id, db_handler)
                    return
            elif new_response[0]:
                print("NEW RESPONSE")
            else:
                print("NO NEW RESPONSE")
                break

            await asyncio.sleep(response_period)
            new_response = db_handler.get_transcripts_from_last_nseconds_for_user_as_string(user_id, n=response_period) # check for new user response

            if new_response[0] and not user_has_responded:
                user_has_responded = True # user has responded, so we can break out of the conversation if they don't respond again
        
        user_reponse = db_handler.get_transcripts_from_last_nseconds_for_user_as_string(user_id, n=ceil(time.time() - start_conversation_time)) # After the user has responded, get the full response
        user_reponse = user_reponse[0]

        # print("USER RESPONSE")
        # print(user_reponse)

        conversation_history.append({"role": "user", "content": user_reponse})

        locations = db_handler.get_gps_location_results_for_user_device(user_id, device_id)
        if len(locations) > 1: # update places if we have new locations
            places = get_nearby_places(locations[-1])

            if places:
                continue

        places = None

    # await cleanup_conversation(user_id, db_handler)
    # return


async def ll_context_convo_agent_processing_loop_async():
    print("START CONTEXTUAL CONVO PROCESSING LOOP ASYNC")
    db_handler = DatabaseHandler(parent_handler=False)

    # Ensure we are using an async event loop
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()

    # wait for some transcripts to load in
    await asyncio.sleep(30)

    # This block initiates the contextual conversation agent for each active user
    while True:
        if not db_handler.ready:
            print("db_handler not ready")
            await asyncio.sleep(0.1)
            continue

        try:
            pLoopStartTime = time.time()
            # print("RUNNING CONTEXTUAL CONVO LOOP ASYNC")

            active_users = db_handler.get_active_users()
            tasks = []

            # start a conversation for each active user
            for user in active_users:
                user_id = user['user_id']
                device_id = user['device_id']

                if db_handler.get_user_settings_value(user_id, "is_having_language_learning_contextual_convo"):
                    # print("Found ongoing conversation for user: ", user_id)
                    continue
                if not db_handler.get_user_feature_enabled(user_id, LL_CONTEXT_CONVO_AGENT):
                    continue
                
                if db_handler.get_user_settings_value(user_id, "command_start_language_learning_contextual_convo"): # check if we should start a conversation, updates are made by Mira as well using the CallCustomFunctionTool
                    print("FORCE CONTEXTUAL CONVO FOR USER: ", user_id)
                    tasks.append(handle_user_conversation(user_id, device_id, db_handler, force_conversation=True)) # force the conversation to start because this command was given by the user             
                else:
                    # print("MAYBE STARTING CONTEXTUAL CONVO FOR USER: ", user_id)
                    tasks.append(handle_user_conversation(user_id, device_id, db_handler))

            if tasks:
                await asyncio.gather(*tasks)

        except Exception as e:
            print("Exception in CONTEXTUAL CONVO loop...:")
            print(e)
            traceback.print_exc()

        finally:
            pLoopEndTime = time.time()
            # print(f"=== CONTEXTUAL CONVO loop completed in {round(pLoopEndTime - pLoopStartTime, 2)} seconds overall ===")

        await asyncio.sleep(run_period)


def ll_context_convo_agent_processing_loop():
    asyncio.run(ll_context_convo_agent_processing_loop_async())
