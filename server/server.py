import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from aiohttp import web
import asyncio
import math
import uuid
from aiohttp.web_response import Response
from pathlib import Path
import json
import time
import traceback
import pandas as pd

# multiprocessing
import multiprocessing
import logging
import logging.handlers
from concurrent.futures import ThreadPoolExecutor

# CORS
import aiohttp_cors
from aiohttp import web, web_exceptions

#Convoscope
from server_config import server_port
from constants import USE_GPU_FOR_INFERENCING, IMAGE_PATH, TESTING_LL_CONTEXT_CONVO_AGENT
from ContextualSearchEngine import ContextualSearchEngine
from DatabaseHandler import DatabaseHandler
from agents.proactive_agents_process import proactive_agents_processing_loop
from agents.expert_agent_configs import get_agent_by_name
from agents.explicit_agent_process import explicit_agent_processing_loop, call_explicit_agent
from agents.proactive_definer_agent_process import proactive_definer_processing_loop
from agents.language_learning_agent_process import language_learning_agent_processing_loop
from agents.ll_context_convo_agent_process import ll_context_convo_agent_processing_loop
from agents.adhd_stmb_agent_process import adhd_stmb_agent_processing_loop
import agents.wake_words
from Modules.RelevanceFilter import RelevanceFilter

from auth.authentication import *


global agent_executor
global db_handler
global relevance_filter
global app


#handle new transcripts coming in
async def chat_handler(request):
    body = await request.json()
    is_final = body.get('isFinal')
    text = body.get('text')
    device_id = body.get('deviceId')
    transcribe_language = body.get('transcribe_language')
    timestamp = time.time() # Never use client's timestamp ### body.get('timestamp')
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()

    if transcribe_language is None or "":
        transcribe_language = "English"

    # 400 if missing params
    if text is None or text == '':
        print("Text none in chat_handler, exiting with error response 400.")
        return web.Response(text='no text in request', status=400)
    if user_id is None or user_id == '':
        print("user_id none in chat_handler, exiting with error response 400.")
        return web.Response(text='no userId in request', status=400)

    success = db_handler.save_transcript_for_user(user_id=user_id, device_id=device_id, text=text, is_final=is_final, transcribe_language=transcribe_language)
    message = "Sending messages too fast" if not success else ""

    return web.Response(text=json.dumps({'success': True, 'message': message}), status=200)


#handle new diarization data 
async def chat_diarization(request):
    body = await request.json()
    transcript_meta_data = body.get('transcript_meta_data')
    device_id = body.get('deviceId')
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()

    success = db_handler.save_deepgram_transcript_for_user(user_id=user_id, device_id=device_id, deepgram_obj=transcript_meta_data, transcribe_language="Unknown")

    return web.Response(text=json.dumps({'success': True, 'message': "good stuff yo"}), status=200)


async def start_recording_handler(request):
    body = await request.json()
    user_id = body.get('userId')
    
    if user_id is None or user_id == '':
        print("user_id none in chat_handler, exiting with error response 400.")
        return web.Response(text='no userId in request', status=400)
    
    result = db_handler.update_recording_time_for_user(user_id)
    
    return web.Response(text=json.dumps({'success': result}), status=200)


async def save_recording_handler(request):
    body = await request.json()
    user_id = body.get('userId')
    recording_name = body.get('recordingName')
    
    if user_id is None or user_id == '':
        print("user_id none in chat_handler, exiting with error response 400.")
        return web.Response(text='no userId in request', status=400)
    
    recording = db_handler.save_recording(user_id, recording_name)
    
    file_path = 'recordings/{}.json'.format(recording_name)
    with open(file_path, 'w') as file:
        json.dump(recording, file)

    return web.FileResponse(file_path, headers={
        'Content-Disposition': 'Attachment;filename={}.json'.format(recording_name)
    })
    # return web.Response(text=json.dumps({'success': True, 'message': recording}), status=200)


async def load_recording_handler(request):
    body = await request.json()
    recording_name = body.get('recordingName')
    
    if recording_name is None or recording_name == '':
        print("recording_name none in chat_handler, exiting with error response 400.")
        return web.Response(text='no recordingName in request', status=400)
    
    file_path = 'recordings/{}.json'.format(recording_name)
    return web.FileResponse(file_path, headers={
        'Content-Disposition': 'Attachment;filename={}.json'.format(recording_name)
    })
    # return web.Response(text=json.dumps({'success': True, 'mess


async def set_user_settings(request):
    body = await request.json()
    target_language = body.get('target_language')
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()
    
    db_handler.update_user_settings(user_id, body)

    return web.Response(text=json.dumps({'success': True, 'message': "Saved your settings."}))


async def get_user_settings(request):
    body = await request.json()
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()
    
    user_settings = db_handler.get_user_settings(user_id)

    return web.Response(text=json.dumps({'success': True, 'settings': user_settings}))


# runs when button is pressed on frontend - right now button ring on wearable or button in TPA
async def button_handler(request):
    body = await request.json()
    button_num = body.get('buttonNum')
    button_activity = body.get('buttonActivity')
    timestamp = body.get('timestamp')
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()
    print('\n=== New Request ===\n', button_num,
          button_activity, timestamp, user_id)

    # 400 if missing params
    if button_num is None or button_num == '':
        return web.Response(text='no buttonNum in request', status=400)
    if button_activity is None or button_activity == '':
        return web.Response(text='no buttonActivity in request', status=400)
    if timestamp is None or timestamp == '':
        return web.Response(text='no timestamp in request', status=400)
    if user_id is None or user_id == '':
        return web.Response(text='no userId in request', status=400)

    if button_activity:  # True if push down, false if button release
        print("button True")
        return web.Response(text=json.dumps({'message': "button up activity detected"}), status=200)
    else:
        return web.Response(text=json.dumps({'message': "button up activity detected"}), status=200)

async def ui_poll_handler(request, minutes=0.5):
    # parse request
    body = await request.json()
    # print(body)
    device_id = body.get('deviceId')
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()

    # 400 if missing params
    if device_id is None or device_id == '':
        return web.Response(text='no device_id in request', status=400)
#    if features is None or features == '':
#        return web.Response(text='no features in request', status=400)
    #if "contextual_search_engine" not in features:
    #    return web.Response(text='contextual_search_engine not in features', status=400)

    resp = dict()
    resp["success"] = True
    
    db_handler.update_active_user(user_id, device_id)

    # get CSE results
#    if "contextual_search_engine" in features:
#        cse_results = db_handler.get_cse_results_for_user_device(
#            user_id=user_id, device_id=device_id)
#
#        if cse_results:
#            print("server.py ================================= CSERESULT")
#            print(cse_results)
#
#        # add CSE response
#        resp["result"] = cse_results

    # get agent results
    #proactive agents
    agent_insight_results = db_handler.get_proactive_agents_insights_results_for_user_device(user_id=user_id, device_id=device_id)
    resp["results_proactive_agent_insights"] = agent_insight_results

    # explicit agent - user queries and agent responses
    explicit_insight_queries = db_handler.get_explicit_query_history_for_user(user_id=user_id, device_id=device_id)
    explicit_insight_results = db_handler.get_explicit_insights_history_for_user(user_id=user_id, device_id=device_id)
    wake_word_time = db_handler.get_wake_word_time_for_user(user_id=user_id)
    resp["explicit_insight_queries"] = explicit_insight_queries
    resp["explicit_insight_results"] = explicit_insight_results
    resp["wake_word_time"] = wake_word_time

    # intelligent definer
    entity_definitions = db_handler.get_agent_proactive_definer_results_for_user_device(user_id=user_id, device_id=device_id)
    resp["entity_definitions"] = entity_definitions

    # language learning rare word translation
    language_learning_results = db_handler.get_language_learning_results_for_user_device(user_id=user_id, device_id=device_id)
    resp["language_learning_results"] = language_learning_results
    
    # language learning contextual convos
    ll_context_convo_results = db_handler.get_ll_context_convo_results_for_user_device(user_id=user_id, device_id=device_id)
    resp["ll_context_convo_results"] = ll_context_convo_results

    # ADHD short term memory buffer
    adhd_stmb_agent_results = db_handler.get_adhd_stmb_results_for_user_device(user_id=user_id, device_id=device_id)
    resp["adhd_stmb_agent_results"] = adhd_stmb_agent_results

    # tell the frontend to update their local settings if needed
    should_update_settings = db_handler.get_should_update_settings(user_id)
    resp["should_update_settings"] = should_update_settings

    return web.Response(text=json.dumps(resp), status=200)


#return images that we generated and gave frontends a URL for
async def return_image_handler(request):
    requested_img = request.rel_url.query['img']
    img_path = Path(IMAGE_PATH).joinpath(requested_img)
    try:
        data = img_path.read_bytes()
    except:
        print("Error reading requested image: " + requested_img)
        data = Path('images/404-2.jpg').read_bytes()
    return Response(body=data, content_type="image/jpg")


# frontend can upload CSVs to run custom data search on
# DEV: we don't use this and it's not exposed on the frontend as it's currently broken and low priority
async def upload_user_data_handler(request):
    # Check file size before doing anything else
    try:
        post_data = await request.post()
    except web_exceptions.HTTPRequestEntityTooLarge:
        return web.Response(text="File too large. Max file size: {}MB".format(MAX_FILE_SIZE_MB), status=413)

    user_file = post_data.get('custom-file')
    id_token = post_data.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()

    if user_file and user_id:
        # Check if the file is a CSV file by looking at its content type
        if user_file.content_type != 'text/csv':
            return web.Response(text="Uploaded file is not a CSV", status=400)

        # Validate data
        try:
            df = pd.read_csv(user_file.file)
        except Exception:
            return web.Response(text="Could not read CSV", status=400)

        #if not cse.is_custom_data_valid(df):
            #return web.Response(text="Bad data format", status=400)
#
        #cse.upload_custom_user_data(user_id, df)

        return web.Response(text="Custom data uploaded successfully", status=200)
    else:
        return web.Response(text="Missing user file or user ID in the received data", status=400)


async def expert_agent_runner(expert_agent_name, user_id):
    print("Starting agent run task of agent {} for user {}".format(expert_agent_name, user_id))
    #get the context for the last n minutes
    n_seconds = 5*60
    convo_context = db_handler.get_transcripts_from_last_nseconds_for_user_as_string(user_id, n_seconds)

    #get the most recent insights for this user
    # insights_history = db_handler.get_agent_insights_history_for_user(user_id)
    insights_history = db_handler.get_recent_nminutes_agent_insights_history_for_user(user_id)
    insights_history = [insight["insight"] for insight in insights_history]

    #spin up the agent
    expert_agent = get_agent_by_name(expert_agent_name)
    if expert_agent:
        agent_insight = await expert_agent.run_agent_async(expert_agent_name, convo_context, insights_history)

    #save this insight to the DB for the user
    if agent_insight != None and agent_insight["agent_insight"] != None:
        db_handler.add_agent_insight_result_for_user(user_id, agent_insight["agent_name"], agent_insight["agent_insight"], agent_insight["reference_url"])

    #agent run complete
    print("--- Done agent run task of agent {} from user {}".format(expert_agent_name, user_id))


#run a single agent with no extra context
async def run_single_expert_agent_handler(request):
    body = await request.json()
    timestamp = time.time() # Never use client's timestamp ### body.get('timestamp')
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()
    agent_name = body.get('agentName')

    # 400 if missing params
    if timestamp is None or timestamp == '':
        print("Timestamp none in send_agent_chat, exiting with error response 400.")
        return web.Response(text='no timestamp in request', status=400)
    if user_id is None or user_id == '':
        print("user_id none in send_agent_chat, exiting with error response 400.")
        return web.Response(text='no userId in request', status=400)

    print("Got single agent request for agent: {}".format(agent_name))

    #spin up agent
    asyncio.ensure_future(expert_agent_runner(agent_name, user_id))
    #loop = asyncio.get_event_loop()
    #loop.create_task(expert_agent_runner(agent_name, user_id))  # Non-blocking
    #loop.run_in_executor(executor, run_single_agent)

    print("Spun up agent, now returning.")

    return web.Response(text=json.dumps({'success': True, 'message': "Running agent: {}".format(agent_name)}), status=200)


#receive a chat message manually typed in the agent chat box
async def send_agent_chat_handler(request):
    body = await request.json()
    timestamp = time.time() # Never use client's timestamp ### body.get('timestamp')
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()
    chat_message = body.get('chatMessage')

    # 400 if missing params
    if timestamp is None or timestamp == '':
        print("Timestamp none in send_agent_chat, exiting with error response 400.")
        return web.Response(text='no timestamp in request', status=400)
    if user_id is None or user_id == '':
        print("user_id none in send_agent_chat, exiting with error response 400.")
        return web.Response(text='no userId in request', status=400)
    if chat_message is None or chat_message == '':
        print("chatMessage none in send_agent_chat, exiting with error response 400.")
        return web.Response(text='no chatMessage in request', status=400)

    # skip into proc loop
    print("SEND AGENT CHAT FOR USER_ID: " + user_id)
    user = db_handler.get_user(user_id)
    await call_explicit_agent(user, chat_message)

    return web.Response(text=json.dumps({'success': True, 'message': "Got your message: {}".format(chat_message)}), status=200)


async def update_gps_location_for_user(request):
    # if TESTING_LL_CONTEXT_CONVO_AGENT:
    #     warnings.warn("TESTING MODE: Using hardcoded user_id, device_id and location. Please remove this warning when done testing.")
    #     user_id = "oO4QvMJELYM6jEYtLDbo1LRFLPO2"
    #     device_id = "android"
    #     location = {'lat': 53.411812, 'lng': -2.210799, 'timestamp': 1709593069, 'uuid': 'e7674554-2a89-44ac-900b-ae21ba817e74'}
    #     db_handler.add_gps_location_for_user(user_id, location)
    #     return web.Response(text=json.dumps({'success': True, 'message': "Got your location: {}".format(location)}), status=200)

    body = await request.json()
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    device_id = body.get('deviceId')

    # print("update_gps_location_for_user #################################")
    # print(user_id, device_id)
    if user_id is None:
        raise web.HTTPUnauthorized()

    location = dict()
    location['lat'] = body.get('lat')
    location['lng'] = body.get('lng')

    # 400 if missing params
    if not user_id:
        print("user_id none in update_gps_location_for_user, exiting with error response 400.")
        return web.Response(text='no userId in request', status=400)
    if not location['lat'] or not location['lng']:
        print("location none in update_gps_location_for_user, exiting with error response 400.")
        return web.Response(text='no chatMessage in request', status=400)

    # print("SEND UPDATE LOCATION FOR USER_ID: " + user_id)
    db_handler.add_gps_location_for_user(user_id, location)
    
    locations = db_handler.get_gps_location_results_for_user_device(user_id, device_id)
    
    # print("locations: ", locations)
    # if len(locations) > 1:
    #     print("difference in locations: ", locations[-1]['lat'] - locations[-2]['lat'], locations[-1]['lng'] - locations[-2]['lng'])

    return web.Response(text=json.dumps({'success': True, 'message': "Got your location: {}".format(location)}), status=200)


async def rate_result_handler(request):
    body = await request.json()
    id_token = body.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is None:
        raise web.HTTPUnauthorized()
    result_uuid = body.get('resultUuid')
    rating = body.get('rating')

     # 400 if missing params
    if user_id is None or user_id == '':
        print("user_id none in rate_result, exiting with error response 400.")
        return web.Response(text='no userId in request', status=400)
    if uuid is None or uuid == '':
        print("uuid none in rate_result, exiting with error response 400.")
        return web.Response(text='no uuid in request', status=400)
    if rating is None or rating == '':
        print("rating none in rate_result, exiting with error response 400.")
        return web.Response(text='no rating in request', status=400)

    res = db_handler.rate_result_by_uuid(user_id=user_id, result_uuid=result_uuid, rating=rating)
    return web.Response(text=json.dumps({'success': True, 'message': str(res)}), status=200)


async def protected_route(request):
    print("PROTECTED ROUTE")
    id_token = request.headers.get('Authorization')
    user_id = await verify_id_token(id_token)
    if user_id is not None:
        # Proceed with the user request
        print()
    else:
        raise web.HTTPUnauthorized()


if __name__ == '__main__':
    print("Starting server...")
    agent_executor = ThreadPoolExecutor()
    db_handler = DatabaseHandler()

    # start proccessing loop subprocess to process data as it comes in
    if USE_GPU_FOR_INFERENCING:
        multiprocessing.set_start_method('spawn')

    # log_queue = multiprocessing.Queue()
    #print("Starting CSE process...")
    #cse_process = multiprocessing.Process(target=cse_loop)
    #cse_process.start()

    # start intelligent definer agent process
    print("Starting Intelligent Definer Agent process...")
    intelligent_definer_agent_process = multiprocessing.Process(target=proactive_definer_processing_loop)
    intelligent_definer_agent_process.start()

    # start the proactive agents process
    print("Starting Proactive Agents process...")
    proactive_agents_background_process = multiprocessing.Process(target=proactive_agents_processing_loop)
    proactive_agents_background_process.start()

    # start the explicit agent process
    print("Starting Explicit Agent process...")
    explicit_background_process = multiprocessing.Process(target=explicit_agent_processing_loop)
    explicit_background_process.start()

    # start the language learning app process
    print("Starting Language Learning Agents process...")
    language_learning_background_process = multiprocessing.Process(target=language_learning_agent_processing_loop)
    language_learning_background_process.start()
    
    # start the contextual convo language larning app process
    #print("Starting Contextual Convo Language learning app process...")
    #ll_context_convo_background_process = multiprocessing.Process(target=ll_context_convo_agent_processing_loop)
    #ll_context_convo_background_process.start()

    # start the contextual convo language larning app process
    print("Starting ADHD STMB app process...")
    adhd_stmb_background_process = multiprocessing.Process(target=adhd_stmb_agent_processing_loop)
    adhd_stmb_background_process.start()

    # setup and run web app
    # CORS allow from all sources
    print("Starting aiohttp server...")
    MAX_FILE_SIZE_MB = 88
    app = web.Application(client_max_size=(1024*1024*MAX_FILE_SIZE_MB))
    app.add_routes(
        [
            web.get('/protected', protected_route),
            web.get('/image', return_image_handler),
            web.post('/chat', chat_handler),
            web.post('/chat_diarization', chat_diarization),
            web.post('/button_event', button_handler),
            web.post('/ui_poll', ui_poll_handler),
            web.post('/upload_userdata', upload_user_data_handler),
            web.post('/run_single_agent', run_single_expert_agent_handler),
            web.post('/send_agent_chat', send_agent_chat_handler),
            web.post('/rate_result', rate_result_handler),
            web.post('/start_recording', start_recording_handler),
            web.post('/save_recording', save_recording_handler),
            web.post('/load_recording', load_recording_handler),
            web.post('/set_user_settings', set_user_settings),
            web.post('/get_user_settings', get_user_settings),
            web.post('/gps_location', update_gps_location_for_user),
        ]
    )
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*"
        )
    })
    for route in list(app.router.routes()):
        cors.add(route)
    print("Running web server...")
    web.run_app(app, port=server_port)

    #let processes finish and join
    proactive_agents_background_process.join()
    intelligent_definer_agent_process.join()
    #cse_process.join()
    language_learning_background_process.join()
    #ll_context_convo_background_process.join()
    explicit_background_process.join()
    adhd_stmb_background_process.join()
