from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio

from pydantic import BaseModel
from apps.AppConnection import *
from typing import List

import warnings

from apps.DataTypes import AugmentOsDataTypes
warnings.simplefilter(action='ignore', category=FutureWarning)

import asyncio
import uuid
from pathlib import Path
import json
import time
import traceback
import pandas as pd
import multiprocessing
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

#Convoscope
from server_config import server_port
from constants import USE_GPU_FOR_INFERENCING, IMAGE_PATH
from DatabaseHandler import DatabaseHandler
from agents.proactive_agents_process import start_proactive_agents_processing_loop
from agents.expert_agent_configs import get_agent_by_name
from agents.explicit_agent_process import explicit_agent_processing_loop, call_explicit_agent
from agents.proactive_definer_agent_process import start_proactive_definer_processing_loop
from agents.language_learning_agent_process import start_language_learning_agent_processing_loop
from agents.ll_word_suggest_upgrade_agent_process import start_ll_word_suggest_upgrade_agent_processing_loop
from agents.ll_context_convo_agent_process import ll_context_convo_agent_processing_loop
from agents.adhd_stmb_agent_process import adhd_stmb_agent_processing_loop
from auth.authentication import verify_id_token


app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
db_handler = DatabaseHandler()
agent_executor = ThreadPoolExecutor()

# Initialize the ConnectionManager
connection_manager = ConnectionManager()

class RegisterAppModel(BaseModel):
    app_id: str
    app_name: str
    app_description: str
    app_webhook_url: str
    websocket_uri: str
    subscriptions: List[str]

@app.post("/chat")
async def chat_handler(request: Request):
    body = await request.json()
    is_final = body.get('isFinal')
    text = body.get('text')
    device_id = body.get('deviceId')
    transcribe_language = body.get('transcribe_language')
    id_token = body.get('Authorization')
    user_id = verify_id_token(id_token)

    if transcribe_language is None or "":
        transcribe_language = "English"

    # 400 if missing params
    if text is None or text == '':
        print("Text none in chat_handler, exiting with error response 400.")
        return JSONResponse(content={'success': False, 'message': 'no text in request'}, status_code=400)
    if user_id is None or user_id == '':
        print("user_id none in chat_handler, exiting with error response 400.")
        return JSONResponse(content={'success': False, 'message': 'unauthorized'}, status_code=400)

    transcript = db_handler.save_transcript_for_user(user_id=user_id, device_id=device_id, text=text, is_final=is_final, transcribe_language=transcribe_language)
    message = "Sending messages too fast" if not transcript else ""

    await connection_manager.smart_broadcast_data(user_id, AugmentOsDataTypes.TRANSCRIPT, transcript)

    return JSONResponse(content={'success': True, 'message': message}, status_code=200)

@app.post("/set_user_settings")
async def set_user_settings(request: Request):
    body = await request.json()
    target_language = body.get('target_language')
    id_token = body.get('Authorization')
    user_id = verify_id_token(id_token)
    if user_id is None:
        return JSONResponse(content={'success': False, 'message': 'unauthorized'}, status_code=400)
    
    db_handler.update_user_settings(user_id, body)

    return JSONResponse(content={'success': True, 'message': "Saved your settings."}, status_code=200)

@app.post("/get_user_settings")
async def get_user_settings(request: Request):
    body = await request.json()
    id_token = body.get('Authorization')
    user_id = verify_id_token(id_token)
    if user_id is None:
        return JSONResponse(content={'success': False, 'message': 'unauthorized'}, status_code=400)
    
    user_settings = db_handler.get_user_settings(user_id)

    return JSONResponse(content={'success': True, 'settings': user_settings}, status_code=200)

@app.post("/ui_poll")
async def ui_poll_handler(request: Request):
    # parse request
    body = await request.json()
    # print(body)
    device_id = body.get('deviceId')
    id_token = body.get('Authorization')
    user_id = verify_id_token(id_token)
    
    if user_id is None:
        print("user_id is None in ui_poll_handler")
        return JSONResponse(content={'success': False, 'message': 'unauthorized'}, status_code=400)

    if device_id is None or device_id == '':
        return JSONResponse(content={'success': True, 'message': 'no device_id in request'}, status=400)

    resp = dict()
    resp["success"] = True
    
    db_handler.update_active_user(user_id, device_id)

    # add display_requests
    resp["display_requests"] = db_handler.get_display_requests_for_user_device(
        user_id=user_id, device_id=device_id)

    # explicit agent - user queries and agent responses
    explicit_insight_queries = db_handler.get_explicit_query_history_for_user(user_id=user_id, device_id=device_id)
    explicit_insight_results = db_handler.get_explicit_insights_history_for_user(user_id=user_id, device_id=device_id)
    wake_word_time = db_handler.get_wake_word_time_for_user(user_id=user_id)
    resp["explicit_insight_queries"] = explicit_insight_queries
    resp["explicit_insight_results"] = explicit_insight_results
    resp["wake_word_time"] = wake_word_time

    #proactive agents
    # Ignore proactive agents if we've had a query in past 15 seconds
    if not explicit_insight_queries or (time.time() - explicit_insight_queries[-1]['timestamp'] < (1000 * 15)):
        agent_insight_results = db_handler.get_proactive_agents_insights_results_for_user_device(user_id=user_id, device_id=device_id)
        resp["results_proactive_agent_insights"] = agent_insight_results

    # intelligent definer
    resp["entity_definitions"] = db_handler.get_agent_proactive_definer_results_for_user_device(user_id=user_id, device_id=device_id)

    # language learning rare word translation
    resp["language_learning_results"] = db_handler.get_language_learning_results_for_user_device(user_id=user_id, device_id=device_id)

    # language learning contextual convos
    resp["ll_context_convo_results"] = db_handler.get_ll_context_convo_results_for_user_device(user_id=user_id, device_id=device_id)

    # ADHD short term memory buffer
    resp["adhd_stmb_agent_results"] = db_handler.get_adhd_stmb_results_for_user_device(user_id=user_id, device_id=device_id)

    #language learning word suggestion upgrade
    resp["ll_word_suggest_upgrade_results"] = db_handler.get_ll_word_suggest_upgrade_results_for_user_device(user_id=user_id, device_id=device_id)

    # tell the frontend to update their local settings if needed
    resp["should_update_settings"] = db_handler.get_should_update_settings(user_id)
    
    
    vocabulary_upgrade_enabled = body.get('vocabulary_upgrade_enabled')
    if vocabulary_upgrade_enabled is not None:
        db_handler.update_single_user_setting(user_id, "vocabulary_upgrade_enabled", vocabulary_upgrade_enabled)
      

    return JSONResponse(content=resp, status_code=200)



@app.post("/run_single_agent")
async def run_single_expert_agent_handler(request: Request):
    body = await request.json()
    timestamp = time.time() # Never use client's timestamp ### body.get('timestamp')
    id_token = body.get('Authorization')
    user_id = verify_id_token(id_token)
    agent_name = body.get('agentName')

    # 400 if missing params
    if timestamp is None or timestamp == '':
        print("Timestamp none in send_agent_chat, exiting with error response 400.")
        return JSONResponse(content={'success': False, 'message': 'no timestamp in request'}, status_code=400)
    if user_id is None or user_id == '':
        print("user_id none in send_agent_chat, exiting with error response 400.")
        return JSONResponse(content={'success': False, 'message': 'unauthorized'}, status_code=400)
    
    print("Got single agent request for agent: {}".format(agent_name))

    asyncio.ensure_future(expert_agent_runner(agent_name, user_id))

    print("Spun up agent, now returning.")

    return JSONResponse(content={'success': True, 'message': f"Running agent: {agent_name}"}, status_code=200)

@app.post("/gps_location")
async def update_gps_location_for_user(request: Request):
    body = await request.json()
    id_token = body.get('Authorization')
    user_id = verify_id_token(id_token)
    device_id = body.get('deviceId')

    location = dict()
    location['lat'] = body.get('lat')
    location['lng'] = body.get('lng')

    # 400 if missing params
    if not user_id:
        print("user_id none in update_gps_location_for_user, exiting with error response 400.")
        return JSONResponse(content={'success': False, 'message': 'unauthorized'}, status_code=400)    
    if not location['lat'] or not location['lng']:
        print("location none in update_gps_location_for_user, exiting with error response 400.")
        return JSONResponse(content={'success': False, 'message': 'no location in request'}, status_code=400)

    # print("SEND UPDATE LOCATION FOR USER_ID: " + user_id)
    db_handler.add_gps_location_for_user(user_id, location)
    
    # locations = db_handler.get_gps_location_results_for_user_device(user_id, device_id)
    
    # print("locations: ", locations)
    # if len(locations) > 1:
    #     print("difference in locations: ", locations[-1]['lat'] - locations[-2]['lat'], locations[-1]['lng'] - locations[-2]['lng'])
    return JSONResponse(content={'success': True, 'message': f'Got your location: {location}'}, status_code=200)

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

### Following endpoints were unused, copy code from old server.py if needed ###

# @app.post("/upload_userdata")
# async def upload_user_data_handler(request: Request):
#     # Implement your stuff here
#     pass

# @app.post("/send_agent_chat")
# async def send_agent_chat_handler(request: Request):
#     # Implement your stuff here
#     pass

# @app.get("/image")
# async def return_image_handler(request: Request):
#     # Implement your stuff here
#     pass

# @app.post("/button_event")
# async def button_handler(request: Request):
#     # Implement your stuff here
#     pass

# @app.post("/rate_result")
# async def rate_result_handler(request: Request):
#     # Implement your stuff here
#     pass

@app.get("/protected")
async def protected_route(request: Request):
    # Implement your stuff here
    pass

# # WebSocket example for chat
# @app.websocket("/ws/chat")
# async def chat_websocket(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             # Implement your WebSocket stuff here
#             pass
#     except WebSocketDisconnect:
#         print("Client disconnected")


# # WebSocket example for button event
# @app.websocket("/ws/button_event")
# async def button_event_websocket(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             # Implement your WebSocket stuff here
#             pass
#     except WebSocketDisconnect:
#         print("Client disconnected")


# I AM YOUR RUSSIAN ONLINE LONG DISTANCXE BOYFRIEND
# THIS IS NOT A ROMANCE SCAM
# SEND ME $4000 IN WALMART GIFT CARDS


@app.post("/register_app")
async def register_app(app_data: RegisterAppModel):
    """t
    Endpoint for TPAs to register themselves with AugmentOS.
    """
    ret = await connection_manager.register_tpa(app_data.app_id, app_data.app_name, app_data.app_description, app_data.app_webhook_url, app_data.websocket_uri, app_data.subscriptions)
    print (f"/Register_App: {ret}")
    return {"message": ret}

@app.websocket("/ws/tpa/{app_id}")
async def websocket_endpoint(websocket: WebSocket, app_id: str):
    """
    WebSocket endpoint for real-time communication with TPAs.
    """
    if not await connection_manager.connect(app_id, websocket):
        print("Woah there was an ERROR processing the incoming websocket")
        return
    
    try:
        while True:
            data = await websocket.receive_text()
            print(f"[server_fast.py] Received data from {app_id}...")
            
            try:
                # TODO: Do some authentication, verify app_id is allowed to send us messages

                data = json.loads(data)
                print(f"Data:\n{json.dumps(data, indent=4)}")
                # Now handle the parsed data based on its type
                await handle_tpa_message(data, app_id, websocket)
                
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
                await websocket.send_text("Invalid JSON format")
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for {app_id}")

async def handle_tpa_message(data: dict, app_id: str, websocket: WebSocket):
    """
    Handle the incoming JSON message from a TPA.
    """
    message_type = data.get("type", None)
    user_id = data.get('user_id', None)
    if not user_id:
        print("No user_id in tpa websocket message")
        return

    if message_type == AugmentOsDataTypes.DISPLAY_REQUEST:
        # Extract data specific to the display request
        display_data = data.get("data", {})
        db_handler.add_display_request_for_user(user_id, display_data)
    else:
        print(f"Unknown message type: {message_type}")
        await websocket.send_text("Unknown message type")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Close all WebSocket connections gracefully on server shutdown.
    """
    await connection_manager.close_all_connections()

# Example of how you might send data to all registered TPAs
@app.get("/broadcast/{message}")
async def broadcast_message(message: str):
    await connection_manager.broadcast_data(message)
    return {"message": f"Broadcasted message: {message}"}

# Background process initialization for agents
def start_background_processes():
    # if USE_GPU_FOR_INFERENCING:
    #     multiprocessing.set_start_method('spawn')

    # print("Starting Proactive Agents process...")
    # proactive_agents_background_process = multiprocessing.Process(target=start_proactive_agents_processing_loop)
    # proactive_agents_background_process.start()

    print("Starting Explicit Agent process...")
    explicit_background_process = multiprocessing.Process(target=explicit_agent_processing_loop)
    explicit_background_process.start()

    # print("Starting Language Learning Agents process...")
    # language_learning_background_process = multiprocessing.Process(target=start_language_learning_agent_processing_loop)
    # language_learning_background_process.start()

    # print("Starting LL Word Suggest Upgrade Agents process...")
    # ll_word_suggest_upgrade_background_process = multiprocessing.Process(target=start_ll_word_suggest_upgrade_agent_processing_loop)
    # ll_word_suggest_upgrade_background_process.start()

    # print("Starting Contextual Convo Language Learning process...")
    # ll_context_convo_background_process = multiprocessing.Process(target=ll_context_convo_agent_processing_loop)
    # ll_context_convo_background_process.start()

    # print("Starting ADHD STMB process...")
    # adhd_stmb_background_process = multiprocessing.Process(target=adhd_stmb_agent_processing_loop)
    # adhd_stmb_background_process.start()

    # Wait for processes to finish
    # proactive_agents_background_process.join()
    explicit_background_process.join()
    # # language_learning_background_process.join()
    # ll_word_suggest_upgrade_background_process.join()
    # ll_context_convo_background_process.join()
    # adhd_stmb_background_process.join()

if __name__ == '__main__':
    print("Starting server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=server_port, log_level="info")
    start_background_processes()