import json
from typing import List
import requests
import asyncio
import websockets
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from threading import Thread
import uvicorn
import aiohttp
from augmentos_client.DataStorage import DataStorage
from augmentos_client.DataTypes import AugmentOsDataTypes
from tenacity import retry, stop_after_attempt, wait_fixed

class TPAClient:
    def __init__(self, app_id, app_name, server_url, app_description="", subscriptions=None):
        """
        Initialize the TPAClient with the necessary details.
        
        :param app_name: Unique name of the TPA.
        :param server_url: Base URL of the AugmentOS server.
        :param subscriptions: List of data types the TPA wants to subscribe to. Defaults to ["*"].
        """

        self.app_id = app_id
        self.app_name = app_name
        self.app_description = app_description
        self.server_url = server_url.rstrip('/')  # Ensure no trailing slash
        self.websocket_uri = f"ws://{self.server_url.split('//')[1]}/ws/tpa/{app_id}"  # Derive WebSocket URI
        self.subscriptions = subscriptions if subscriptions is not None else []
        self.websocket = None

        self.data_storage = DataStorage()

        self.on_other_received_callback = None
        self.on_transcript_received_callback = None
        self.on_location_received_callback = None
        self.on_camera_received_callback = None

        # Start the FastAPI app in a separate thread
        self.app = FastAPI()

        @self.app.post("/trigger-websocket")
        async def trigger_websocket(request: Request):
            """
            Handle incoming webhook from AugmentOS and initiate WebSocket connection.
            """
            # await self.initiate_websocket()
            await self.send_data({'type': 'pong'})

        @self.app.on_event("startup")
        async def startup_event():
            """
            On startup, register the TPA with AugmentOS and start WebSocket connection.
            """
            await self.register_with_augmentos()
            await self.initiate_websocket()

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """
            On shutdown, close the WebSocket connection.
            """
            await self.close_connection()

        @self.app.get("/")
        async def root():
            return {"message": "TPAClient is running"}

    def start(self):
        """
        Start the FastAPI application in a separate thread.
        """
        thread = Thread(target=uvicorn.run, args=(self.app,), kwargs={"host": "0.0.0.0", "port": 8000})
        thread.start()

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    async def register_with_augmentos(self):
        """
        Register the TPA with AugmentOS, including its subscription preferences.
        """
        registration_data = {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "app_description": self.app_description,
            "app_webhook_url": f"http://localhost:8000/trigger-websocket",  # Replace with actual public webhook URL
            "websocket_uri": self.websocket_uri,
            "subscriptions": self.subscriptions
        }

        async with aiohttp.ClientSession() as session:
            try:
                response = await session.post(f"{self.server_url}/register_app", json=registration_data)
                response_content = await response.read()
                print(f"[{self.app_name}] Registered with AugmentOS successfully with subscriptions: {self.subscriptions}:\n{response_content}")
            except requests.exceptions.RequestException as e:
                print(f"[{self.app_name}] Failed to register with AugmentOS: {e}")

    async def websocket_task(self):
        """
        Background task to maintain the WebSocket connection.
        """
        while True:
            try:
                await self.initiate_websocket()
                print(f"[{self.app_name}] WebSocket connection established.")
                await self.listen_for_data()  # This will run until the connection is closed
            except Exception as e:
                print(f"[{self.app_name}] WebSocket connection failed: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(5))
    async def initiate_websocket(self):
        """
        Initiate the WebSocket connection with AugmentOS if it isn't already established.
        """
        if self.websocket is None or self.websocket.closed:
            print(f"[{self.app_name}] Starting a websocket on address {self.websocket_uri}")
            try:
                self.websocket = await websockets.connect(self.websocket_uri)
                print(f"[{self.app_name}] WebSocket connection established with AugmentOS.")
                asyncio.create_task(self.listen_for_data())
            except Exception as e:
                print(f"[{self.app_name}] Failed to establish WebSocket connection: {e}")
                self.websocket = None

    async def listen_for_data(self):
        """
        Continuously listen for data from AugmentOS and trigger the callback when data is received.
        """
        while self.websocket:
            try:
                data = await self.websocket.recv()
                json_data = json.loads(data)
                print(f"[{self.app_name}] Data received from AugmentOS:\n{json.dumps(json_data, indent=4)}")
                await self.handle_augmentos_message(json_data)
            except Exception as e:
                print(f"[{self.app_name}] Error receiving data: {e}")
                self.websocket = None
                break

    async def handle_augmentos_message(self, data):
        message_type = data.get("type")
        user_id = data.get('user_id')
        
        self.data_storage.store_data(data)
        if message_type == AugmentOsDataTypes.TRANSCRIPT and self.on_transcript_received_callback:
            await self.on_transcript_received_callback(data)
        elif message_type == AugmentOsDataTypes.LOCATION and self.on_location_received_callback:
            await self.on_location_received_callback(data)
        elif message_type == AugmentOsDataTypes.CAMERA and self.data_storage.store_data(data):
            await self.on_camera_received_callback(data)
        elif message_type == AugmentOsDataTypes.OTHER and self.on_other_received_callback:
            await self.on_other_received_callback(data)

    async def send_data(self, data):
        """
        Send data to AugmentOS via WebSocket. Initiates the WebSocket connection if not already active.
        
        :param data: The data to send.
        """
        await self.initiate_websocket()
        
        if self.websocket:
            try:
                # Check if data is already a string, otherwise serialize it
                if not isinstance(data, str):
                    data = json.dumps(data)
                
                await self.websocket.send(data)
                print(f"[TPAClient - {self.app_name}] Data sent to AugmentOS: {data}")
            except Exception as e:
                print(f"[TPAClient - {self.app_name}] Failed to send data: {e}")
                self.websocket = None  # Reset the connection on failure

    async def send_text_wall(self, user_id, body):
        """
        Helper function to display things on the user's glasses. Currently only supports a basic textline. 
        """
        data = {
            "user_id": user_id,
            "type": AugmentOsDataTypes.DISPLAY_REQUEST,
            "data": {
                "layout": AugmentOsDataTypes.TEXT_WALL,
                "content": {
                    "body": body,
                }
            }
        }
        return await self.send_data(data) 
    
    async def send_reference_card(self, user_id, title, body, image_url = ""):
        data = {
            "user_id": user_id,
            "type": AugmentOsDataTypes.DISPLAY_REQUEST,
            "data": {
                "layout": AugmentOsDataTypes.REFERENCE_CARD,
                "content": {
                    "title": title,
                    "body": body,
                    "image_url": image_url
                } 
            }
        }
        return await self.send_data(data)
    
    async def send_rows_card(self, user_id, rows: List[str]):
        data = {
            "user_id": user_id,
            "type": AugmentOsDataTypes.DISPLAY_REQUEST,
            "data": {
                "layout": AugmentOsDataTypes.ROWS_CARD,
                "content": {
                    "rows": rows
                }
            }
        }
        return await self.send_data(data)
    
    async def send_text_line(self, user_id, body):
        """
        Helper function to display things on the user's glasses. Currently only supports a basic textline. 
        """
        data = {
            "user_id": user_id,
            "type": AugmentOsDataTypes.DISPLAY_REQUEST,
            "data": {
                "layout": AugmentOsDataTypes.TEXT_LINE,
                "content": {
                    "body": body
                }
            }
        }
        return await self.send_data(data) 

    async def close_connection(self):
        """
        Close the WebSocket connection gracefully.
        """
        if self.websocket:
            try:
                await self.websocket.close()
                print(f"[{self.app_name}] WebSocket connection closed with AugmentOS.")
            except Exception as e:
                print(f"[{self.app_name}] Error closing WebSocket: {e}")
            finally:
                self.websocket = None

    def on_other_received(self, callback):
        self.on_other_received_callback = callback

    def on_transcript_received(self, callback):
        self.on_transcript_received_callback = callback

    def on_location_received(self, callback):
        self.on_location_received_callback = callback

    def on_camera_received(self, callback):
        self.on_camera_received_callback = callback