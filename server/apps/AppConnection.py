import asyncio
import json
import requests
from typing import Dict, List

class AppConnection:
    def __init__(self, app_id: str, app_name: str, app_description: str, app_webhook_url: str, websocket_uri: str, subscriptions: List[str]):
        """
        Initialize the AppConnection with the necessary details.
        
        :param app_id: Unique id of the TPA.
        :param app_name: Unique name of the TPA.
        :param app_description: Description of the TPA.
        :param app_webhook_url: URL to send webhooks to the TPA.
        :param websocket_uri: WebSocket URI for the TPA.
        """

        self.app_id = app_id
        self.app_name = app_name
        self.app_description = app_description
        self.app_webhook_url = app_webhook_url
        self.websocket_uri = websocket_uri
        self.subscriptions = subscriptions
        self.websocket = None
        self.lock = asyncio.Lock()  # To prevent concurrent connection attempts

    async def send_webhook(self):
        """
        Send a webhook to the TPA to trigger WebSocket connection.
        """
        # await asyncio.sleep(4)
        try:
            print(f"[AppConnection - {self.app_name}] CONNECTING TO APP URL: {self.app_webhook_url}")
            response = requests.post(self.app_webhook_url, json={"message": "initiate_websocket"})
            response.raise_for_status()
            print(f"[AppConnection - {self.app_name}] Webhook sent successfully.")
        except requests.exceptions.RequestException as e:
            print(f"[AppConnection - {self.app_name}] Failed to send webhook: {e}")
            raise e

    async def ensure_connection(self):
        """
        Ensure that the WebSocket connection is active. If not, initiate it.
        """
        async with self.lock:
            if self.websocket is None :
                print(f"[AppConnection - {self.app_name}] No active WebSocket connection. Initiating connection.")
                await self.send_webhook()
                await self.wait_for_connection()

    async def wait_for_connection(self):
        """
        Wait for the TPA to establish the WebSocket connection.
        """
        attempt = 0
        while (self.websocket is None) and attempt < 10:  # Retry a few times
            try:
                # self.websocket = await websockets.connect(self.websocket_uri)
                print(f"[AppConnection - {self.app_name}] WebSocket connection established.")
                return
            except Exception as e:
                print(f"[AppConnection - {self.app_name}] Waiting for TPA to establish WebSocket connection... attempt {attempt + 1}")
                attempt += 1
                await asyncio.sleep(2)  # Wait a bit before retrying

        if self.websocket is None:
            raise ConnectionError(f"[{self.app_name}] Failed to establish WebSocket connection after multiple attempts.")


    async def send_data(self, data):
        """
        Send data over the WebSocket connection. Initiates connection if not active.
        
        :param data: The data to send.
        """
        await self.ensure_connection()
        if self.websocket:
            try:
                if isinstance(data, str):
                    await self.websocket.send_text(data) 
                else:
                    await self.websocket.send_json(data) 
                print(f"[AppConnection - {self.app_name}] Data sent: {data}")
            except Exception as e:
                print(f"[AppConnection - {self.app_name}] Failed to send data: {e}")
                self.websocket = None  # Reset the connection on failure

    async def close_connection(self):
        """
        Close the WebSocket connection gracefully.
        """
        if self.websocket:
            try:
                await self.websocket.close()
                print(f"[AppConnection - {self.app_name}] WebSocket connection closed.")
            except Exception as e:
                print(f"[AppConnection - {self.app_name}] Error closing WebSocket: {e}")
            finally:
                self.websocket = None


class ConnectionManager:
    def __init__(self):
        """
        Initialize the ConnectionManager with an empty registry.
        """
        self.registry: Dict[str, AppConnection] = {}
        self.lock = asyncio.Lock()  # To ensure thread-safe operations on the registry

    async def register_tpa(self, app_id: str, app_name: str, app_description: str, app_webhook_url: str, websocket_uri: str, subscriptions: List[str]):
        """
        Register a new TPA with AugmentOS.
        
        :param app_name: Unique id of the TPA.
        :param app_name: Unique name of the TPA.
        :param app_webhook_url: URL to send webhooks to the TPA.
        :param websocket_uri: WebSocket URI for the TPA.
        """
        async with self.lock:
            if app_id in self.registry:
                return f"[ConnectionManager] TPA '{app_id}' is already registered."
            app_connection = AppConnection(app_id, app_name, app_description, app_webhook_url, websocket_uri, subscriptions)
            self.registry[app_id] = app_connection
            # await app_connection.send_data("[From Server] Hello client!")
            return f"[ConnectionManager] TPA '{app_id}' registered successfully."


    async def unregister_tpa(self, app_id: str):
        """
        Unregister a TPA from AugmentOS.
        
        :param app_name: Unique name of the TPA.
        """
        async with self.lock:
            app_connection = self.registry.pop(app_id, None)
            if app_connection:
                await app_connection.close_connection()
                print(f"[ConnectionManager] TPA '{app_id}' unregistered successfully.")
            else:
                print(f"[ConnectionManager] TPA '{app_id}' not found in registry.")

    async def send_data(self, app_id: str, data: str):
        """
        Send data to a registered TPA.
        
        :param app_name: Unique name of the TPA.
        :param data: The data to send.
        """
        async with self.lock:
            app_connection = self.registry.get(app_id)
            if not app_connection:
                print(f"[Manager] TPA '{app_id}' is not registered.")
                return

        await app_connection.send_data(data)

    async def broadcast_data(self, data: str):
        """
        Send data to all registered TPAs.
        
        :param data: The data to send.
        """
        async with self.lock:
            connections = list(self.registry.values())

        await asyncio.gather(*(conn.send_data(data) for conn in connections))

    async def close_all_connections(self):
        async with self.lock:
            connections = list(self.registry.values())

        await asyncio.gather(*(conn.close_connection() for conn in connections))

    async def connect(self, app_id, websocket):
        if not self.registry.get(app_id): 
            print("[ConnectionManager] Non-registered TPA tried to send a websocket")
            print("REGISTRY PRINT")
            print(json.dumps(self.registry))
            return False
        await websocket.accept()
        self.registry.get(app_id).websocket = websocket
        return True

    async def smart_broadcast_data(self, user_id, data_type, data):
        """
        Send data to all registered TPAs that subscribe to the specified data type.
        
        :param user_id: The ID of the user whose apps will receive the data.
        :param data_type: The type of data being sent.
        :param data: The actual data to send.
        """
        
        json = {
            "type": data_type,
            "user_id": user_id,
            "data": data,
        }

        # TODO: Users will have a list of apps. For now, just have a global list
        # user_apps = db_handler.get_user_app_ids(user_id)  # Get a list of app ids or names associated with the user
        async with self.lock:
            user_apps = list(self.registry.keys())  # List of all registered app_ids

        async with self.lock:
            # Loop through the user's apps and filter based on subscription
            for app_id in user_apps:
                app_connection = self.registry.get(app_id)

                if app_connection:
                    # Check if the app subscribes to this data type
                    if "*" in app_connection.subscriptions or data_type in app_connection.subscriptions:
                        print(f"[Smart Send] Sending data to {app_connection.app_name} for type {data_type}")
                        await app_connection.send_data(json)
                    else:
                        print(f"[Smart Send] {app_connection.app_name} does not subscribe to {data_type}")
                else:
                    print(f"[Smart Send] App {app_id} is not registered.")


# Example Usage
async def main():
    manager = ConnectionManager()

    # Register TPAs
    await manager.register_tpa(
        app_id="com.exampletpa.1",
        app_name="TPA1",
        app_webhook_url="http://localhost:8001/trigger-websocket",
        websocket_uri="ws://localhost:8001/ws",
        subscriptions=["*"]
    )
    await manager.register_tpa(
        app_id="com.exampletpa.2",
        app_name="TPA2",
        app_webhook_url="http://localhost:8002/trigger-websocket",
        websocket_uri="ws://localhost:8002/ws",
        subscriptions=["*"]
    )

    # Send data to a specific TPA
    # await manager.send_data("TPA1", "Hello TPA1!")

    # # Broadcast data to all TPAs
    # await manager.broadcast_data("Hello All TPAs!")

    # # Unregister a TPA
    # await manager.unregister_tpa("TPA2")

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
