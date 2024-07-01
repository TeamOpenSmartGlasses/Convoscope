from typing import Optional
import warnings
import asyncio

from agents.call_custom_function_tool_config import get_call_custom_function_tool_config
from DatabaseHandler import DatabaseHandler


class CallCustomFunctionTool:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.db_handler = DatabaseHandler(parent_handler=False)
        self.call_custom_function_tool_config = get_call_custom_function_tool_config()

    def query(self, query: str) -> Optional[str]:
        pass

    async def a_query(self, query: str) -> Optional[str]:
        if query is None:
            warnings.warn("Mira made a query with None value.")
            return "Query is None. Please provide a valid query."

        response = None

        print("####### MIRA MADE A QUERY #######")
        match query:
            case "ll_cc_on": # TODO: change this
                print("ll_cc_on")
                self.db_handler.update_single_user_setting(self.user_id, "command_start_language_learning_contextual_convo", True) # change this variable to start the conversation
                response = self.call_custom_function_tool_config["Start_Contextual_Conversation"]["successful_response"] # TODO: change this before merging
            case "ll_cc_off":
                print("ll_cc_off")
                print(self.user_id)
                self.db_handler.update_single_user_setting(self.user_id, "is_having_language_learning_contextual_convo", False) # change this variable to end the conversation
                response = self.call_custom_function_tool_config["Stop_Contextual_Conversation"]["successful_response"]
            case _:
                warnings.warn(f"Mira made an unsupported query: {query}")
                response = f"Unsupported query: {query}, allowed queries are: ll_cc_on, ll_cc_off"

        # print("@@@@@", self.db_handler.get_gps_location_results_for_user_device(self.user_id, None))

        return response