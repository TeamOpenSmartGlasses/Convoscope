from langchain.agents import Tool
from agents.Tools.CallCustomFunctionTool import CallCustomFunctionTool
from agents.call_custom_function_tool_config import get_call_custom_function_tool_config_as_string


call_custom_function_tool = None


def get_call_custom_function_tool_for_explicit_agent(user_id: str) -> Tool:

    global call_custom_function_tool

    call_custom_function_tool = CallCustomFunctionTool(user_id)

    call_custom_function_tool_for_explicit_agent = Tool(
        name="Call_Custom_Function",
        func=call_custom_function_tool.query,
        coroutine=call_custom_function_tool.a_query,
        description=f"""
        This tool is designed to call specific functions based on user queries. 
        It recognizes certain commands within a conversation and triggers the appropriate functions accordingly.

        Each function has the following entries:
        - Command: The command that triggers the function.
        - Successful Response: The response that confirms the function has been successfully executed.
        - Example Usage: Examples of how to use the command called a conversation.

        {get_call_custom_function_tool_config_as_string()}
        """
    )

    return call_custom_function_tool_for_explicit_agent
