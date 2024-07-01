call_custom_function_tool_config = {
    "Start_Contextual_Conversation": {
        "function_name": "Start language learning convo.",
        "function_command": """ll_cc_on""",
        "successful_response": """Contextual conversation has been started.""",
        "example_usage": """
        Conversation: "I want to start a conversation."
        Action: "ll_cc_on"
        Output: "Final Answer: Running: Start language learning convo."

        Conversation: "Begin contextual mode."
        Action: "ll_cc_on"
        Output: "Final Answer: Running: Start language learning convo."
        """,
    },
    "Stop_Contextual_Conversation": {
        "function_name": "Stop language learning convo.",
        "function_command": """ll_cc_off""",
        "successful_response": """Contextual conversation has been ended.""",
        "example_usage": """
        Conversation: "Please stop the language learning conversation."
        Action: "ll_cc_off"
        Output: "Final Answer: Running: Stop language learning convo."
        
        Conversation: "End this convo."
        Action: "ll_cc_off"
        Output: "Final Answer: Running: Stop language learning convo."
        
        Conversation: "Finish this language exchange"
        Action: "ll_cc_off"
        Output: "Final Answer: Running: Stop language learning convo."
        """,
    },
}


def get_call_custom_function_tool_config():
    return {function_name: call_custom_function_tool_config[function_name] for function_name in call_custom_function_tool_config.keys()}


def get_call_custom_function_tool_config_as_string():
    config = ""
    
    for function_name in call_custom_function_tool_config.keys():
        config += f"\n{function_name}:\n"
        config += f"""   - Command: {call_custom_function_tool_config[function_name]["function_command"]}\n"""
        config += f"""   - Successful Output: {call_custom_function_tool_config[function_name]["successful_response"]}\n"""
        config += f"""   - Example Usage: {call_custom_function_tool_config[function_name]["example_usage"]}\n"""
    
    return config
