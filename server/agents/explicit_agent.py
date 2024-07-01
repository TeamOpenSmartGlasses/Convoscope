if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from agents.search_tool_for_agents import get_search_tool_for_agents
from agents.expert_agent_configs import default_expert_agent_list
from langchain.agents import initialize_agent
from langchain.agents import AgentType
import asyncio
from Modules.LangchainSetup import *
from helpers.time_function_decorator import time_function
from agents.call_custom_function_tool_for_explicit_agent import get_call_custom_function_tool_for_explicit_agent
from agents.call_custom_function_tool_config import get_call_custom_function_tool_config
import langchain


# langchain.debug = True

# llm = get_langchain_gpt4()
llm = get_langchain_gpt4o(max_tokens=350)
# llm = get_langchain_gpt35(max_tokens=1024)

call_custom_function_tool_config = get_call_custom_function_tool_config()

custom_commands = "\n".join([f"- {call_custom_function_tool_config[function_name]['function_name']}: {call_custom_function_tool_config[function_name]['function_command']}" for function_name in call_custom_function_tool_config])
successful_responses_for_custom_commands = "\n".join([f"- {call_custom_function_tool_config[function_name]['function_name']}: {call_custom_function_tool_config[function_name]['successful_response']}" for function_name in call_custom_function_tool_config])

# explictly respond to user queries
explicit_agent_prompt_blueprint = """[Conversation Transcript]
This is the transcript of the current live conversation:
<transcript history>{transcript_history}</transcript history>

Use the above transcript as context and a data source to help the user.

[Your Tools]
Tools can be used at your discretion if they are absolutely necessary to answer the query - try to answer questions without using Tools if you can. If a query asks for something you already know, don't search for it, just provide it from memory. But if you don't know the answer, use your SearchEngine. If the user asks for a specific command to be executed, use the Call_Custom_Function tool.

These are your tools: Search_Engine, Call_Custom_Function

[Custom Commands]
Mira can perform specific custom commands beyond just answering queries. If a user query matches one of the predefined requests, Mira should call a custom function using the `Call_Custom_Function` tool. The custom function tool takes an argument specifying which function to call. When Mira calls a custom function, her response should be “Executing Command: <user facing custom function string>”.

Custom Commands include:
{custom_commands}

Successful responses:
{successful_responses_for_custom_commands}

As soon as you receive a successful response, you must return 'Final Answer: Running: <user facing custom function string>'


YOU MUST IGNORE THE TRANSCRIPT HISTORY WHEN EXECUTING COMMANDS. ONLY USE THE USER QUERY TO DETERMINE WHICH COMMAND TO EXECUTE.

[Your History]
This is the history of your previous inputs + responses:
<history>{insight_history}</history>

[Actions]
Actions are only for using tools.

[Your Task]
Answer the User Query to the best of your ability. The query may contain some extra unrelated speech not related to the query - ignore any noise to answer just the user's intended query. Make your answer concise, leave out filler words, make the answer high entropy, answer in 10 words or less (no newlines). Use telegraph style writing.

The Search_Engine is not for personal queries! It will only search the web, don't use it if the user asks a query about their personal data or conversation, only use it to retrieve information from the web.


[User Query]
Query: ```{query}```

NOTE: Only use the Search_Engine to search for public information, never use Search_Engine to answer questions about the current conversation. If the user asks you a question about something that was just said, or to summarize the conversation, use the Conversation Transcript and History to do so.
NOTE: Make sure your Final Answer is always returned in the specified format (must have prefix "Final Answer: ")! Never return just a string, return a Final Answer formatted as directed.
NOTE: Commands must be called ONLY IF the user explicitly asks for them in the query. Do not call commands based on tranascript history.
NOTE: If the query is empty, return "Final Answer: No query provided."
""" # TODO: commands should come from json


# Example:
# - If the user says "Start the LLCC conversation," Mira should respond with "Executing Command: Starting Contextual Conversation" and then execute the "ll_cc_on" command using Call_Custom_Function.

# [Custom Function Definitions]
# 1. Example Queries:
# - Start the contextual conversation
# - Turn on the LLCC conversation
# - Begin contextual mode
# 2. Function Name: ll_cc_on
# 3. Response from the Tool: Starting Contextual Conversation

# 4. Thought: The contextual conversation has been started.
# Final Answer: Executing Command: Starting Contextual Conversation



# 1. Example Queries:
# - Stop the contextual conversation
# - Turn off the LLCC conversation
# - End contextual mode
# 2. Function Name: ll_cc_off
# 3. Response from the Tool: Ending Contextual Conversation

# 4. Thought: The contextual conversation has been ended.

# Final Answer: Executing Command: Ending Contextual Conversation


# How to use the Expert Agents: they are like workers in your team that can help you do certain tasks. Imagine you are a human manager and your agents as human workers. You can assign tasks to your agents and they will help you complete the tasks. Speak to them like how you would speak to a human worker, give detailed context and instructions.

def get_expert_agents_name_list():
    return [ea.agent_name for ea in default_expert_agent_list]

# generate expert agents as tools (each one has a search engine, later make the tools each agent has programmatic)
# @time_function()
def make_expert_agents_as_tools(transcript):
    return [ea.get_agent_as_tool(transcript) for ea in default_expert_agent_list]

# @time_function()
def get_explicit_agent(transcript, user_id):
    #hold all the tools of the explicit agent
    tools = []

    # get all the expert agents as tools
    expert_agents_as_tools = make_expert_agents_as_tools(transcript)
    tools.extend(expert_agents_as_tools)

    print(transcript)
    # also add normal tools to explicit agent toolset
    tools.append(get_search_tool_for_agents())
    tools.append(get_call_custom_function_tool_for_explicit_agent(user_id=user_id))

    explicit_agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
            max_iterations=4, 
            # handle_parsing_errors="That output triggered an error, check your output and make sure it conforms, use the Action/Action Input syntax",
            handle_parsing_errors=True,
            verbose=True)
    return explicit_agent

# @time_function()
async def run_explicit_agent_async(user_id, query, transcript_history = "", insight_history = ""):
    expert_agents_name_list = get_expert_agents_name_list()
    explicit_agent_prompt = explicit_agent_prompt_blueprint.format(
        insight_history=insight_history,
        query=query,
        custom_commands=custom_commands,
        successful_responses_for_custom_commands=successful_responses_for_custom_commands,
        transcript_history=transcript_history,
        expert_agents_name_list=expert_agents_name_list
    )
    # print("#### EXPLICIT AGENT PROMPT ####")
    # print(explicit_agent_prompt)
    transcript = "{}\nQuery: {}".format(insight_history, query)
    insight = await (get_explicit_agent(transcript, user_id).ainvoke(explicit_agent_prompt))
    return insight['output']


if __name__ == '__main__':
    # result = asyncio.run(run_explicit_agent_async("THIS IS THE CONTEXT", "FACT CHECK THAT CARS HAVE 4 WHEELS?"))
    # print ("RESULT: " + result)
    result = asyncio.run(run_explicit_agent_async("1234", "Start Contextual convo", "START CONTEXTUAL CONVERSATION NOW, START CONTEXTUAL CONVERSATION, START CONTEXTUAL CONVERSATION"))
    print ("RESULT: " + result)
