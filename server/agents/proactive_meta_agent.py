#custom
from collections import defaultdict
from agents.expert_agent_configs import default_expert_agent_list
from agents.agent_utils import format_list_data
from server_config import openai_api_key
from constants import DEBUG_FORCE_EXPERT_AGENT_RUN

#langchain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field
from helpers.time_function_decorator import time_function
import asyncio

from Modules.LangchainSetup import *

force_run_agents_prompt = ""
if DEBUG_FORCE_EXPERT_AGENT_RUN:
    force_run_agents_prompt = "For this run, you MUST specify at least 1 expert agent to run. Do not output an empty list."

#proactively decides which agents to run and runs them
proactive_meta_agent_prompt_blueprint = """You are the higly intelligent and skilled proactive master agent of "Convoscope". "Convoscope" is a tool that listens to a user's live conversation and enhances their conversation by providing them with real time "Insights". The "Insights" generated should aim to lead the user to deeper understanding, broader perspectives, new ideas, more accurate information, better replies, and enhanced conversations.

[Your Objective]
"Convoscope" is a multi-agent system in which you are the proactive meta agent. You will be given direct access to a live stream of transcripts from the user's conversation alongside information about a number of different 'expert agents` who have the power to generate "Insights". Your goal is to recognize when the thoughts or work of an 'expert agent' would be useful to the conversation and to output a list of which agents should be run. It's OK to output an empty list if no agents should be run right now. It's OK to specify multiple agents, but you should ussually just specify an empty list or only 1 agent.

{force_run_agents_prompt}

[Timing]
The longer it's been without any insights generated, the more likely it is that an insight would be useful and welcome. It's good to have at least 1 insight every few minutes. So if the last insight time was 12 seconds ago, it's very unlikely that we need another insight right now. But if the last insight was 2 minutes ago, it's very likely we want a new insight. If the last insight was over 4 minutes ago and there's new transcripts, you should almost definitly specify an expert agent to create an insight, because it's been so long.

[Your Agents]
You have access to "Expert Agents", which are like workers in your team that with special abilities. These are the agents you can run:
{expert_agents_descriptions_prompt}

[Conversation Transcript]
This is the current live transcript of the conversation you're assisting:
<Transcript start>{conversation_context}<Transcript end>

[Recent Insights History]
Here are the insights that have been generated recently, you should not call the expert agent if you think it will generate the same insight as one of these:
{insights_history}

<Task start>You should now output a list of the expert agents you think should run. Feel free to specify no expert agents. {format_instructions}<Task end>"""

# , but they're expensive to run, so only specify agents that will be helpful

#generate expert agents as tools (each one has a search engine, later make the tools each agent has programmatic)
def make_expert_agents_prompts():
    expert_agents_descriptions_prompt = str()

    for idx, expert_agent in enumerate(default_expert_agent_list):
        expert_agents_descriptions_prompt += expert_agent.get_agent_info_for_proactive_agent(idx+1)

    return expert_agents_descriptions_prompt


@time_function()
def run_proactive_meta_agent_and_experts(conversation_context: str, insights_history: list):
    #run proactive agent to find out which expert agents we should run
    proactive_meta_agent_response = run_proactive_meta_agent(conversation_context, insights_history)

    print("RUNNING THESE AGENTS")
    print(proactive_meta_agent_response)

    #do nothing else if proactive meta agent didn't specify an agent to run
    if proactive_meta_agent_response == []:
        return []

    #parse insights history into a dict of agent_name: [agent_insights] so expert agent won't repeat the same insights
    insights_history_dict = defaultdict(list)
    for insight in insights_history:
        insights_history_dict[insight["agent_name"]].append(
            insight["agent_insight"])

    #get the configs of any expert agents we should run
    experts_to_run = [ea for ea in default_expert_agent_list if (ea.agent_name in proactive_meta_agent_response)]

    #run all the agents in parralel
    loop = asyncio.get_event_loop()
    agents_to_run_tasks = [expert_agent.run_agent_async(conversation_context, insights_history_dict[expert_agent.agent_name]) for expert_agent in experts_to_run]
    insights_tasks = asyncio.gather(*agents_to_run_tasks)
    insights = loop.run_until_complete(insights_tasks)
    return insights


@time_function()
def run_proactive_meta_agent(conversation_context: str, insights_history: list):
    #get expert agents descriptions
    expert_agents_descriptions_prompt = make_expert_agents_prompts()

    #start up GPT4 connection
    llm = get_langchain_gpt4(temperature=0.2)

    class ProactiveMetaAgentQuery(BaseModel):
        """
        Proactive meta agent that determines which agents to run
        """
        agents_list: list = Field(
            description="the agents to run given the conversation context")

    proactive_meta_agent_query_parser = PydanticOutputParser(pydantic_object=ProactiveMetaAgentQuery)

    extract_proactive_meta_agent_query_prompt = PromptTemplate(
        template=proactive_meta_agent_prompt_blueprint,
        input_variables=["conversation_context", "expert_agents_descriptions_prompt", "insights_history", "force_run_agents_prompt"],
        partial_variables={
            "format_instructions": proactive_meta_agent_query_parser.get_format_instructions()}
    )

    if len(insights_history) > 0:
        insights_history=format_list_data(insights_history)
    else:
        insights_history="None"

    proactive_meta_agent_query_prompt_string = extract_proactive_meta_agent_query_prompt.format_prompt(
            conversation_context=conversation_context, 
            expert_agents_descriptions_prompt=expert_agents_descriptions_prompt,
            insights_history=insights_history,
            force_run_agents_prompt=force_run_agents_prompt
        ).to_string()

    # print("Proactive meta agent query prompt string", proactive_meta_agent_query_prompt_string)

    response = llm([HumanMessage(content=proactive_meta_agent_query_prompt_string)])
    try:
        expert_agents_to_run_list = proactive_meta_agent_query_parser.parse(response.content).agents_list
        return expert_agents_to_run_list
    except OutputParserException:
        return None

