from DatabaseHandler import DatabaseHandler
from agents.agent_utils import format_list_data
from langchain.agents import initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType
from agents.search_tool_for_agents import get_search_tool_for_agents
from agents.math_tool_for_agents import get_wolfram_alpha_tool_for_agents
from Modules.LangchainSetup import *
from helpers.time_function_decorator import time_function
import time
from agents.generic_agent.agent_insight import *
from agents.generic_agent.generic_agent_prompts import *
from langchain.schema.messages import HumanMessage

db_handler = DatabaseHandler(parent_handler=False)
llm4 = get_langchain_gpt4(max_tokens=180, temperature=0.2)
llmMedium = get_langchain_gpt4o(temperature=0.0)


class GenericAgent:

    def __init__(self,
                 agent_name,
                 tools,
                 insight_num_words,
                 agent_insight_type,
                 agent_plan,
                 validation_criteria,
                 proactive_tool_description,
                 proactive_tool_example,
                 examples,
                 discourage_tool_use=False,
                 try_small_model_first=True,
                 small_model_confidence_threshold=5,
                 min_gatekeeper_score=7) -> None:
        self.agent_name = agent_name
        self.tools = tools
        self.insight_num_words = insight_num_words
        self.agent_insight_type = agent_insight_type
        self.agent_plan = agent_plan
        self.validation_criteria = validation_criteria
        self.proactive_tool_description = proactive_tool_description
        self.proactive_tool_example = proactive_tool_example
        self.examples = examples

        self.discourage_tool_use = discourage_tool_use
        self.try_small_model_first = try_small_model_first
        self.small_model_confidence_threshold = small_model_confidence_threshold
        self.min_gatekeeper_score = min_gatekeeper_score

        self.agent_small = initialize_agent([
            get_search_tool_for_agents(),
        ], llmMedium, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, max_iterations=3, early_stopping_method="generate", verbose=False)

        self.agent_large = initialize_agent([
            get_search_tool_for_agents(),
        ], get_langchain_gpt4o(), agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, max_iterations=3, early_stopping_method="generate", verbose=False)

    def get_agent_prompt(
        self,
        conversation_transcript,
        format_instructions="",
        insights_history: list = [],
        final_command="",
        use_tools_prompt=True,
        use_agent_plan_prompt=True
    ):
        # Populating the blueprint string with values from the agent_config dictionary
        if final_command != "":
            final_command = "\n\n" + final_command

        if len(insights_history) > 0:
            insights_history = format_list_data(insights_history)
        else:
            insights_history = "None"

        expert_agent_prompt = PromptTemplate(
            template=expert_agent_prompt_blueprint,
            input_variables=[
                "agent_name",
                "agent_insight_type",
                "general_tools_prompt",
                "discourage_tool_use_prompt",
                "agent_plan_prompt",
                "examples",
                "conversation_transcript",
                "agent_plan",
                "validation_criteria",
                "insights_history",
                "final_command"
            ],
            partial_variables={
                "format_instructions": format_instructions,
            },
        )

        this_agent_plan_prompt = get_agent_plan_prompt(agent_plan=self.agent_plan, insight_num_words=self.insight_num_words, validation_criteria=self.validation_criteria) if use_agent_plan_prompt else ""

        expert_agent_prompt_string = (
            expert_agent_prompt.format_prompt(
                **vars(self),
                final_command=final_command,
                conversation_transcript=conversation_transcript,
                insights_history=insights_history,
                general_tools_prompt=general_tools_prompt if use_tools_prompt else "",
                discourage_tool_use_prompt=discourage_tool_use_prompt if self.discourage_tool_use else "",
                agent_plan_prompt=this_agent_plan_prompt,
                format_instructions=format_instructions,
            ).to_string()
        )

        # print("expert_agent_prompt\n\n", expert_agent_prompt_string)

        return expert_agent_prompt_string

    def run_agent_wrapper_async(self, agent, agent_explicit_prompt):
        async def run_expert_agent_wrapper_async(command):
            return await agent.ainvoke(agent_explicit_prompt + '\n[Extra Instructions]\n' + command)

        return run_expert_agent_wrapper_async

    def get_agent_as_tool(self, transcript):
        # make the expert agent with it own special prompt
        expert_agent_explicit_prompt = self.get_agent_prompt(transcript)

        agent_tools = []

        if "Search_Engine" in self.tools:
            agent_tools.append(get_search_tool_for_agents())
        if "Wolfram_Alpha" in self.tools:
            agent_tools.append(get_wolfram_alpha_tool_for_agents())

        # make the agent with tools
        new_expert_agent = initialize_agent(agent_tools, llm4, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, max_iterations=4, handle_parsing_errors=True)

        # use function factory to make expert agent runner wrapper
        run_expert_agent_wrapper = self.run_agent_wrapper_async(new_expert_agent, expert_agent_explicit_prompt)

        return Tool(
            name=self.agent_name,
            func=run_expert_agent_wrapper,
            coroutine=run_expert_agent_wrapper,
            description="Use this tool when: " + self.proactive_tool_description
        )

    #
    # Below: Used by proactive meta agent
    #

    def get_agent_info_for_proactive_agent(self, index, simple=False):
        prompt = ""
        prompt += f"\n- Agent {index}:\n"
        prompt += f"""   - Name: {self.agent_name}\n"""
        prompt += f"""   - When to call: {self.proactive_tool_description}\n"""
        prompt += f"""   - Example insight generated: {self.proactive_tool_example}\n"""

        return prompt

    @time_function()
    async def run_agent_async(self, convo_context, insights_history: list):

        agent_start_time = time.time()
        confidence_score = -1
        small_insight = None
        expert_agent_response = None

        if self.try_small_model_first:
            print("Attempting query w/ small model first")
            expert_agent_response = await self.run_simple_llm_async(convo_context, insights_history)
            if expert_agent_response and 'confidence_score' in expert_agent_response:
                confidence_score = expert_agent_response['confidence_score']
            if expert_agent_response and 'agent_insight' in expert_agent_response:
                small_insight = expert_agent_response['agent_insight']
                print("Small model insight: " + small_insight)

        if expert_agent_response is None or confidence_score < self.small_model_confidence_threshold or small_insight == "null":
            print("Small model failed - attempting query w/ large model")
            prompt_str = self.get_agent_prompt(convo_context, format_instructions=agent_insight_parser.get_format_instructions(), insights_history=insights_history, use_tools_prompt=True, use_agent_plan_prompt=True)
            res = await self.agent_large.ainvoke({"input": prompt_str})
            expert_agent_response = res['output']

        print("=== the {} AGENT GENERATION ended in {} seconds ===".format(self.agent_name, round(time.time() - agent_start_time, 2)))

        if expert_agent_response is None:
            return None

        expert_agent_response = post_process_agent_output(expert_agent_response, self.agent_name)  
        return expert_agent_response

    async def run_simple_llm_async(self, convo_context, insights_history: list):
        prompt_str = self.get_agent_prompt(convo_context, format_instructions=agent_insight_parser.get_format_instructions(), insights_history=insights_history, use_tools_prompt=False, use_agent_plan_prompt=False)
        res = await llm4.ainvoke([HumanMessage(content=prompt_str)])
        response = agent_insight_parser.parse(res.content)
        response = response.dict()

        print("Simple LLM response:\n" + str(response))

        return response

    ### Run this specific agent's gatekeeper & run agent if gatekeeper passes ###
    async def run_aio_agent_gatekeeper_async(self, user_id, conversation_context, insights_history: list):
        generic_gatekeeper_start_time = time.time()

        generic_gatekeeper_score_prompt = PromptTemplate(
            template = generic_agent_gatekeeper_prompt_blueprint,
            input_variables = ["agent_name", "agent_insight_type", "conversation_context"],
            partial_variables = {
                "format_instructions": generic_agent_gatekeeper_score_query_parser.get_format_instructions(),
            },
        )

        generic_gatekeeper_score_prompt_string = (
            generic_gatekeeper_score_prompt.format_prompt(
                agent_name=self.agent_name,
                agent_insight_type=self.agent_insight_type,
                conversation_context=conversation_context,
            ).to_string())

        print("GENERIC AGENT GATEKEEPER PROMPT: ")
        print(generic_gatekeeper_score_prompt_string)

        score_response = await llmMedium.ainvoke(
            [HumanMessage(content=generic_gatekeeper_score_prompt_string)]
        )

        try:
            content = generic_agent_gatekeeper_score_query_parser.parse(score_response.content)
            score = int(content.insight_usefulness_score)

            print("=== the {} GATEKEEPER ended in {} seconds ===".format(self.agent_name, round(time.time() - generic_gatekeeper_start_time, 2)))

            if score < self.min_gatekeeper_score:
                # print("SCORE BAD ({} < {})! GATEKEEPER SAYS NO!".format(str(score), str(self.min_gatekeeper_score)))
                return None
            print("{} SCORE GOOD ({} > {})! RUNNING GPT4!".format(self.agent_name, str(score), str(self.min_gatekeeper_score)))
        except OutputParserException as e:
            print("ERROR: " + str(e))
            return None

        # Add this proactive query to history
        db_handler.add_agent_insight_query_for_user(user_id, conversation_context)

        # Generate an insight

        insight = await self.run_agent_async(conversation_context, insights_history)
        print("LE EPIC INSIGHT : ")
        print(str(insight))
        if insight is not None:
            db_handler.add_agent_insight_result_for_user(user_id, insight["agent_name"], insight["agent_insight"], insight["reference_url"])
