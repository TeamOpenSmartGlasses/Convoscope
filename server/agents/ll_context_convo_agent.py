from typing import Optional

# custom
from agents.agent_utils import format_list_data

# langchain
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field
from helpers.time_function_decorator import time_function

from Modules.LangchainSetup import get_langchain_gpt4o

# pinyin
import jieba
from pypinyin import pinyin, Style


ll_context_convo_prompt_blueprint = """You are a polyglot expert language teacher. You help the language learner by talking to them about their environment.

Target Language: {target_language}
Fluency Level: {fluency_level}

Your Job:
Have a simple conversation with the user, in the language they're trying to learn, about the world around them. Consider the conversation so far, nearby points of interest as conversation topics, and the user's fluency level to moderate the complexity of your outputs. Responses should be short (5-10 words).

Nearby Points of Interest:
{places}

Examples:
Input 1: 10, Starbucks Coffee, Russian
Output 1: Вам нравится кофе?

Input 2: 35, Starbucks Coffee, Russian
Output 2: Какой ваш любимый напиток в Starbucks Coffee?

Input 3: 52, [], French
Output 3: Que faites-vous actuellement pour vivre?

Input 4: 61, The British Museum, Chinese
Output 4: 如何询问去大英博物馆内某个展览的路线？

Remember that this is a conversation, and your next output should be a continuation of the conversation. Do not repeat what you previously said! Try different conversation starters related to the user's environment or interests to keep the conversation engaging. The conversation shouldn't be strictly about the nearby places, and you can talk about anything that comes to mind if it's appropriate. Responses should be short (5-10 words).

If the users asks you a question try to answer it using language appropriate for their fluency level and try to ask a follow-up question to keep the conversation going.

Here is the conversation so far (you are the agent) which all the user will ever see:
{conversation_history}

For beginners (fluency level 1-20), use very basic words and simple sentences to ensure the conversation is easy to follow. If they change the subject, follow them.

Now provide the (very short) output in {target_language} which continues the above conversation. Don't repeat what you said previously, always say new stuff!

Output Format: {format_instructions}
"""


@time_function()
def run_ll_context_convo_agent(places: list, target_language: str = "Russian", fluency_level: int = 35, conversation_history: Optional[list[dict[str, str]]] = None):
    # start up GPT3 connection
    llm = get_langchain_gpt4o(temperature=0.3, max_tokens=200)

    places_string = "\n".join(places)

    # remove Pinyin so it output Hanzi
    remove_pinyin = " (Pinyin)"
    target_language = target_language.replace(remove_pinyin, "")

    class ContextConvoAgentQuery(BaseModel):
        """
        Second language learning contextual conversations agent
        """
        response: str = Field(
            description="the next (short, 5-10 words) response to the user in your conversation.")

    ll_context_convo_agent_query_parser = PydanticOutputParser(
        pydantic_object=ContextConvoAgentQuery)

    extract_ll_context_convo_agent_query_prompt = PromptTemplate(
        template=ll_context_convo_prompt_blueprint,
        input_variables=["places",
                         "target_language", "fluency_level", "conversation_history"],
        partial_variables={
            "format_instructions": ll_context_convo_agent_query_parser.get_format_instructions()}
    )

    ll_context_convo_agent_query_prompt_string = extract_ll_context_convo_agent_query_prompt.format_prompt(
        places=places_string,
        target_language=target_language,
        fluency_level=fluency_level,
        conversation_history=conversation_history,
    ).to_string()
    # print(ll_context_convo_agent_query_prompt_string)
    # print("LL CONTEXT CONVO PROMPT********************************")
    # print(ll_context_convo_agent_query_prompt_string)

    response = llm.invoke(
        [HumanMessage(content=ll_context_convo_agent_query_prompt_string)])
    # print(response)

    try:
        response = ll_context_convo_agent_query_parser.parse(
            response.content).response

        # Function to convert a list of Chinese words to Pinyin
        def chinese_to_pinyin(text):
            # Segment the text into words using jieba
            words = jieba.cut(text)
            # Convert each segmented word to pinyin and join them
            pinyin_output = ' '.join([''.join([py[0] for py in pinyin(word, style=Style.TONE)]) for word in words])
            return pinyin_output

        response_obj = dict()
        raw_tts_string = response

        response_obj["ll_context_for_agent"] = response  # save the response for the agent in case it is converted to Pinyin
        # Apply Pinyin conversion if target_language is "Chinese (Pinyin)"
        if target_language == "Chinese":
            response = chinese_to_pinyin(response)
        
        target_language = "Chinese" if "Chinese" in target_language else target_language # for TTS
        
        response_obj["ll_context_convo_response"] = response # pack the response into a dictionary
        response_obj["to_tts"] = {"text": raw_tts_string, "language": target_language}

        print("LL CONTEXT CONVO RESPONSE: ")
        print(response_obj)
        return response_obj

    except OutputParserException as e:
        print('parse fail')
        print(e)
        return None
