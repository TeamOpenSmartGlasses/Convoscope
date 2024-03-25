# custom
from collections import defaultdict
from server_config import openai_api_key

# langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field
from helpers.time_function_decorator import time_function

from Modules.LangchainSetup import *

speech_coach_prompt_blueprint = """You are a speech coach which helps the user speak better, specifically, reduce the number of filler words during a conversation.

You will receive the transicript of the user's conversation.

Your role is:
1. Output a boolean to tell us if the transcription has filler words.
3. Output the number of filler words.

Transcript:
```{transcription}```

Output Format: {format_instructions}"""

egometer_ego_assessment_prompt_blueprint = """You are an Ego Meter, designed to help users understand when their speech may be perceived as self-centered, overly confident, or diminishing to others.

You will analyze the transcript of the user's conversation.

Your role is:
1. Output the severity of ego-centric speech on a scale from 1 to 10, where 10 indicates extremely ego-centric speech, and 0 is not egocentric.

Examples of ego-centric speech include:
- Overuse of "I", "me", "my" when not necessary for the conversation.
- Bragging about achievements or skills without context.
- Interrupting others to bring the conversation back to oneself.
- Making comparisons that elevate oneself above the group or another person.
- Using language that implies the speaker is the sole reason for successes or that their contributions are the most significant.

Transcript:
```{transcription}```

Output Format: {format_instructions}"""

understandability_meter_prompt_blueprint = """You are an Understandability Meter, designed to help users gauge how easily their speech can be understood by a general audience. Your analysis will focus on the clarity of explanation, appropriate use of vocabulary, and overall accessibility of the speech.

You will analyze the transcript of the user's conversation.

Your role is:
1. Output the level of understandability of the speech on a scale from 1 to 10, where 10 indicates extremely easy to understand, and 1 indicates very difficult to understand.

Factors affecting understandability include:
- Use of esoteric vocabulary not common to the listener's experience or unnecessary for the topic.
- Overly complex sentence structures that obscure the message.
- Lack of context or explanation for complex topics, assuming prior knowledge.
- Rapid topic changes without clear transitions.
- Failing to summarize or reiterate key points for emphasis and clarity.

Conversely, factors improving understandability:
- Simplifying explanations and using analogies where appropriate.
- Introducing complex topics with a brief overview before diving into details.
- Ensuring vocabulary matches the audience's understanding level.
- Using clear, concise sentences.
- Logical flow of ideas with clear transitions.

Transcript:
```{transcription}```

Output Format: {format_instructions}"""

#If the current summary of the 'Recent Transcript' is still correct, just re-output the current summary, don't output a new one.

@time_function()
def run_speech_coach_agent(transcription):
    # start up GPT3 connection
    #llm = get_langchain_gpt35(temperature=0.2, max_tokens=512)
    # start up GPT4 connection
    llm = get_langchain_gpt4(temperature=0.2, max_tokens=512)

    class SpeechCoachAgentQuery(BaseModel):
        """
        ADHD Short Term Memory Buffer agent
        """        
        has_filler_words: bool = Field(
            description="True if the input text has filler words, False otherwise. Examples of filler words include: 'um', 'uh', 'like', 'you know', 'so', 'well', 'actually', 'basically', 'literally', 'seriously', 'honestly', 'truly', 'obviously', 'somewhat', 'kind of', 'sort of', 'a lot', 'lots of'. Make sure not to confuse proper use of the word with filler words. For example, 'like' can be used as a filler word, but it can also be used as a verb or a preposition. If the word is used as a verb or preposition, it is not.")

        number_of_filler_words: int = Field(
            description="The number of filler words in the input text. If there are no filler words, output 0.")

    speech_coach_agent_query_parser = PydanticOutputParser(
        pydantic_object=SpeechCoachAgentQuery)

    extract_speech_coach_agent_query_prompt = PromptTemplate(
        template=speech_coach_prompt_blueprint,
        input_variables=["transcription"],
        partial_variables={
            "format_instructions": speech_coach_agent_query_parser.get_format_instructions()}
    )

    speech_coach_agent_query_prompt_string = extract_speech_coach_agent_query_prompt.format_prompt(
        transcription=transcription,
    ).to_string()

    #print("ADHD STMB PROMPT********************************")
    #print(adhd_stmb_agent_query_prompt_string)

    response = llm(
        [HumanMessage(content=speech_coach_agent_query_prompt_string)])
    print("SPEECH COACH AGENT RESPONSE ********************************")
    print(response)

    try:
        has_filler_words = speech_coach_agent_query_parser.parse(
            response.content).has_filler_words

        number_of_filler_words = speech_coach_agent_query_parser.parse(
            response.content).number_of_filler_words
        # print("-"*20)
        # print("GPT: Num of most recent filler words: ", number_of_filler_words)
        # print("-"*20)
        return number_of_filler_words
        # if has_filler_words:
        #     return number_of_filler_words
        # else:
        #     return "0"
    except OutputParserException as e:
        print('parse fail')
        print(e)
        return None


@time_function()
def run_egometer_agent(transcription):
    # Assuming get_langchain_gpt4 and related functions are defined elsewhere
    # start up GPT4 connection
    llm = get_langchain_gpt4(temperature=0.2, max_tokens=512)

    class EgometerAgentQuery(BaseModel):
        """
        Egometer agent for assessing ego-centric speech
        """        
        ego_severity_score: int = Field(
            description="The ego meter severity score on a scale from 1 (not egocentric at all) to 10 (extremely egocentric).")

    egometer_agent_query_parser = PydanticOutputParser(
        pydantic_object=EgometerAgentQuery)

    extract_egometer_agent_query_prompt = PromptTemplate(
        template=egometer_ego_assessment_prompt_blueprint,
        input_variables=["transcription"],
        partial_variables={
            "format_instructions": egometer_agent_query_parser.get_format_instructions()}
    )

    egometer_agent_query_prompt_string = extract_egometer_agent_query_prompt.format_prompt(
        transcription=transcription,
    ).to_string()

    print("EGOMETER AGENT RESPONSE ********************************")
    response = llm(
        [HumanMessage(content=egometer_agent_query_prompt_string)])
    print(response)

    try:
        ego_severity_score = egometer_agent_query_parser.parse(
            response.content).ego_severity_score
        return ego_severity_score
    except OutputParserException as e:
        print('parse fail')
        print(e)
        return None


@time_function()
def run_understandability_agent(transcription):
    # Assuming get_langchain_gpt4 and related functions are defined elsewhere
    # start up GPT4 connection
    llm = get_langchain_gpt4(temperature=0.2, max_tokens=512)

    class UnderstandabilityAgentQuery(BaseModel):
        """
        Understandability Meter agent for assessing speech clarity and accessibility
        """        
        understandability_score: int = Field(
            description="The understandability score on a scale from 1 (very difficult to understand) to 10 (extremely easy to understand).")

    understandability_agent_query_parser = PydanticOutputParser(
        pydantic_object=UnderstandabilityAgentQuery)

    extract_understandability_agent_query_prompt = PromptTemplate(
        template=understandability_meter_prompt_blueprint,
        input_variables=["transcription"],
        partial_variables={
            "format_instructions": understandability_agent_query_parser.get_format_instructions()}
    )

    understandability_agent_query_prompt_string = extract_understandability_agent_query_prompt.format_prompt(
        transcription=transcription,
    ).to_string()

    print("UNDERSTANDABILITY METER RESPONSE ********************************")
    response = llm(
        [HumanMessage(content=understandability_agent_query_prompt_string)])
    print(response)

    try:
        understandability_score = understandability_agent_query_parser.parse(
            response.content).understandability_score
        return understandability_score
    except OutputParserException as e:
        print('parse fail')
        print(e)
        return None
