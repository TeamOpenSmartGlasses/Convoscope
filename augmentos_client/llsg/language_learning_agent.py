# custom
from collections import defaultdict
from llsg.agent_utils import format_list_data

# langchain
from langchain.prompts import PromptTemplate
from langchain.schema import (
    HumanMessage
)
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import OutputParserException
from pydantic import BaseModel, Field
from llsg.helpers.time_function_decorator import time_function

#pinyin
import jieba
from pypinyin import pinyin, Style

from llsg.Modules.LangchainSetup import *

language_learning_agent_prompt_blueprint = """You are listening to a user's conversation right now. You help the language learner user by identifying vocabulary/words from the conversation transcript (Input Text) that the user might not understand and then translate just those words into the Ouput language. You output 0 to 3 words. If the learner's fluency level is less than 50, they will need about 1/5 words define. 50-75 fluency level might need 1 word per sentence. If fluency level is >75, only choose and translate very rare words.

Input Text Language: {transcribe_language}
Output (translated) language: {output_language}
Fluency Level: {fluency_level}

Process:
0. Consider the fluency level of the user, which is {fluency_level}, where 0<=fluency_level<=100, with 0 being complete beginner, 50 being conversational, 75 intermediate and 100 being native speaker.
1. Skim through the Input Text and identify 0 to 3 words that may unfamiliar to someone with a fluency level of {fluency_level} AND that have not been previously defined.
2. Consider how common a word is (the word frequency percentile) to determine how likely the user knows that word.
3. For each of the zero to three identified words in the Input Text, provide a translation in {output_language}. Make translations short. Use context from the conversation to inform translation of homonyms.
4. Output response using the format instructions below, provide words in the order they appear in the text. Don't redefine any words that are in the "Recently Translated" list of words.

Examples:
Conversation 1: 'I ran for the train, but the fruit stand was in the way'
Input Language 1: English
Output Language 1: Chinese
Output 1: {{'train': '火车', 'fruit stand': '水果摊'}}

Conversation 2: "О, так вы студент биологии, это здорово"
Input Language 2: Russian
Output Language 2: English
Output 2: {{"студент" : "student", "биология" : "biology"}}

Conversation 3: "let's go and say hello to her"
Input Language 3: English
Output Language 3: Spanish
Output 3: {{}}

Input Text: `{conversation_context}`

Frequency Ranking: The frequency percentile of each word tells you how common it is in daily speech (~0.1 is very common, >1.2 is rare, >13.5 is very rare). The frequency ranking of the words in the "Input Text" are: `{word_rank}`

Recently Translated: `{live_translate_word_history}`

Output Format: {format_instructions}

Always translate at least 1 word.

Don't output punctuation or periods! Output all lowercase! Define 1/5 of the words in the input text (never define all of the words in the input, never define highly common words like "the", "a", "it", etc.). Now provide the output:"""

#opposite language (either {source_language} or {target_language}, whatever is

#This level influences the selection of words to translate:
#   - 0-49 (Beginner): Translate all but the most common words, translate every few words, you'll ussually translate multiple words. Define most words with percentile rank > 0.25.
#   - 50-74 (Conversational): Translate approximately one word per sentence, selecting words that are somewhat common but might still pose challenges.
#   - 75-99 (Intermediate): Only translate rare or complex words that an intermediate learner might not encounter frequently.
#   - 100 (Native Speaker): No translation is necessary unless extremely rare or technical terms are used.

#A word with frequency percentile 1.5 is a little uncommon, a word with percentile 0.15 is very common, a word with percentile >=30 is super rare. Use the percentile to determine words that the user might not know, where higer number is more rare.

#*** Never do intralanguage translation (don't translate within the same language).
# using the format instructions above:
#
#There should be <= 3 words per run in the dict. Don't output any explanation or extra data, just this simple info. It's OK to output zero words if there are no appopriately rare words in the input text.

#Don't define stop words or super common words like "yes", "no", "he", "hers", "to", "from", "thank you", "please", etc. in ANY language, but you can define even semi-common words like "exactly", "bridge", "computer", etc. for beginners with a fluency level less than 50 (current user fluency is {fluency_level}.

# Define lots of words, at least 1 per run.
#
#Conversation 4: "that museum’s architecture is very beautiful."
#Source Language 4: English
#Target Language 4: Chinese (Pinyin)
#Output 4: {{"museum" : "bówùguǎn", "architecture" : "jiànzhú", "beautiful" : "měilì"}}


#Preface Rule:
#Use Pinyin when writing Chinese. Never use Chinese characters, use Pinyin.
#, use Pinyin if writing Chinese)
#3.a. If writing Chinese, output exclusively in Pinyin, avoiding Chinese characters entirely.
#NEVER OUTPUT CHINESE CHARACTERS. USE PINYIN, USE THE LATIN ALPHABET PINYIN FOR CHINESE.

#Use Pinyin when writing Chinese. Never use Chinese characters, use Pinyin.

# in_word_translation must be in Pinyin or Latin characters.

# If writing Chinese, output exclusively in Pinyin, avoiding Chinese characters entirely.


def format_list_data(data: dict) -> str:
    """
    Formats a dict into a string that can be used in a prompt
    """
    data_str = ""
    for key, value in data.items():
        data_str += f"{key} : {value}\n"
    return data_str


@time_function()
async def run_language_learning_agent(conversation_context: str, word_rank: dict, target_language="Russian", transcribe_language="English", source_language="English", live_translate_word_history=""):
    # start up GPT4o connection
    llm = get_langchain_gpt4o(temperature=0.2, max_tokens=80)
    #llm = get_langchain_gpt35(temperature=0.2, max_tokens=80)

    # remove punctuation
    conversation_context = conversation_context
    fluency_level = 15  # Example fluency level
    remove_pinyin = " (Pinyin)"
    output_language = source_language.replace(remove_pinyin, "")
    if transcribe_language == source_language:
        output_language = target_language.replace(remove_pinyin, "")

    class LanguageLearningAgentQuery(BaseModel):
        """
        Proactive language learning agent
        """
        translated_words: dict = Field(
            description="the chosen input text and their translations")

    language_learning_agent_query_parser = PydanticOutputParser(
        pydantic_object=LanguageLearningAgentQuery)

    extract_language_learning_agent_query_prompt = PromptTemplate(
        template=language_learning_agent_prompt_blueprint,
        input_variables=["conversation_context", "target_language", "transcribe_language", "output_language", "source_language", "fluency_level", "word_rank", "live_translate_word_history"],
        partial_variables={
            "format_instructions": language_learning_agent_query_parser.get_format_instructions()}
    )

    word_rank_string = format_list_data(word_rank)

    language_learning_agent_query_prompt_string = extract_language_learning_agent_query_prompt.format_prompt(
        conversation_context=conversation_context,
        source_language=source_language,
        target_language=target_language,
        fluency_level=fluency_level,
        word_rank=word_rank_string,
        output_language=output_language,
        transcribe_language=transcribe_language,
        live_translate_word_history=live_translate_word_history
    ).to_string()

    response = await llm.ainvoke(
        [HumanMessage(content=language_learning_agent_query_prompt_string)])

    try:
        translated_words = language_learning_agent_query_parser.parse(
            response.content).translated_words

        # Drop too common words
        word_rank_threshold = 0.2
        translated_words_rare = dict()
        for word, translation in translated_words.items():
            if word in word_rank and word_rank[word] >= word_rank_threshold:
                translated_words_rare[word] = translation

        # Function to convert a list of Chinese words to Pinyin
        def chinese_to_pinyin(text):
            # Segment the text into words using jieba
            words = jieba.cut(text)
            # Convert each segmented word to pinyin and join them
            pinyin_output = ' '.join([''.join([py[0] for py in pinyin(word, style=Style.TONE)]) for word in words])
            return pinyin_output

        # Apply Pinyin conversion if needed
        if "Pinyin" in target_language or "Pinyin" in source_language:
            translated_words_pinyin = {chinese_to_pinyin(word): chinese_to_pinyin(translated_words_rare[word]) if isinstance(translated_words_rare[word], str) and any('\u4e00' <= char <= '\u9fff' for char in translated_words_rare[word]) else translated_words_rare[word] for word in translated_words_rare}

            # Convert input words to Pinyin if the source language contains Pinyin
            if "Pinyin" in source_language:
                translated_words_pinyin = {chinese_to_pinyin(word): translation for word, translation in translated_words_pinyin.items()}
        else:
            translated_words_pinyin = translated_words_rare

        # Drop any repeats and then pack into list
        translated_words_obj = []
        live_translate_word_history_set = set(live_translate_word_history)  # Convert to set for efficient lookup
        for word, translation in translated_words_pinyin.items():
            if word not in live_translate_word_history_set:  # Check if word is not in the already translated words
                translated_words_obj.append({"in_word": word, "in_word_translation": translation})

        return translated_words_obj
    except OutputParserException as e:
        print('parse fail')
        print(e)
        return None


if __name__ == "__main__":
    # "It's a beautiful day to be out and about at the library! And you should come to my house tomorrow!"
    conversation_context = "It's a beautiful day to be out and about at the library! And you should come to my house tomorrow!"
    word_rank = {"beautiful": 100, "library": 200, "house": 300}
    run_language_learning_agent(conversation_context, word_rank)
