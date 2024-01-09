from langchain.chat_models import ChatOpenAI
from langchain.chat_models.azure_openai import AzureChatOpenAI
from server_config import use_azure_openai, openai_api_key, azure_openai_api_key, azure_openai_api_base, azure_openai_api_gpt4_deployment, azure_openai_api_gpt35_deployment
from langchain.chat_models.azure_openai import *
from constants import GPT_4_MODEL, GPT_4_MAX_TOKENS, GPT_TEMPERATURE, GPT_35_MODEL, GPT_35_MAX_TOKENS
from langchain.callbacks import openai_info

def get_langchain_gpt4(temperature=GPT_TEMPERATURE, model=GPT_4_MODEL, max_tokens=GPT_4_MAX_TOKENS):
    if use_azure_openai:
        return AzureChatOpenAI(openai_api_key=azure_openai_api_key, 
                            openai_api_base=azure_openai_api_base,
                            openai_api_version = "2023-05-15",
                            deployment_name=azure_openai_api_gpt4_deployment,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
    else:
        return ChatOpenAI(temperature=GPT_TEMPERATURE, openai_api_key=openai_api_key, model=model, max_tokens=GPT_4_MAX_TOKENS)
    
def get_langchain_gpt35(temperature=GPT_TEMPERATURE, model=GPT_35_MODEL, max_tokens=GPT_35_MAX_TOKENS):
    if use_azure_openai:
        return AzureChatOpenAI(openai_api_key=azure_openai_api_key, 
                            openai_api_base=azure_openai_api_base,
                            openai_api_version = "2023-05-15",
                            deployment_name=azure_openai_api_gpt35_deployment,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            )
    else:
        return ChatOpenAI(temperature=GPT_TEMPERATURE, openai_api_key=openai_api_key, model=model, max_tokens=GPT_35_MAX_TOKENS)