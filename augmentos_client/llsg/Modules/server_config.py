# OpenAI API or Azure OpenAI API can be used.
use_azure_openai = True

# Populate this if using OpenAI API
openai_api_key = ""

# Populate this if using Azure OpenAI API
azure_openai_api_key = "8fb4dd771c3047cbb33821b96464c18c"
azure_openai_api_base = "https://convoscoperesource2.openai.azure.com/"
azure_openai_api_gpt35_deployment = "ConvoscopeGPT3"
azure_openai_api_gpt35_16k_deployment = "ConvoscopeGPT3-16K"
azure_openai_api_gpt4_deployment = "ConvoscopeGPT4"
azure_openai_api_gpt4o_deployment = "ConvoscopeGPT4o"

# Unused for now
azure_api_llama_13b_chat_endpoint = "https://Llama-2-13b-chat-aypss-serverless.eastus2.inference.ai.azure.com/v1/chat/completions"
azure_api_llama_13b_chat_api_key = "HwPgqg72o5eUepbIcMJldrkn7fOtZ6t9"

time_everything_spreadsheet_id = "1UD_Lf4V9yP5fdpgNb1PwLJAye2A7K5ihw7ozoO3cmz4"

# Use these for dev/testing purposes
clear_users_on_start = True
clear_cache_on_start = False
ignore_auth = False

# Other API Keys
gcp_project_id = "alexwearllm"
google_maps_api_key = "AIzaSyDrLB_P3dXj-CktMkNrHzlLS7p5cAZS2_Q"
serper_api_key = "0313f26c9142afce8e6c09d296aef3f3649d2361"
wolframalpha_api_key = "U7P33Q-WLYJUA7GWA"

# Uncomment one of the following configs:
# Local:
# database_uri = "mongodb://localhost:27017"
# server_port = 8080
# path_modifier = ""

# These are for use with official Convoscope(tm) backend - ignore if self-hosting

# Prod:
database_uri = "mongodb://localhost:27017"
server_port = 8080
path_modifier = ""

# Dev:
#database_uri = "mongodb://localhost:27019"
#server_port = 8081
#path_modifier = "dev/"

# Dev2:
# database_uri = "mongodb://localhost:27020"
# server_port = 8082
# path_modifier = "dev2/"

# MIT:
#database_uri = "mongodb://localhost:27021"
#server_port = 8083
#path_modifier = "mit/"
