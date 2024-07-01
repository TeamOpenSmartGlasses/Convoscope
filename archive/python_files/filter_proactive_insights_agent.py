# a critic agent that filters down the insights that the rest of the agent system proactively generated
import json
import openai
from Modules.QueryLLM import *
from constants import LLM_FILTER_THRESHOLD
# from server.Prompts.relevance_filter_prompt import relevance_filter_prompt
from helpers.time_function_decorator import time_function

relevance_filter_prompt = (
    lambda context, entity: f"""
[Introduction]
You are an expert in evaluating the relevance of entity definitions within conversational contexts.
[Your Objective]
You are the "Definitions Relevance Filter" subsystem of "Convoscope". Other subsystems in "Convoscope" listen to the live conversation transcript and attempt to define any entity (person, place, thing, event, etc.) that is mentioned. Your primary role is to decide if these entity definitions are worth displaying to the user during their conversation. You will do so by receiving a list of entities, and providing an "Entity Score" between 1 and 10. A low score indicates the entity is not worth showing the user. A high score indicates that the entity should likely be shown to the user.
Higher Scores are assigned if:
- The entity is rare, making it probable that someone in the conversation is unaware of it
- The definition of the entity will probably enhance the quality of the conversation
- the entity is specific
- There's an apparent desire among the interlocutors to know more about the entity
Lower Scores are assigned if:
- they are common and it is thus likely that someone in the conversation already knows about that entity
- the entity is general/generic
- The definition of the entity doesn't contribute to the conversation's quality
Examples: Very gamous countries, locations, companies or public figures   
You will receive many false positives and thus most "Entity Scores" will be low. You should only provide a high score if you are confident that the entity is worth showing to the user.
[Definitions]
- "Entity Name": A specific reference (person, place, thing, event, etc.) mentioned in the conversation.
- "Entity Definition": A short description of the entity.
- "Entity Score":  score reflecting the significance of an entity definition in the context of the conversation.
[Examples of "Entity Score"]
These examples will guide you in determining when an entity is pertinent or not in the given context:
1) "Context": "Recently, I've been diving deep into ancient civilizations, especially the Indus Valley Civilization. Their urban planning and architectural prowess truly astonishes me. I wonder how they managed their water supply system."
    - "Entity Name": Indus Valley Civilization
    - "Entity Definition": "An ancient civilization that flourished in what is now Pakistan and northwest India during the Bronze Age. It's known for its advanced urban planning and architecture."
    - "Entity Score": 9
<Explantion>The context delves into an ancient civilization, which isn't commonly known to everyone. Given the detailed mention and curiosity in the context, the definition can be highly relevant to those unfamiliar with it.</Explantion>
   
2) "Context": "I was reading a book on theoretical physics, and I came across the concept of string theory. It's quite fascinating, though I must admit, some parts were beyond my grasp."
    - "Entity Name": String theory
    - "Entity Definition": "A theoretical framework in which the point-like particles of particle physics are replaced by one-dimensional objects known as strings."
    - "Entity Score": 8
<Explantion>While theoretical physics is a well-discussed topic, not everyone is familiar with the intricacies of string theory. The context indicates some difficulty in understanding, so the definition can provide clarity.</Explantion>
   
3) "Context": "For my vacation, I'm planning to visit Europe. I've always wanted to see the Eiffel Tower and roam the streets of Paris."
    - "Entity Name": Eiffel Tower
    - "Entity Definition": "A wrought-iron lattice tower located in Paris, France. It's one of the most recognized structures in the world."
    - "Entity Score": 4
<Explantion>The Eiffel Tower is a globally recognized structure. While it's a specific point of interest in the context, most people already know what it is. The definition, therefore, has moderate relevance.</Explantion>
   
4) "Context": "Do you have any recommendations for a good coffee machine? I'm looking for something durable and efficient."
    - "Entity Name": Coffee machine
    - "Entity Definition": "A machine used to brew coffee. There are various types, including drip coffee makers, espresso machines, and French presses."
    - "Entity Score": 2
<Explantion>Coffee machines are common household items. The context is looking for recommendations, not necessarily a definition. Thus, the definition might not be highly relevant.</Explantion>
   
5) "Context": "It's so hot outside; I just want to sit indoors with the air conditioning on and read a book."
    - "Entity Name": Air conditioning
    - "Entity Definition": "A system for controlling the temperature, humidity, and sometimes the purity of the air in a building or vehicle."
    - "Entity Score": 1
<Explantion>Air conditioning is a widely understood concept. The context doesn't delve deep into its mechanism or history, so a definition is of minimal relevance.</Explantion>
6) Context: "I've been thinking about my next travel destination, and Kyoto has been on my list for a while. I've heard about its beautiful temples and serene gardens."
    - Entity Name: "Kyoto"
    - Entity Definition: "A city in Japan known for its historic temples, traditional wooden houses, and beautiful gardens. It was the imperial capital of Japan for over a millennium."
    - Entity Score: "6"
<Explantion>While many might know Kyoto as a city in Japan, its historical significance and unique features might not be common knowledge. The context indicates a desire to visit, making the definition moderately relevant.</Explantion>
    
7) Context: "I recently attended a seminar on cryptography, and the speaker delved into the concept of zk-SNARKs. It seemed pivotal to the future of blockchain privacy, but I couldn't quite grasp its intricacies."
    - Entity Name: "zk-SNARKs"
    - Entity Definition: "Stands for 'Zero-Knowledge Succinct Non-Interactive Argument of Knowledge.' It's a form of proof where one can prove possession of certain information without revealing that information, and without any interaction between the prover and verifier."
    - Entity Score: "10"
<Explantion>This is a complex and specialized topic. The context indicates a seminar and the importance of the topic but also a lack of full understanding. A definition can be highly beneficial.</Explantion>
8) "Context": "My doctor recommended that I incorporate more omega-3 fatty acids into my diet. She said it's beneficial for heart health."
    - "Entity Name": "Omega-3 fatty acids"
    - "Entity Definition": "A type of fat found in certain foods and supplements. They can have various health benefits, especially related to heart and brain health. Common sources include fish oil and flaxseeds."
    - "Entity Score": "9"
<Explantion>While omega-3 is a term many might have heard, not everyone knows its sources and benefits. Given the context of a doctor's recommendation, understanding its significance can be very relevant.</Explantion>
9) "Context": "I've been reading a lot about World War II recently, and the leadership of Winston Churchill during those turbulent times is truly commendable."
    - "Entity Name": "Winston Churchill"
    - "Entity Definition": "A British statesman, military officer, and writer who served as Prime Minister of the United Kingdom during two critical periods in history: from 1940 to 1945 when he led Britain to victory in the Second World War, and again from 1951 to 1955. He's known for his leadership during World War II and his role in shaping post-war Europe."
    - "Entity Score": "7"
<Explantion>Winston Churchill is a well-known figure, especially in the context of World War II. However, not everyone might be aware of his detailed contributions and significance. The context praises his leadership, so the definition has good relevance.</Explantion>
10) "Context": "I've started using a new programming language for my latest project called Rust. It promises memory safety without sacrificing performance."
    - "Entity Name": "Rust"
    - "Entity Definition": "A multi-paradigm programming language designed for performance and safety, especially safe concurrency. Rust is syntactically similar to C++, but its designers intend it to provide better memory safety while maintaining high performance."
    - "Entity Score": "8"
<Explantion>The programming language Rust might be known within developer communities but isn't universally understood. The context indicates its use for a project and its features, making the definition highly relevant for clarity.</Explantion>
11) "Context": "While discussing scientific achievements, it's hard not to mention Albert Einstein's contributions."
    - "Entity Name": "Albert Einstein"
    - "Entity Definition": "A German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. He is best known for his equation E=mc^2, which describes the relationship between energy and mass."
    - "Entity Score": "4"
<Explantion>Albert Einstein is one of the most famous physicists, and most people are aware of his significant contributions to science. Thus, his definition might not contribute much to the conversation's quality.</Explantion>
12) "Context": "In the realm of distributed systems, I recently stumbled upon Paxos, a consensus algorithm. I understand it was introduced by someone named Leslie Lamport. Do you know more about him?"
    - "Entity Name": "Leslie Lamport"
    - "Entity Definition": "An American computer scientist best known for his foundational work in distributed systems, particularly the development of the Paxos algorithm. Lamport has also made significant contributions to the fields of formal methods and has received numerous awards, including the Turing Award in 2013."
    - "Entity Score": "9"
<Explantion>While those in computer science might recognize Leslie Lamport for his contributions, the broader public may not. The context suggests a desire to know more about him and his work, making the definition highly relevant to the conversation.</Explantion>
13) "Context": "I was talking to Shakespeare while riding a Tesla and eating an apple near the Eiffel Tower."
    - "Entity Name": "Shakespeare"
    - "Entity Definition": "An English playwright, poet, and actor, widely regarded as the greatest writer in the English language and the world's greatest dramatist."
    - "Entity Score": "2"
<Explantion>The mention of Shakespeare is random and lacks a contextual basis for a detailed explanation, making the definition less relevant.</Explantion>
14) "Context": "I've recently been researching African nations, and Rwanda has caught my attention. I'm particularly intrigued by its post-genocide recovery and the strides it has made in terms of economic development and women's representation in politics. I'm considering a visit next summer."
    - "Entity Name": "Rwanda"
    - "Entity Definition": "A landlocked country in East Africa, known for its lush greenery and diverse wildlife. After facing a devastating genocide in the 1990s, Rwanda has made significant progress in socio-economic development, political stability, and reconciliation efforts. It's also notable for having one of the highest representations of women in parliament globally."
    - "Entity Score": "9"
<Explantion>Rwanda, while gaining global attention for its remarkable progress post-genocide, might not be widely known in all its facets by everyone. The context shows a deep interest and potential visit, making the definition highly relevant for enhancing understanding.</Explantion>
[Input]
1. Context: the last few minutes of the conversation in a single string.
2. Entity definitions, a list of entity definitions like this:
{{ 
    "entity 1 name" : "entity 1 definition",
    "entity 2 name" : "entity 2 definition",
    ...
    "entity n name" : "entity n definition",
}}
[Output]
A list of "Entity Scores", formatted like this:
{{
    "entity 1 name" : <single integer "Entity Score">,
    "entity 2 name" : <single integer "Entity Score">,
    ...
    "entity n name" : <single integer "Entity Score">,
}}
<Context>{context}</Context>
<Entity>{entity}</Entity>
Only output the JSON entity score list, nothing else!
"""
)


class RelevanceFilter:
    def __init__(self, db_handler):
        print("RelevanceFilter initialized")
        self.db_handler = db_handler
        self.banned_terms = ["LOL", "AI", "Caden Pierce", "Alex Israel",
                             "Professor", "God", "Jesus", "Google", "David Newman", "Patty"]

    @time_function()
    def should_display_result_based_on_context(self, user_id, cse_outputs, context):
        valid_outputs = []

        terms_defined_in_last_n_seconds = self.db_handler.get_defined_terms_from_last_nseconds_for_user_device(
            user_id, n=120
        )

        # print(f"===========================CONTEXT: {context}==============================")
        # print(f"===========================TERMS DEFINED IN LAST N SECONDS: {terms_defined_in_last_n_seconds}==============================")
        # print(f"===========================CSE OUTPUTS: {cse_outputs}==============================")

        # Collect all banned terms from the last N seconds and the predefined banned terms
        terms_to_filter = [term["name"]
                           for term in terms_defined_in_last_n_seconds] + self.banned_terms
        # print(f"===========================TERMS TO FILTER: {terms_to_filter}==============================")

        # Filter the cse_outputs based on the combined banned terms
        cse_outputs_filtered = [
            cse_output for cse_output in cse_outputs if cse_output["name"] not in terms_to_filter]
        # print(f"===========================CSE OUTPUTS FILTERED: {cse_outputs_filtered}==============================")
        if not cse_outputs_filtered:
            return []

        # print(f"===========================ENTITIES: {cse_outputs_filtered}==============================")
        entities_filtered = self.llm_relevance_filter(
            cse_outputs_filtered, context)
        try:
            entities_filtered_dict = json.loads(entities_filtered)
            print(
                f"===========================ENTITIES FILTERED: {str(entities_filtered_dict)}==============================")
        except json.JSONDecodeError as e:
            print(
                f"filter_proactive_insights_agent.py: Error parsing JSON: {e}")
            entities_filtered_dict = {}
        else:
            print("Successfully parsed the JSON string.")
        # print(f"===========================ENTITIES FILTERED: {str(entities_filtered)}==============================")

        # valid_outputs.extend(entities_filtered)
        final_entities = [entity for entity in entities_filtered_dict.keys(
        ) if entities_filtered_dict[entity] >= LLM_FILTER_THRESHOLD]

        return final_entities

    @time_function()
    def llm_relevance_filter(self, entities, context):

        llm_entities_input = dict()
        for entity in entities:
            llm_entities_input[entity["name"]] = entity["summary"]

        # Combine context and result for input
        input_text = relevance_filter_prompt(context, str(llm_entities_input))

        # print(f"===========================LLM INPUT: {input_text}==============================")
        response = one_off_query(input_text, 193)

        print(
            f"===========================LLM PREDICTION: {response}==============================")
        return response

# if __name__ == "__main__":
#     relevance_filter = RelevanceFilter(None)
#     relevance_filter.llm_relevance_filter([{"name":"fdsfds", "summary":"fdsfds"}, {"name":"fdsfds", "summary":"fdsfds"}], "fdsf")
