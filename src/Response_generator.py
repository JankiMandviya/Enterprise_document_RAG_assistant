"""
This module takes context and generates response using LLM

This stage includes:
- Load prompt template (strict or relaxed)
- Inject:
    -Context
    -User query
- Call the LLM
- Return raw answer text
"""
import requests
import time
import os
import initialize
from langchain_core.prompts import PromptTemplate

def promptSelector(query,chunks):
    # query intent selection:(strict/relax)
    threshold = 0.80
    StrongChunks = 0
    query = query.lower()

    strict_triggers = [
        "what is", "what are", "who", "why", "when", "where", "according to", "define", "how to", "what"
    ]

    for chunk in chunks:
        if chunk['similarity_score'] >= threshold:
            StrongChunks += 1

    for trigger in strict_triggers:
        if trigger in query:
            return "strict"

    if StrongChunks<1:
        return "strict"

    return "relax"


# load prompt : first test with strict prompt
def promptBuilder(Final_context, user_query, mode):
    """
    take prompt template and insert custom context and query in it.

    Args:
    Final_context: take context in string format
    user_query: original query entered by user
    mode: decides template selection. Possible values:["relax","strict"]

    Returns:
    Final prompt containing all information ready to be fed to LLM
    """
    if mode == "relax":
        print("relaxed mode")
        with open(os.path.join(initialize.BASE_DIR,'src/Relaxed_LLM_prompt.txt'),'r') as f:
            prompt_template = f.read()
    else:
        print("strict mode")
        with open(os.path.join(initialize.BASE_DIR,'src/Strict_LLM_prompt.txt'),'r') as f:
            prompt_template = f.read()
        
    t = PromptTemplate(
        input_variables = ["content", "query"],
        template = prompt_template
    )

    prompt = t.format(content = Final_context, query = user_query)
    return prompt

# def CallLLM(prompt):  # using ollama
#     headers = {
#     "Content-Type": "application/json"
#     }

#     payload = {
#         "model" : "mistral:7b-instruct-q4_0",
#         "prompt" : prompt,
#         "stream" : False,
#         "options": {
#             "temperature": 0.1,
#             "top_p": 0.9,
#             "num_predict": 512
#         }
#     }
#     print("hello 1")
#     start = time.time()
#     response = requests.post(initialize.OLLAMA_URL, json=payload, headers = headers)
#     print("hello 2")
#     end = time.time()
#     print(end-start)
#     response.raise_for_status() # raises HTTP error if any occurs.

#     return response.json()["response"]

# def CallLLM(final_prompt):   # using LM studio
#     """
#     Call LLM and pass final prompt to it and return text response

#     Args:
#     final_prompt: final prompt with context and query in string format

#     Returns:
#     response content
#     """

#     # LM studio requires following format of input data to model
#     payload = {
#         "model": "mistralai/mistral-7b-instruct-v0.3",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": final_prompt
#             }
#         ],
#         "temperature": 0.1,
#         "top_p": 0.9,
#         "max_tokens": 200,
#         "stream": False
#     }

#     headers = {"Content-Type": "application/json"}  
#     response = requests.post(initialize.LM_STUDIO_URL, json=payload, headers=headers)
#     response.raise_for_status()

#     return response.json()["choices"][0]["message"]["content"]

def CallLLM(final_prompt):   # Using Mistral official API mistral model
    """
    Call LLM and pass final prompt to it and return text response

    Args:
    final_prompt: final prompt with context and query in string format

    Returns:
    response content
    """

    response = initialize.Mistral_client.chat.complete(
        model="open-mistral-7b",  # Mistral 7b instruct
        temperature=0.1,
        max_tokens=500, 
        top_p = 0.9,
        stream = False,
        messages=[
            {"role": "user", "content": final_prompt}
        ]
    )
    
    return response.choices[0].message.content