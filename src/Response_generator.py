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
from langchain_core.prompts import PromptTemplate

OLLAMA_URL = "http://localhost:11434/api/generate"

# load prompt : first test with strict prompt
def promptBuilder(Final_context, user_query):
    """
    take prompt template and insert custom context and query in it.

    Args:
    Final_context: take context in string format
    user_query: original query entered by user

    Returns:
    Final prompt containing all information ready to be fed to LLM
    """
    with open('src/Strict_LLM_prompt.txt','r') as f:
        prompt_template = f.read()

    t = PromptTemplate(
        input_variables = ["content", "query"],
        template = prompt_template
    )

    prompt = t.format(content = Final_context, query = user_query)
    return prompt

def CallLLM(prompt):
    headers = {
    "Content-Type": "application/json"
    }

    payload = {
        "model" : "mistral:7b-instruct-q4_0",
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 512
        }
    }
    print("hello 1")
    response = requests.post(OLLAMA_URL, json=payload, headers = headers)
    print("hello 2")
    response.raise_for_status() # raises HTTP error if any occurs.

    return response.json()["response"]
