# import dependencies
import faiss
import pickle
import numpy as np
import Retriever
from sentence_transformers import SentenceTransformer
import Response_generator
# from chat_history import store
import chat_history

# restoring persistent data
index = faiss.read_index("../Persistent_data/FAISS.index") # index
print("1")
with open('../Persistent_data/embeddings.pkl', 'rb') as f: # Load embeddings from Pickle
    embeddings = pickle.load(f)
print("2")
with open('../Persistent_data/text_chunks.pkl', 'rb') as f: # Load chunks from pickle
    text_chunks = pickle.load(f)
print("3")
# embedding model 'e5-base-v2' loading
embedding_model = SentenceTransformer('intfloat/e5-base-v2')
print("4")
while 1:
    session = "abc123"
    query = str(input("enter your query: "))
    chat_history.save_message(session, "user", query) # save user message to db
    print("10")
    # session_history = chat_history.get_session_history(session)      # get session history from in memory store
    session_history = chat_history.load_session_history(session, conversations = 5) # get session history from db
    print("15")
    rewritten_query = chat_history.rewrite_query(query,session_history)

    # searching query in Database
    results = Retriever.search_query(embedding_model,index,rewritten_query,5,text_chunks)
    # print(results)

    # select whether to take relaxed prompt or strict prompt
    mode = Response_generator.promptSelector(rewritten_query,results)

    # build context ready to be fed to LLM
    Final_context = Retriever.build_context(results, debug=True)
    # print(Final_context)
    # build prompt by replacing context and query in selected mode's template
    Final_prompt = Response_generator.promptBuilder(Final_context, rewritten_query, mode)
    # print(Final_prompt)
    RAW_response = Response_generator.CallLLM(Final_prompt)
    # print(RAW_response)
    clean_answer = chat_history.extract_clean_response(RAW_response)

    # print(chat_history.save_session_history(session,query,clean_answer)) # save history to local in memory store
    chat_history.save_message(session, "AI", clean_answer) # save AI message to db
    # print(store)


