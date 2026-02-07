# import dependencies
import faiss
import pickle
import numpy as np
import Retriever
from sentence_transformers import SentenceTransformer
import Response_generator

# restoring persistent data
index = faiss.read_index("Persistent_data/FAISS.index") # index

with open('Persistent_data/embeddings.pkl', 'rb') as f: # Load embeddings from Pickle
    embeddings = pickle.load(f)

with open('Persistent_data/text_chunks.pkl', 'rb') as f: # Load chunks from pickle
    text_chunks = pickle.load(f)

# embedding model 'e5-base-v2' loading
embedding_model = SentenceTransformer('intfloat/e5-base-v2')

while 1:
    query = str(input("enter your query: "))
    # searching query in Database
    results = Retriever.search_query(embedding_model,index,query,5,text_chunks)
    # print(results)
    Final_context = Retriever.build_context(results)
    Final_prompt = Response_generator.promptBuilder(Final_context, query)
    print(Final_prompt)
    RAW_response = Response_generator.CallLLM(Final_prompt)
    print(RAW_response)


