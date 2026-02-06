"""
This file contains retrieves query result from FAISS database and filters chunks for context construction for LLM.

This stage includes:
1.	Searching queries in FAISS
2. Thresholding to remove irrelevant/bad quality chunks
3. Context construction
"""

# import dependencies
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def search_query(index,query,k,chunks):
    """
    returns top K results for given query

    Args:
    index: index of vector database
    query : query string
    k: number of top nearest neighbors
    chunks: original main document chunks

    Returns:
    final_result = final sorted list(primary:similarity, secondary: page) of max_chunks chunks which contain
    content, page, title, source, similarity score.
    """
    results = []
    threshold = 0.70
    max_chunks = 3 # allow maximum 3 chunks to get into LLM prompt.
    chunk = {}

    query = "query: " + query
    query_em = embedding_model.encode([query],normalize_embeddings=True)
    print(query_em.shape)
    Distances,Indexes = index.search(query_em,k)  # Distances = list of similarity score, Indexes = list of indexes
    Distances,Indexes = Distances.squeeze(),Indexes.squeeze()

    if k>1:
        for i,Index in enumerate(Indexes):
            # if the index type were L2, then Distances will contain euclidean distance. smaller the better, therefore take k elements with smallest distances.
            # For index type IP, Distances will contain cosine similarity. Bigger the better, therefore take k elements with largest distances.
            if threshold <= Distances[i]:          
                # chunks[Index].metadata['similarity_score'] = Distances[i]
                chunk['content'] = chunks[Index].page_content
                chunk['title'] = chunks[Index].metadata['title']
                chunk['source'] = chunks[Index].metadata['source']
                chunk['page'] = chunks[Index].metadata['page']
                chunk['similarity_score'] = float(Distances[i])
                results.append(chunk)
                chunk = {}
    else:
        if threshold<=Distances:
            # chunks[Indexes].metadata['similarity_score'] = Distances
            chunk['content'] = chunks[Indexes].page_content
            chunk['title'] = chunks[Indexes].metadata['title']
            chunk['source'] = chunks[Indexes].metadata['source']
            chunk['page'] = chunks[Indexes].metadata['page']
            chunk['similarity_score'] = float(Distances)
            results.append(chunk)
            chunk = {}

    # once results are acquired, sort them in (similarity_score,page) order to get bext context.
    # similarity score should be in descending and page should be in ascending.
    results_sorted = sorted(
        results,
        key=lambda x: (-x["similarity_score"], x["page"])
    )

    # allow only max_chunk number of chunks. If result_sorted has too many good quality chunks, pick only top max_chunks to save token limit and keeping LLM focused on 1-2 points.
    final_result = results_sorted[:max_chunks]
    return final_result


def build_context(results,debug=False):
    """
    build context ready to be fed to LLMs
    Args:
    results: chunks sorted in priority order
    debug: if true give similarity score as well in input, if false remove it.

    Returns:
    Final context: list of block containing metadata and content in string format.
    """
    Final_context = []
    max_chunk_length = 800    # if any chunk is longer than max_chunk_length, strip it. so that it doesn't dominate the LLM response
    max_context_length = 2500  # maximum context size.
    current_len = 0            # current context size

    for chunk in results:
        content = chunk['content']
        if len(chunk['content'])>max_chunk_length:
            content = chunk['content'][:max_chunk_length].rstrip() + "..."

        if debug:
            block = f"""
            [source: {chunk['source']} | title: {chunk['title']} | page: {chunk['page']} | similarity_score: {chunk['similarity_score']}]
            {content}
            """.strip()
        
        else:
            block = f"""
            [source: {chunk['source']} | title: {chunk['title']} | page: {chunk['page']}]
            {content}
            """.strip()

        if current_len + len(block) > max_context_length:
            break
        Final_context.append(block)
        Final_context.append("\n---\n")
        current_len+=len(block)
    
    Final_context = "\n\n".join(Final_context)
    return Final_context

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
    results = search_query(index,query,5,text_chunks)
    print(results)
    print(build_context(results))

    