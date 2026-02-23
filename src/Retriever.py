"""
This file contains retrieves query result from FAISS database and filters chunks for context construction for LLM.

This stage includes:
1. Searching queries in FAISS
2. Thresholding to remove irrelevant/bad quality chunks
3. Context construction
"""

import os
import numpy as np
import initialize

def search_query(embedding_model,session_id:int,query,k):
    """
    returns top K results for given query in FAISS and find its equivalent chunk metadata and content from SQLite using chunk id.

    Args:
    embedding_mode = preloaded sentence transformer
    session_id = in integer, find only in current session chunks.
    query : query string
    k: number of top nearest neighbors

    Returns:
    final_result = final sorted list(primary:similarity, secondary: page) of max_chunks chunks which contain
    content, page, title, source, similarity score.
    """
    if os.path.exists(initialize.index_path):
        index = initialize.faiss.read_index(initialize.index_path)
    else:
        print("Faiss index doesn't exist")
        return []
    
    if index.ntotal == 0:
        return []
    
    db = next(initialize.get_db())
    results = []
    threshold = 0.70
    max_chunks = 3               # allow maximum 3 chunks to get into LLM prompt.
    valid_ids = []               # keep indexes which has confidence greater than threshold.
    id_to_score = {}             # stores chunk id as key and respective similarity score result as value.
    chunk_dict = {}              # Store search result chunk's content and metadata in this dictionary.

    query = "query: " + query
    query_em = embedding_model.encode([query],normalize_embeddings=True).astype("float32")
    print(query_em.shape)

    # if multiple results, Distances and Indexes are = [[]] --> therefore Distances[0], Indexes[0] needed to remove extra dimension
    Distances, Indexes = index.search(query_em,k)  # Distances = list of similarity score, Indexes = list of indexes(we will use this index as direct chunk id to search in SQLite to find respective original chunk metadata)
    Distances, Indexes = Distances[0], Indexes[0]

    # filter out all the indexes which have similarity score less than threshold and index is -1(no similar result found)
    for i,Index in enumerate(Indexes):
        if threshold <= Distances[i] and Index!=-1: 
            valid_ids.append(int(Index))
            id_to_score[Index] = float(Distances[i])

    # if the index type were L2, then Distances will contain euclidean distance. smaller the better, therefore take k elements with smallest distances.
    # For index type IP, Distances will contain cosine similarity. Bigger the better, therefore take k elements with largest distances.
    # query to db to find all the chunk objects with indexes in valid_ids and are in current session_id.
    valid_chunks = db.query(initialize.DocumentChunk).filter(initialize.DocumentChunk.id.in_(valid_ids), initialize.DocumentChunk.session_id == session_id).all()

    # extract information about these valid chunks from DB.
    for chunk in valid_chunks:
        chunk_dict['content'] = chunk.chunk_text
        chunk_dict['title'] = chunk.doc_title
        chunk_dict['source'] = chunk.doc_source
        chunk_dict['page'] = chunk.page_number
        chunk_dict['similarity_score'] = id_to_score.get(chunk.id,0.0)
        results.append(chunk_dict)
        chunk_dict = {}
            
    # once results are acquired, sort them in (similarity_score,page) order to get bext context.
    # similarity score should be in descending and page should be in ascending.
    results_sorted = sorted(
        results,
        key=lambda x: (-x["similarity_score"], x["page"])
    )

    # allow only max_chunk number of chunks. If result_sorted has too many good quality chunks, pick only top max_chunks to save token limit and keeping LLM focused on 1-2 points.
    final_result = results_sorted[:max_chunks]
    print("hey janki: ", final_result)
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

