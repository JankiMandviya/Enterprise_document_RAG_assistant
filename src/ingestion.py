"""
This file contains document ingestion and chunking modules. This module will run only 1 time offline to generate document embedding and save it to vector DB. This will not run each time when user enters query.

This stage includes:
1.	Loads documents
2.	Extracts raw text
3.	Splits text into chunks
4.	Attaches metadata (source, page)
5.  converts chunks into normalized embeddings using sentence transformer
6.  Adds these document chunk embeddings to FAISS
"""

# import dependencies
import faiss
import pickle
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def text_cleaner(docs):
    """
    cleans text by removing leading, trailing whitespaces and removing tabs and replacing it with 1 whitespace.
    Args:
    docs: input document with whitespaces, tabs, leading and trailing spaces
    Returns:
    docs: document free of tabs, leading and trailing spaces 
    """
    # for each ith page in docs
    for i in range(len(docs)):
        docs[i].page_content = docs[i].page_content.strip()              # remove leading and traling whitespaces from page_content field and save it back in page_content field
        docs[i].page_content = docs[i].page_content.replace('\t'," ")    # replace tabs with single whitespace from page_content field and save it back in page_content field
    return docs

def sentence_transformer_inputData(chunks):
    """
    generates input data for sentence transformer input from chunks page content
    Args:
    chunks: array of chunks which contain page content and metadata
    Returns:
    inputData: array of chunks page content in format "passage: page_content" 
    """
    inputData = []
    for chunk in chunks:
        inputData.append("passage: "+chunk.page_content)
    return inputData

if __name__ == "__main__":
    # embedding model 'e5-base-v2' loading
    embedding_model = SentenceTransformer('intfloat/e5-base-v2')

    # loading documents
    file_path = "Documentation_demo/book.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    # print(docs[5].page_content[:300])  # extract first 300 characters of content on 6th page 
    # print(docs[5].metadata['author'])  # extract author name from metadata
    # print(docs[5].metadata)            # metadata
    # print(len(docs))                   # number of pages in document

    # clean the document 
    clean_doc = text_cleaner(docs)

    # chunking from text using recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)  # chunk size of 800 characters roughly, overlapping characters 150 for context.
    text_chunks = text_splitter.split_documents(clean_doc)  # returns list of chunks. Each chunk contains cleaned text content, metadata 

    # embedding chunks in batching
    # input_texts = [
    #     'query: how much protein should a female eat',
    #     'query: summit define',
    #     "passage: As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    #     "passage: Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
    # ]

    # get input data ready to be fed to embedding_model with passage prefix.
    input_texts = sentence_transformer_inputData(text_chunks)
    # Embeddings are L2-normalized
    # Dot product == cosine similarity
    # Ready for:
    # manual similarity
    embeddings = embedding_model.encode(input_texts, normalize_embeddings=True)  
    print(embeddings.shape)
    print(embeddings[0].dtype)
    #--------------test code------------------------------------------
    # query = "query: how to find purpose in life?"
    # q_emb = embedding_model.encode(query, normalize_embeddings=True)  
    # scores = embeddings @ q_emb.T    # perform matrix multiplication. .T stands for transpose because, embedding shape = (chunk_count,768), q_emb = (1,768), q_emb.T = (768,1)
    # print(scores)
    # top_k = scores.squeeze().argsort()[::-1][:5]  # remove extra dimension, then sort in descending order and fetch top 5 results.
    # print(top_k)

    # for k in top_k:
    #     print(text_chunks[k])
    #-------------------------------------------------------------------

    # Adding data in FAISS database
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # using flat index which uses Internal product while searching. since we already have normalized index, we can get cosine similarity by directly doing dot products of vectors.
    print("index: ",index)
    index.add(embeddings) # type: ignore
    print("Total vectors in index:", index.ntotal)

    # persisting data for other modules in Disk.
    faiss.write_index(index, "Persistent_data/FAISS.index")  # saving index with data 
    
    # Save embeddings to Pickle
    with open('Persistent_data/embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    # save chunks to pickle
    with open('Persistent_data/text_chunks.pkl', 'wb') as f:
        pickle.dump(text_chunks, f)


