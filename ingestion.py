"""
This file contains document ingestion and chunking modules.

This stage includes:
1.	Loads documents
2.	Extracts raw text
3.	Splits text into chunks
4.	Attaches metadata (source, page)

5. converts chunks into normalized embeddings using sentence transformer
6. 
"""

# import dependencies
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

# embedding model 'e5-base-v2' loading
embedding_model = SentenceTransformer('intfloat/e5-base-v2')

# loading documents
file_path = "book.pdf"
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

#--------------test code------------------------------------------
# query = "query: how to find purpose in life?"
# q_emb = embedding_model.encode(query, normalize_embeddings=True)  
# scores = embeddings @ q_emb.T
# print(scores)
# top_k = scores.squeeze().argsort()[::-1][:5]
# print(top_k)

# for k in top_k:
#     print(text_chunks[k])

#-------------------------------------------------------------------

