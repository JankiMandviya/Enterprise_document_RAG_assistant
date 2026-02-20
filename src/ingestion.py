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
import os
import numpy as np
import chat_history
from threading import Lock
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# embedding model 'e5-base-v2' loading
embedding_model = SentenceTransformer('intfloat/e5-base-v2')
index_path = "../Persistent_data/FAISS.index"
faiss_lock = Lock()  # lock to prevent two/multiple simultaneous read/write operations on FAISS. It stops FAISS from getting corrupted and multiple overwrites of each other's data.

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

def document_embedding_generator(filepath:str, doc_id:int, session_id:int):
    print("inside document embedding generator")
    db = next(chat_history.get_db())
    chunk_ids = []
    current_doc = db.query(chat_history.Document).filter(chat_history.Document.doc_id == doc_id).first()

    # ----------------------------
    # Check document exists
    # ----------------------------
    current_doc = db.query(chat_history.Document).filter(chat_history.Document.doc_id == doc_id).first()

    if not current_doc:
        print("Document does not exist. Aborting.")
        return
    
    current_doc.status = "processing" # The document is in processing stage. # type: ignore

    # ----------------------------
    # Load & process document
    # ----------------------------

    file_path = filepath
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
    embeddings = embedding_model.encode(input_texts, normalize_embeddings=True).astype("float32")
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

    # Appending document data in FAISS database
    # saving original chunks (cleaned chunk content[not embedding of it], metadata[title, source, page], index at which this chunk is stored in vector DB, session id and document id of this chunk) in SQLite db.
    try:

        # ----------------------------
        # FAISS + DB
        # ----------------------------

        with faiss_lock:

            # Re-check document not deleted
            db.expire_all()
            current_doc = db.query(chat_history.Document).filter(chat_history.Document.doc_id == doc_id).first()

            if not current_doc:
                print("Document deleted mid-process. Aborting safely.")
                return
            
            # load existing index file or create new if doesn't exist
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
            else:
                dimension = embeddings.shape[1]
                index = faiss.IndexIDMap(faiss.IndexFlatIP(dimension)) # using flat index which uses Internal product while searching. since we already have normalized index, we can get cosine similarity by directly doing dot products of vectors.

            # save metadata and chunk content with vector position in FAISS in db first
            for i,chunk in enumerate(text_chunks):
                chunk_entry = chat_history.DocumentChunk(session_id = session_id, document_id = doc_id, chunk_text= chunk.page_content, doc_title = chunk.metadata.get('title',"Title unavailable"), doc_source = chunk.metadata.get('source',"Source unavailable") ,page_number = chunk.metadata.get('page',0))
                db.add(chunk_entry)  # db.add() does NOT save to database. it just asks sqlalchemy to remember the object to add later during commit()
            
            db.commit()  # Nothing is permanently written to SQLite until you commit(). commit() saves all pending changes permanently to the database. If app crashes before commit(), the data is lost. 

            # update FAISS index with this documents chunks. For each embedding give respective unique chunk id as well to map between FAISS chunk embedding and SQLite chunk metadata and content.
            chunks = db.query(chat_history.DocumentChunk).filter(chat_history.DocumentChunk.document_id == doc_id).all()  # get all chunks of document whose id matches with doc_id. returns list of objects of type DocumentChunk
            for chunk in chunks:                    # for each chunk in chunks, find unique primary key chunk id from object and append it in list
                chunk_ids.append(chunk.id)
            chunk_ids = np.array(chunk_ids).astype("int64")  # this ids must be in numpy array.
            index.add_with_ids(embeddings, chunk_ids)  # add new document embeddings in index (already existing index or new index) along with respective chunk id. # type: ignore
            print("new embeddings will be stored from {}".format(index.ntotal))
            
            # persisting data for other modules in Disk.
            faiss.write_index(index, index_path)  # saving index with data   
            
            # ----------------------------
            # Final status update
            # ----------------------------
            db.expire_all()
            current_doc = db.query(chat_history.Document).filter(chat_history.Document.doc_id == doc_id).first()

            if current_doc:
                current_doc.status = "completed" # type: ignore
                db.commit()
            else:
                print("Document deleted before completion update.")

    except chat_history.SQLAlchemyError:
        print("Error adding chunks to SQLite")
        db.rollback()

        # Try marking failed only if doc still exists
        doc_check = db.query(chat_history.Document).filter(chat_history.Document.doc_id == doc_id).first()

        if doc_check:
            doc_check.status = "failed" # type: ignore
            db.commit()

    finally:
        db.close()

    print("returning from document_embedding_generator")

