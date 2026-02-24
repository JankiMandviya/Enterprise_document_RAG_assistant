"""
This module handles query rewriting and chat memory using SQLite database

This stage includes:
- rewrite query using original user query and chat history
- saves and loads chat history from SQLite database
"""


# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import requests
from datetime import datetime
import initialize
import numpy as np
import pytz
from sqlalchemy.orm import Mapped, mapped_column
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory



# create new session
def create_Session(session_id:str):
    db = next(initialize.get_db())
    session = db.query(initialize.Session).filter(initialize.Session.session_id == session_id).first()
    if not session:   # if session doesn't exist create it and save to db refresh to get session id.
        session = initialize.Session(session_id=session_id)
        db.add(session)
        db.commit()

        session.session_title = f"chat {session.id}" # type: ignore
        db.commit()

    return session_id


def delete_session(session_id:str):
    db = next(initialize.get_db())
    chunk_ids = []

    try:
        session = db.query(initialize.Session).filter_by(session_id=session_id).first()
        if session:
            internal_session_id = session.id
            # To delete embeddings of this session from FAISS, fetch all chunk ids belonging to the documents of this session. 
            chunks = db.query(initialize.DocumentChunk).filter(initialize.DocumentChunk.session_id == internal_session_id).all()
            chunk_ids = [chunk.id for chunk in chunks]
            
            if chunk_ids:
                chunk_ids = np.array(chunk_ids).astype("int64")
                with initialize.faiss_lock:
                    index = initialize.faiss.read_index(initialize.index_path)
                    index.remove_ids(chunk_ids)
                    initialize.faiss.write_index(index, initialize.index_path)
                    print(f"faiss chunks deleted for session : {session_id}")

            db.delete(session) # cascade automatically deletes related documents, messages and document chunks from table Documents, Messages and DocumentChunk respectively.
            db.commit()
        
    except initialize.SQLAlchemyError as e:
        db.rollback()
        print("unexpected error: rolling back to last stage: " , e)

    finally:
        db.close()
        print(f"session deleted: {session_id}")

# Save ONE message to DB
def save_message(session_id: str, role: str, content: str):
    db = next(initialize.get_db())
    try:
        session = db.query(initialize.Session).filter(initialize.Session.session_id == session_id).first()
        if not session:   # if session doesn't exist create it and save to db refresh to get session id.
            session = initialize.Session(session_id=session_id)
            db.add(session)
            db.commit()

            session.session_title = f"chat {session.id}" # type: ignore
            db.commit()

            db.refresh(session)

        db.add(initialize.Message(session_id=session.id, role=role, content=content))
        db.commit()

        if session:  # if session already exists just update the last_updated_time while saving message in this session.
            session.last_updated_time = datetime.now(pytz.timezone('Asia/Kolkata')) # type: ignore
            db.commit()

    except initialize.SQLAlchemyError:
        db.rollback()  # If anything goes wrong in previous operation/transaction, The database goes back to previous clean state.
    finally:
        db.close()

# Function to load chat history for query rewriting of size N conversations. this will not load entire chat history of a session.
def load_session_history(session_id: str, conversations = 5):  # conversations : user+AI pair, eg. if conversations=5. then total messages = 10 (user+AI)
    message_limit = 2*conversations
    db = next(initialize.get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(initialize.Session).filter(initialize.Session.session_id == session_id).first()
        if session:
            messages = (
                db.query(initialize.Message)
                .filter(initialize.Message.session_id == session.id)  # find messages with same session_id as selected session's id.
                .order_by(initialize.Message.id.desc())   # newest first, So chat order stays correct. id is in ascending order auto incrementing. if we use descending, we will get messages in desceding order of message's id.
                .limit(message_limit)                  # limit to N, get top n messages after converting in descending. if messages in db are less than message_limit -> automatically returns all available messages.
                .all()                         # return result as a list.
            )
            print("inside load_session_history")
    
            # reverse to keep the order of messages because we fetched newest first. Build chat_history from selected messages  list just by adding role and message.
            for msg in reversed(messages):
                if  str(msg.role) == "user":
                    chat_history.add_user_message(str(msg.content))
                elif str(msg.role) == "AI":
                    # in chat history, save only clean extracted "answer" from raw response containing context summary, citations, user query, answer.
                    clean_answer = extract_clean_response(msg.content)  
                    chat_history.add_ai_message(str(clean_answer))

    except initialize.SQLAlchemyError:
        pass
    finally:
        db.close()
    return chat_history

def get_all_messages(session_id: str):
    """
    Return list of all messages from a selected session to desplay in UI.
    """
    db = next(initialize.get_db())
    session = db.query(initialize.Session).filter(initialize.Session.session_id == session_id).first()
    if session:
        session_id_int = session.id # type: ignore
    messages = (
        db.query(initialize.Message)
        .filter(initialize.Message.session_id == session_id_int)
        .order_by(initialize.Message.created_at.asc())
        .all()
    )
    return messages

def return_all_sessions():
    """
    return all the existing sessions from session table in descending order of last updates to show in UI
    """
    db = next(initialize.get_db())
    sessions = db.query(initialize.Session).order_by(initialize.Session.last_updated_time.desc()).all()
    return sessions

def return_all_documents(session_id:int):
    """
    Return all documents present in one session to show on UI
    """
    db = next(initialize.get_db())
    docs = (
        db.query(initialize.Document)
        .filter(initialize.Document.session_id == session_id)
        .all()
    )
    return docs

def save_doc_to_table(session_id:str, filename:str):
    """
    save uploaded document in Document table.
    """
    print("inside save_doc function : ", session_id, filename, type(session_id))
    db = next(initialize.get_db())
    session_id_int = db.query(initialize.Session).filter(initialize.Session.session_id == session_id).scalar().id
    new_doc = initialize.Document(session_id=session_id_int, filename = filename, status = "pending")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    doc_id = new_doc.doc_id
    print(session_id_int,doc_id)
    return doc_id,session_id_int

def delete_document(doc_id: int):
    db = next(initialize.get_db())

    try:
        # Fetch document with doc id
        document = db.query(initialize.Document).filter(initialize.Document.doc_id == doc_id).first()

        if not document:
            return

        # Get chunk IDs BEFORE deleting to delete these chunks from faiss
        chunk_ids = [chunk.id for chunk in document.chunks]

        if chunk_ids:
            ids_array = np.array(chunk_ids).astype("int64")

            with initialize.faiss_lock:
                index = initialize.faiss.read_index(initialize.index_path)
                index.remove_ids(ids_array)
                initialize.faiss.write_index(index, initialize.index_path)

        # Delete document (chunks auto-deleted)
        db.delete(document)

        db.commit()

    except initialize.SQLAlchemyError:
        db.rollback()
        print("Error deleting document")

    finally:
        db.close()
    print("deleted document {}".format(doc_id))

# Ensure you save the chat history to the database when needed (not in use for now)
# def save_all_sessions():
#     for session_id, chat_history in store.items():
#         for message in chat_history.messages:
#             save_message(session_id, message["role"], message["content"])

# Example of saving all sessions before exiting the application (not in use for now)
# atexit.register(save_all_sessions)

# rewriting query and saving chats to in memory chat history store.
def rewrite_query(query,chat_history):
    """
    rewrite the original query using chat history

    Args:
    query: original query entered by user
    chat_history: dictionary containing ChatMessageHistory() with session id as key

    Returns:
    rewritten_query:  rewritten query
    """
    if not chat_history.messages:
        return query

    with open('query_rewrite.txt','r') as f:
        rewrite_prompt = f.read()

    t = PromptTemplate(
        input_variables = ["chat_history", "query"],
        template = rewrite_prompt
    )

    prompt = t.format(chat_history = chat_history, query = query)
    print(prompt)
    rewritten_query = CallLLM_Rewrite_query(prompt)
    print(rewritten_query)
    return rewritten_query


def CallLLM_Rewrite_query(rewrite_query_prompt):   # using LM studio
    """
    Call LLM and pass final prompt to it and return text response

    Args:
    final_prompt: final prompt with context and query in string format

    Returns:
    response content
    """

    # LM studio requires following format of input data to model
    payload = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "messages": [
            {
                "role": "user",
                "content": rewrite_query_prompt
            }
        ],
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False
    }

    headers = {"Content-Type": "application/json"}  
    response = requests.post(initialize.LM_STUDIO_URL, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def extract_clean_response(response):
    """
    response contains context summary, query, answer and citations. out of which only answer is stored in chat history. This function extracts answer.
    
    Args:
    response: original LLM response

    returns:
    clean_ans: extracted answer
    """
    start = "2. Answer:"
    end = "3. Citations:"
    if start in response:
        clean_ans = response.split(start,1)[1]   # Only the first occurrence of start is used. split() splits response in two parts. response[0] contains part before start. and response[1] contains part after start.
        if end in clean_ans:
            clean_ans = clean_ans.split(end,1)[0]  # Only the first occurrence of end is used. split() splits response in two parts. clean_ans[0] contains part before end. and clean_ans[1] contains part after end.
        return clean_ans.strip() # remove extra leading and trailing whitespaces or \n.
    else:
        start = "1. Answer:"
        end = "2. Citations:"

        if start in response:
            clean_ans = response.split(start,1)[1]   # Only the first occurrence of start is used. split() splits response in two parts. response[0] contains part before start. and response[1] contains part after start.
            if end in clean_ans:
                clean_ans = clean_ans.split(end,1)[0]  # Only the first occurrence of end is used. split() splits response in two parts. clean_ans[0] contains part before end. and clean_ans[1] contains part after end.
            return clean_ans.strip() # remove extra leading and trailing whitespaces or \n.
        
        return response

# setting up SQLite database for persistent chat history


# save query and response in chat history 'store'  (not in use for now)
# def save_session_history(session_id: str,query,RAW_response):
#     """
#     saves the query and answer in chat history dictionary store

#     Args:
#     session_id : unique session id in string 
#     query: original user query
#     RAW_response: cleaned answer from original LLM response

#     Returns:
#     store[session_id] : returns updated chat history for entered session id
#     """
#     global store
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     store[session_id].add_user_message(query)
#     store[session_id].add_ai_message(RAW_response)
#     return store[session_id]


# Modify the get_session_history function to use the database (not in use for now)
# def get_session_history(session_id: str, ) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = load_session_history(session_id)
#     return store[session_id]

# get chat history of perticular session. (not in use for now)
# def get_session_history(session_id:str):
#     """
#     fetch chat history of particular session

#     Args:
#     session_id: unique session id in string format

#     Returns:
#     store.get(session_id): if session id exists in store as a key, return chat history
#     """
#     return store.get(session_id)