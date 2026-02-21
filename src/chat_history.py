"""
This module handles query rewriting and chat memory using SQLite database

This stage includes:
- rewrite query using original user query and chat history
- saves and loads chat history from SQLite database
"""


# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import requests
import pytz
import atexit
import ingestion
import numpy as np
from datetime import datetime
from sqlalchemy import DateTime
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Mapped, mapped_column
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"   # URL where mistral model is hosted by LM studio
DATABASE_URL = "sqlite:///../Persistent_data/chat_history.db"   # Use SQLite, Create (or open) a file named chat_history.db
store = {}  # stores chat_history

# setup for SQLite database
Base = declarative_base()  # The base class for all database models, SQLAlchemy uses it to track tables, all classes must inherit from base class


# Why Do We Need Both ForeignKey AND relationship()?
# 🔹 ForeignKey → Database level
# Defines:
# messages.session_id → sessions.id
# 
# 🔹 relationship() → Python ORM level
# Lets you use:
# session.messages
# message.session

# ForeignKey connects tables.
# relationship connects Python objects.
# back_populates connects two relationships together.

# -------------------------------
# Session Table
# -------------------------------
class Session(Base):
    __tablename__ = "sessions_table"  
    # Name of the SQL table in the database

    id = Column(Integer, primary_key=True)  
    # Internal auto-incrementing primary key used only for database relationships.
    # This is NOT exposed to the frontend.

    session_id = Column(String, unique=True, nullable=False)  
    # Public unique session identifier (random string/UUID).
    # This is the ID used in APIs and frontend.
    # Must be unique and cannot be NULL.

    session_title = Column(String, nullable=False)  
    # Short title used to display the chat/session in the UI sidebar.
    # Not unique because multiple sessions can have similar titles.

    created_time = Column(DateTime, default=datetime.now(pytz.timezone('Asia/Kolkata')), nullable=False)
    # session creation time 

    last_updated_time = Column(
        DateTime,
        default=lambda: datetime.now(pytz.timezone("Asia/Kolkata")),
        nullable=False
    )
    # session update time (by add/delete actions)

    messages = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    # One-to-many relationship:
    # One Session → Many Messages
    # cascade="all, delete-orphan" ensures that
    # when a Session is deleted, all related Messages are also deleted.

    documents = relationship(
        "Document",
        back_populates="session",
        cascade="all, delete-orphan"
    )
    # One-to-many relationship:
    # One Session → Many Documents
    # Deleting a Session will also delete all related Documents.


# -------------------------------
# Message Table
# -------------------------------
class Message(Base):
    __tablename__ = "messages_table"

    id = Column(Integer, primary_key=True)  
    # Unique auto-incrementing ID for each message.

    session_id = Column(
        Integer,
        ForeignKey("sessions_table.id"),
        nullable=False
    )
    # Foreign key referencing Session.id (internal database ID).
    # This links each message to exactly one session.
    # NOT referencing Session.session_id (public ID).

    role = Column(String, nullable=False)  
    # Indicates who sent the message.
    # Expected values: "user" or "AI"

    content = Column(Text, nullable=False)  
    # Stores the full message text (user query or AI response).

    created_at = Column(DateTime, default=datetime.now(pytz.timezone('Asia/Kolkata')), nullable=False)
    session = relationship("Session", back_populates="messages")
    # Many-to-one relationship:
    # Allows access to the parent Session from a Message using message.session


# -------------------------------
# Document Table
# -------------------------------
class Document(Base):
    __tablename__ = "document_table"

    doc_id = Column(Integer, primary_key=True)  
    # Unique auto-incrementing ID for each uploaded document.

    session_id = Column(
        Integer,
        ForeignKey("sessions_table.id"),
        nullable=False
    )
    # Foreign key referencing Session.id (internal database ID).
    # Each document belongs to exactly one session.

    filename = Column(String, nullable=False)  
    # Name of the uploaded file (stored filename or original filename).

    status = Column(String, default="pending", nullable=False) # status tracks current progress of the document embedding generation.
    # "pending" - just received document 
    # "processing" - embedding generation started
    # "completed" -successfully added to database 

    session = relationship("Session", back_populates="documents")
    # Many-to-one relationship:
    # Allows access to the parent Session from a Document using document.session. fetch the related session object for document.

    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan"
    )  # deleting document automatically results in deleting chunks from DocumentChunks table but not in FAISS(take care of it later).  document.chunks accessible 

# -------------------------------
# Document chunk Table
# -------------------------------
class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True)  # chunk id auto incrementing

    session_id = Column(  # session id in integer mapped from session table
        Integer,
        ForeignKey("sessions_table.id"),
        nullable=False
    )

    document_id = Column(      # document id in integer mapped from document table
        Integer,
        ForeignKey("document_table.doc_id"),
        nullable=False
    )

    chunk_text = Column(Text, nullable=False)  # keep single chunk not list of chunks
    doc_title = Column(Text, nullable=False) # keep title of the book/document retrieved from document loader of langchain
    doc_source = Column(Text, nullable=False)  # source of the document in your system retrieved from document loader of langchain
    page_number = Column(Integer, nullable=False) # page number on which this chunk appears in document
    document = relationship("Document", back_populates="chunks")  # create relationship between chunk and parent document. chunk.document accessible.

# create DB
try:
    engine = create_engine(DATABASE_URL)  # Defining the Engine
except Exception as E:
    print(E)

Base.metadata.create_all(engine)  # look for all classes who inherits from Base class, create tables of that class if they don't exist, do nothing if already exists.
SessionLocal = sessionmaker(bind=engine)   # ?


def get_db():   # open a database session
    db = SessionLocal()  
    try:    # DB connection is always closed. to prevent resource leaks
        yield db
    finally:
        db.close()

# create new session
def create_Session(session_id:str, session_title:str):
    db = next(get_db())
    session = db.query(Session).filter(Session.session_id == session_id).first()
    if not session:   # if session doesn't exist create it and save to db refresh to get session id.
        session = Session(session_id=session_id, session_title = session_title)
        db.add(session)
        db.commit()
    return session_id


# Save ONE message to DB
def save_message(session_id: str, session_title:str, role: str, content: str):
    db = next(get_db())
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:   # if session doesn't exist create it and save to db refresh to get session id.
            session = Session(session_id=session_id, session_title = session_title)
            db.add(session)
            db.commit()
            db.refresh(session)
        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()

        if session:  # if session already exists just update the last_updated_time while saving message in this session.
            session.last_updated_time = datetime.now(pytz.timezone('Asia/Kolkata')) # type: ignore
            db.commit()

    except SQLAlchemyError:
        db.rollback()  # If anything goes wrong in previous operation/transaction, The database goes back to previous clean state.
    finally:
        db.close()

# Function to load chat history for query rewriting of size N conversations. this will not load entire chat history of a session.
def load_session_history(session_id: str, conversations = 5):  # conversations : user+AI pair, eg. if conversations=5. then total messages = 10 (user+AI)
    message_limit = 2*conversations
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if session:
            messages = (
                db.query(Message)
                .filter(Message.session_id == session.id)  # find messages with same session_id as selected session's id.
                .order_by(Message.id.desc())   # newest first, So chat order stays correct. id is in ascending order auto incrementing. if we use descending, we will get messages in desceding order of message's id.
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

    except SQLAlchemyError:
        pass
    finally:
        db.close()
    return chat_history

def get_all_messages(session_id: str):
    """
    Return list of all messages from a selected session to desplay in UI.
    """
    db = next(get_db())
    session = db.query(Session).filter(Session.session_id == session_id).first()
    if session:
        session_id_int = session.id # type: ignore
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id_int)
        .order_by(Message.created_at.asc())
        .all()
    )
    return messages

def return_all_sessions():
    """
    return all the existing sessions from session table in descending order of last updates to show in UI
    """
    db = next(get_db())
    sessions = db.query(Session).order_by(Session.last_updated_time.desc()).all()
    return sessions

def return_all_documents(session_id:int):
    """
    Return all documents present in one session to show on UI
    """
    db = next(get_db())
    docs = (
        db.query(Document)
        .filter(Document.session_id == session_id)
        .all()
    )
    return docs

def save_doc_to_table(session_id:str, filename:str):
    """
    save uploaded document in Document table.
    """
    print("inside save_doc function : ", session_id, filename, type(session_id))
    db = next(get_db())
    session_id_int = db.query(Session).filter(Session.session_id == session_id).scalar().id
    new_doc = Document(session_id=session_id_int, filename = filename, status = "pending")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    doc_id = new_doc.doc_id
    print(session_id_int,doc_id)
    return doc_id,session_id_int

def delete_document(doc_id: int):
    db = next(get_db())

    try:
        # Fetch document with doc id
        document = db.query(Document).filter(Document.doc_id == doc_id).first()

        if not document:
            return

        # Get chunk IDs BEFORE deleting to delete these chunks from faiss
        chunk_ids = [chunk.id for chunk in document.chunks]

        if chunk_ids:
            ids_array = np.array(chunk_ids).astype("int64")

            with ingestion.faiss_lock:
                index = ingestion.faiss.read_index(ingestion.index_path)
                index.remove_ids(ids_array)
                ingestion.faiss.write_index(index, ingestion.index_path)

        # Delete document (chunks auto-deleted)
        db.delete(document)

        db.commit()

    except SQLAlchemyError:
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
    response = requests.post(LM_STUDIO_URL, json=payload, headers=headers)
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
    start = "3. Answer:"
    end = "4. Citations:"
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