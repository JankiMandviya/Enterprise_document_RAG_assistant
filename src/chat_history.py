# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import requests
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from langchain_core.prompts import PromptTemplate
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory

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

# session table:
class Session(Base):
    __tablename__ = "sessions_table" # This defines a SQL table named sessions.
    id = Column(Integer, primary_key=True)  # auto incrementing integer. this is internal database id. we don't use it directly in code.
    session_id = Column(String, unique=True, nullable=False)  # session ids stored here. it must be unique and can't be null/missing.
    messages = relationship("Message", back_populates="session")  # back_populates connects two relationship() fields together. One session → many messages. Allows Session.messages and returns list of Message objects
    
# Message table. contains human query and AI response    
class Message(Base):
    __tablename__ = "messages_table"
    id = Column(Integer, primary_key=True) # unique message id
    session_id = Column(Integer, ForeignKey("sessions_table.id"), nullable=False)  # Each message belongs to one session, messages.session_id points to sessions.id
    role = Column(String, nullable=False)  # role would be either "user" or "AI"
    content = Column(Text, nullable=False)  # this column stores actual message
    session = relationship("Session", back_populates="messages") # access to session table through messages. allows Message.session and returns Session object

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

# Save ONE message to DB
def save_message(session_id: str, role: str, content: str):
    db = next(get_db())
    try:
        print("5")
        session = db.query(Session).filter(Session.session_id == session_id).first()
        if not session:   # if session doesn't exist create it and save to db refresh to get session id.
            session = Session(session_id=session_id)
            db.add(session)
            db.commit()
            db.refresh(session)
            print("6")
        db.add(Message(session_id=session.id, role=role, content=content))
        db.commit()
        print("7")
    except SQLAlchemyError:
        db.rollback()
        print("8")
    finally:
        db.close()
        print("9")

# Function to load chat history
def load_session_history(session_id: str, conversations = 5):  # conversations : user+AI pair, eg. if conversations=5. then total messages = 10 (user+AI)
    print("11")
    message_limit = 2*conversations
    db = next(get_db())
    chat_history = ChatMessageHistory()
    try:
        print("12")
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
                    chat_history.add_ai_message(str(msg.content))
            print(chat_history)
            print("13")
    except SQLAlchemyError:
        pass
    finally:
        db.close()
        print("14")
    return chat_history

# Modify the get_session_history function to use the database (not in use for now)
# def get_session_history(session_id: str, ) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = load_session_history(session_id)
#     return store[session_id]

# Ensure you save the chat history to the database when needed
def save_all_sessions():
    for session_id, chat_history in store.items():
        for message in chat_history.messages:
            save_message(session_id, message["role"], message["content"])

# Example of saving all sessions before exiting the application
import atexit
atexit.register(save_all_sessions)

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
    print("16")
    if not chat_history.messages:
        return query
    print("17")

    with open('query_rewrite.txt','r') as f:
        rewrite_prompt = f.read()

    t = PromptTemplate(
        input_variables = ["chat_history", "query"],
        template = rewrite_prompt
    )

    prompt = t.format(chat_history = chat_history, query = query)
    print(prompt)
    print("18")
    rewritten_query = CallLLM_Rewrite_query(prompt)
    print("20")
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
    print("19")
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
def save_session_history(session_id: str,query,RAW_response):
    """
    saves the query and answer in chat history dictionary store

    Args:
    session_id : unique session id in string 
    query: original user query
    RAW_response: cleaned answer from original LLM response

    Returns:
    store[session_id] : returns updated chat history for entered session id
    """
    global store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    store[session_id].add_user_message(query)
    store[session_id].add_ai_message(RAW_response)
    return store[session_id]

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