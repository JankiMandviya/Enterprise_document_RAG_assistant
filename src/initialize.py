"""
This module contains file paths, SQlite data base initialization and FAISS initialization.
This module will act as a lower level module which will be used by all retriever, ingestion and chat_history module.

"""
import os
import pytz
import faiss
import time
import pandas as pd
from mistralai import Mistral # type: ignore
from dotenv import load_dotenv
from threading import Lock
from datetime import datetime
from sqlalchemy import DateTime
from sqlalchemy.exc import SQLAlchemyError
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

OLLAMA_URL = "http://localhost:11434/api/generate"
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"   # URL where mistral model is hosted by LM studio
DATABASE_URL = "sqlite:///Persistent_data/chat_history.db"   # Use SQLite, Create (or open) a file named chat_history.db
load_dotenv()
api_key = os.getenv("Mistral_API")
Mistral_client = Mistral(api_key=api_key)
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

    session_title = Column(String, default="New chat" ,nullable=False)  
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


# embedding model 'e5-base-v2' loading
embedding_model = SentenceTransformer('intfloat/e5-base-v2')
index_path = "Persistent_data/FAISS.index"
faiss_lock = Lock()  # lock to prevent two/multiple simultaneous read/write operations on FAISS. It stops FAISS from getting corrupted and multiple overwrites of each other's data.

dimension = 768  # Sentence transformer 'intfloat/e5-base-v2' outputs embedding vector of dimension 768.

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    index = faiss.IndexIDMap(
        faiss.IndexFlatIP(dimension)
    )
    faiss.write_index(index, index_path)

# -----------------------Logging-----------------------

query_csv = "Evaluation/queries.csv"  # Log per query
# stores :
# query_id (unique for evaluation)
# session_id
# user query
# timestamp
# num_documents_in_session
# retrieved_chunks_count (after threshold)
# retrieval_time
# generation_time
# total_time
# final_answer_length (tokens or characters)

retrieval_csv = "Evaluation/retrieval_details.csv"  # Log per retrieved chunk
# stores:
# query_id
# rank (1 = most similar)
# chunk_id
# document_id
# doc_title
# doc_source
# page_number
# similarity_score
# chunk_length (tokens/characters)

evaluation_dataset = "Evaluation/evaluation_dataset.csv" # A ground truth dataset to measure correctness.
# stores:
# question_text
# ground_truth_answer (manual extraction from book)
# source_chunk_ids (optional, but ideal)
# source_page_numbers (optional)

generation_eval = "Evaluation/generation_eval.csv"  # Manual+LLM scoring of LLM answers
# stores:
# Metrics:
#  - Faithfulness — Is answer grounded in retrieved chunks?
#  - Relevance — Does answer address the question?
#  - Hallucination — Does answer include information not in retrieved context?
#  - Answer length — Optional, can detect truncation.

headers_query = ["query_id", "session_id", "user_query","rewritten_query","mode", "timestamp", "num_documents_in_session", "retrieved_chunks_count", "LLM response", "retrieval_time(in s)", "generation_time(in s)", "total_time(in s)", "final_answer_length(in tokens)"]
df_query = pd.DataFrame(columns = headers_query)
# Check if file exists
if os.path.exists(query_csv):
    print("CSV already exists. Headers left unchanged.")
else:
    df_query.to_csv(query_csv, index=False)
    print("CSV created with headers.")


headers_retrieval = ["query_id", "rank", "chunk_id","document_id", "doc_title","doc_source", "page_number","similarity_score","chunk_length"]
df_retrieval = pd.DataFrame(columns = headers_retrieval)
# Check if file exists
if os.path.exists(retrieval_csv):
    print("CSV already exists. Headers left unchanged.")
else:
    df_retrieval.to_csv(retrieval_csv, index=False)
    print("CSV created with headers.")


headers_dataset = ["question_text", "ground_truth_answer","source_page_numbers"]
df_dataset = pd.DataFrame(columns = headers_dataset)
# Check if file exists
if os.path.exists(evaluation_dataset):
    print("CSV already exists. Headers left unchanged.")
else:
    df_dataset.to_csv(evaluation_dataset, index=False)
    print("CSV created with headers.")


headers_generation = ["Faithfulness", "Relevance","Hallucination","Answer_length"]
df_generation = pd.DataFrame(columns = headers_generation)
# Check if file exists
if os.path.exists(generation_eval):
    print("CSV already exists. Headers left unchanged.")
else:
    df_generation.to_csv(generation_eval, index=False)
    print("CSV created with headers.")