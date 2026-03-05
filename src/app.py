
"""
This module contain User interface for RAG system.
"""
import os
import uuid
import csv
import hashlib
import streamlit as st
import initialize
import chat_history
import ingestion
import Retriever
import Response_generator

st.header(":blue[DocuMind] : Chat with your documents !")

sessions = chat_history.return_all_sessions()
DOCUMENT_FOLDER = "../stored_documents"
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
db = next(initialize.get_db())
query_id = 0

# read URL of streamlit current chat and check if session with session_id from url exist. if yes return the session id as it is. if not, create session with url's session id 
def create_or_load_session(isNewChat:bool):

    if isNewChat:  # new chat button pressed --> create new session with random uuid as session_id
        new_session_id = str(uuid.uuid4())
        new_session_id = chat_history.create_Session(new_session_id)
        st.query_params["key"] = new_session_id
        query_params = st.query_params

        messages = chat_history.get_all_messages(st.query_params["key"])
        for m in messages:
            with st.chat_message(str(m.role)):
                st.markdown(m.content)
        return new_session_id
    
    else:      # only tab reload without new chat button click action --> open currently open session from url.
        query_params = st.query_params

        if query_params:
            st.query_params["key"] = query_params["key"]  # if url had a session_id, after reload set current url's session id as previous one to open the same chat after reload.
            # session_id = chat_history.create_Session(query_params['key'], query_params['key'])              
            messages = chat_history.get_all_messages(st.query_params["key"])

            for m in messages:
                with st.chat_message(str(m.role)):
                    st.markdown(m.content)
            print("hi")
        
        else:   # if url didn't have session_id
            print("hello")
            if not sessions:   # if no previous session exist in sqlite database, create new session with new uuid4 session id.
                new_session_id = str(uuid.uuid4())
                new_session_id = chat_history.create_Session(new_session_id)
                st.query_params["key"] = new_session_id
                query_params = st.query_params

                messages = chat_history.get_all_messages(query_params["key"])

                for m in messages:
                    with st.chat_message(str(m.role)):
                        st.markdown(m.content)

                return new_session_id
            
            else:     # if previous sessions exist in sqlite database, open most recent chat/session
                st.query_params["key"] = sessions[0].session_id
                messages = chat_history.get_all_messages(st.query_params["key"])

                for m in messages:
                    with st.chat_message(str(m.role)):
                        st.markdown(m.content)
                st.rerun()
        return query_params["key"]

def log_queries(session_id:int, user_query:str, rewritten_query:str,mode:str, retrieved_chunks_count:int, LLM_response: str, retrieval_time, generation_time, total_time, final_answer_length:int):
    """
    Logs query_id, session_id, user_query, rewritten_query,timestamp, num_documents_in_session, retrieved_chunks_count, retrieval_time(in s), generation_time(in s), total_time(in s), final_answer_length(in tokens) in queries.csv
    """
    global query_id
    query_id+=1
    documents_in_session = db.query(initialize.Document).filter(initialize.Document.session_id == session_id).all()
    num_documents_in_session = len(documents_in_session)
    timestamp = initialize.datetime.now(initialize.pytz.timezone('Asia/Kolkata'))

    df = [query_id, session_id, user_query,rewritten_query,mode,timestamp, num_documents_in_session, retrieved_chunks_count, LLM_response, retrieval_time, generation_time, total_time, final_answer_length]
    # Append without writing the header again
    with open(initialize.query_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(df)        
    return

def log_chunk_retrieval(results):
    global query_id
    df = []

    for i,chunk in enumerate(results):
        df.append(query_id)
        df.append(i+1)
        df.append(chunk['chunk_id'])
        df.append(chunk['doc_id'])
        df.append(chunk['title'])
        df.append(chunk['source'])
        df.append(chunk['page'])
        df.append(chunk['similarity_score'])
        df.append(chunk['chunk_length'])

        # Append without writing the header again
        with open(initialize.retrieval_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(df) 
        df = []
    return


if "processed_files" not in st.session_state:     # check if "processed_files" exists inside session_state dictionary maintained by streamlit
    st.session_state.processed_files = {}         # if session_state dictionary empty during app initialization, create new processed_files dictionary inside session_state in memory.


# button for new chat/session creation
if st.sidebar.button("➕ New Chat"):
    new_id = create_or_load_session(isNewChat = True)
    st.rerun()

# load or create the session if ctrl+R is pressed, navigation between existing chats, or while loading the application open recent chat
st.sidebar.markdown("**Chat history**")
new_id = create_or_load_session(isNewChat = False)

# for any session from existing chats, if any button is clicked from side bar to navigate to other chats, change URL key of streamlit to open that chat and rerun the application.
for s in sessions:
    col1, col2 = st.sidebar.columns([4, 1])

    with col1:
        if st.sidebar.button(str(s.session_title)):
            st.query_params["key"] = s.session_id
    with col2:
        if st.button("🗑", key=f"delete_{s.id}"):
            chat_history.delete_session(s.session_id) # type: ignore

            # If deleted session was active
            if st.query_params["key"] == s.session_id:
                st.query_params["key"]  = sessions[0].session_id
            st.rerun()

failed_docs = db.query(initialize.Document).filter(initialize.Document.status != "completed").all()

for doc in failed_docs:
    chat_history.delete_document(doc.doc_id) # type: ignore

# Take prompt/query from user
prompt = st.chat_input("Ask something...")  
# Let user upload pdfs
uploaded_file = st.file_uploader("Upload document", type=["pdf"], key="pdf_uploader")

current_Session_id = st.query_params["key"]  # current open session in string
# display documents uploaded in current session
current_session_int = db.query(initialize.Session).filter(initialize.Session.session_id == current_Session_id).scalar().id # in int
docs = chat_history.return_all_documents(current_session_int) # return all documents present in current session from SQLite Document table

# display existing session's documents in SQLite whose embeddings are present in FAISS
if docs:
    st.markdown("### Uploaded Documents")

    for doc in docs:
        st.write(f"📄 {doc.filename}")

#----------------------{ if a new file is uploaded }-------------------------

# st.session_state = {
#     "processed_files": {
#         "current_session_id": set()
#     }
# }

# check if current session id exists as key in processed_files dictionary.
# if no -> create new key with current_session_id and make a set for it to store document hash to prevent duplication of documents in vector DB and SQlite.
if current_Session_id not in st.session_state.processed_files:
    st.session_state.processed_files[current_Session_id] = set()

if uploaded_file:
    # create unique hash for a document(not document id).
    # getvalue() : returns raw bytes of file
    # md5() : creates a fingerprint
    # haxdigest() : converts to string
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    # if the file_hash doesn't exist in set(), process this new document and then add the hash to the set.
    if file_hash not in st.session_state.processed_files[current_Session_id]:
        db = next(initialize.get_db())

        # Save file to disk
        file_path = os.path.join(DOCUMENT_FOLDER, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # get session in which the document is uploaded and save this document in Document table
        doc_Session = st.query_params["key"]
        doc_id,session_id = chat_history.save_doc_to_table(doc_Session, uploaded_file.name)
        
        # inject embeddings of this document in DocumentChunk table
        ingestion.document_embedding_generator(file_path, doc_id, session_id) # type: ignore

        st.session_state.processed_files[current_Session_id].add(file_hash)
        st.success("Document processed Successfully.")

        # show document below uploader in UI
        docs = chat_history.return_all_documents(session_id)
        st.markdown("### Uploaded Documents")

        for doc in docs:
            st.write(f"📄 {doc.filename}")
    else:
        st.error("document already exists")
#----------------------{ if user enters a prompt/query }-----------------------------
if prompt:
    t3  = initialize.time.time() # start time when query is entered through UI

    query = prompt
    session_history = chat_history.load_session_history(current_Session_id, conversations = 5) # get session history from db
    chat_history.save_message(current_Session_id, "user", query) # save user message to db
    
    # show message on UI
    with st.chat_message("user"):
        st.markdown(query)

    rewritten_query = chat_history.rewrite_query(query,session_history)  # rewrite the query using LLM by providing chat history, query in prompt.
    
    # searching query in Database
    t1 = initialize.time.time()
    results = Retriever.search_query(embedding_model = initialize.embedding_model,session_id = current_session_int, query = rewritten_query, k = 5)  # change it
    retrieval_time = initialize.time.time() - t1   # time taken to retrieve results chunks from faiss.
    retrieved_chunks_count = len(results)

    if retrieved_chunks_count != 0:
        # select whether to take relaxed prompt or strict prompt
        mode = Response_generator.promptSelector(rewritten_query,results)

        # build context ready to be fed to LLM
        Final_context = Retriever.build_context(results, debug=True)
        # print(Final_context)

        # build prompt by replacing context and query in selected mode's template
        Final_prompt = Response_generator.promptBuilder(Final_context, rewritten_query, mode)
        # print(Final_prompt)
        
        # Use the prompt to get response from LLM and store original response in RAW_response.
        t2 = initialize.time.time()
        RAW_response = Response_generator.CallLLM(Final_prompt)
        generation_time = initialize.time.time() - t2
    
    else:
        mode = "strict"
        RAW_response = "I don't know. No information found for the entered query."

    chat_history.save_message(current_Session_id, "AI", RAW_response) # save original raw response in AI message to db
    # show message on UI
    with st.chat_message("AI"):
        st.markdown(RAW_response)
    
    total_time = initialize.time.time() - t3  # time taken to process entire query and display results on UI
    final_answer_length = 100
    log_queries(current_session_int, query, rewritten_query, mode, retrieved_chunks_count, RAW_response, retrieval_time, generation_time, total_time, final_answer_length)
    log_chunk_retrieval(results)

    print("ai response saved")


