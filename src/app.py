
"""
This module contain User interface for RAG system.
"""
import os
import uuid
import hashlib
import streamlit as st
import chat_history
import ingestion
import Retriever
import Response_generator

st.header(":blue[DocuMind] : Chat with your documents !")

sessions = chat_history.return_all_sessions()
DOCUMENT_FOLDER = "../stored_documents"
os.makedirs(DOCUMENT_FOLDER, exist_ok=True)
db = next(chat_history.get_db())

# read URL of streamlit current chat and check if session with session_id from url exist. if yes return the session id as it is. if not, create session with url's session id 
def create_or_load_session(isNewChat:bool):

    if isNewChat:  # new chat button pressed --> create new session with random uuid as session_id
        new_session_id = str(uuid.uuid4())
        new_session_title = new_session_id
        new_session_id = chat_history.create_Session(new_session_id,new_session_title)
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
                new_session_title = new_session_id
                new_session_id = chat_history.create_Session(new_session_id,new_session_title)
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


# defining session state for mode, list of document ids and session id
# if 'mode' not in st.session_state:
#     st.session_state['mode'] = "User"
# if 'document_ids' not in st.session_state:
#     st.session_state['document_ids'] = []
# if 'session_id' not in st.session_state:
#     st.session_state['session_id'] = uuid.uuid4()

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
    if st.sidebar.button(str(s.session_title)):
        st.query_params["key"] = s.session_id
        st.rerun()

failed_docs = db.query(chat_history.Document).filter(chat_history.Document.status != "completed").all()

for doc in failed_docs:
    chat_history.delete_document(doc.doc_id) # type: ignore

# Take prompt/query from user
prompt = st.chat_input("Ask something...")  
# Let user upload pdfs
uploaded_file = st.file_uploader("Upload document", type=["pdf"], key="pdf_uploader")

current_Session_id = st.query_params["key"]  # current open session in string
# display documents uploaded in current session
current_session_int = db.query(chat_history.Session).filter(chat_history.Session.session_id == current_Session_id).scalar().id # in int
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
        db = next(chat_history.get_db())

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
    query = prompt
    session_history = chat_history.load_session_history(current_Session_id, conversations = 5) # get session history from db
    chat_history.save_message(current_Session_id,current_Session_id, "user", query) # save user message to db
    
    # show message on UI
    with st.chat_message("user"):
        st.markdown(query)

    rewritten_query = chat_history.rewrite_query(query,session_history)  # rewrite the query using LLM by providing chat history, query in prompt.
    
    # searching query in Database
    results = Retriever.search_query(embedding_model = ingestion.embedding_model,session_id = current_session_int, query = rewritten_query, k = 5)  # change it

    # select whether to take relaxed prompt or strict prompt
    mode = Response_generator.promptSelector(rewritten_query,results)

    # build context ready to be fed to LLM
    Final_context = Retriever.build_context(results, debug=True)
    # print(Final_context)

    # build prompt by replacing context and query in selected mode's template
    Final_prompt = Response_generator.promptBuilder(Final_context, rewritten_query, mode)
    # print(Final_prompt)
    
    # Use the prompt to get response from LLM and store original response in RAW_response.
    RAW_response = Response_generator.CallLLM(Final_prompt)

    chat_history.save_message(current_Session_id, current_Session_id ,"AI", RAW_response) # save original raw response in AI message to db
    # show message on UI
    with st.chat_message("AI"):
        st.markdown(RAW_response)
    print("ai response saved")


