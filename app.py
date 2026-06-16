import streamlit as st
import os
import tempfile
import time
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore # or InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

#STEPS
#document loading
#splitting 
#embeddings and vector stores


# Load environment variables from .env file
load_dotenv()

# App Configuration
st.set_page_config(page_title="PDFizer - Chat with PDFs", layout="centered", page_icon="📄")

st.title("📄 PDFizer")
st.subheader("RAG Pipeline: Chat with your PDF document")

# Function to get embeddings (cached to prevent reloading)
@st.cache_resource
def get_embeddings(key):
    # Using Google Embeddings since your API key works perfectly
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=key)

# Check for API Key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key or api_key == "":
    st.warning("Please set your GOOGLE_API_KEY in the `.env` file to enable the LLM.")
    st.info(" Google API key.")
    st.stop()

# Initialize LLM (Gemini 3.1 Flash Lite)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=api_key)
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

embeddings = get_embeddings(api_key)

uploaded_file = st.file_uploader("Upload a PDF document to begin", type="pdf")

if uploaded_file is not None:
    # We need to save the file to disk temporarily because PyPDFLoader expects a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    if "last_widget_file" not in st.session_state:
        st.session_state.last_widget_file = None
        
    is_new_upload = (uploaded_file.name != st.session_state.last_widget_file)
    if is_new_upload:
        st.session_state.last_widget_file = uploaded_file.name

    if "current_file" in st.session_state and not is_new_upload and st.session_state.current_file != uploaded_file.name:
        st.info(f"💡 You are currently viewing history for document: **{st.session_state.current_file}**")
    else:
        st.success(f"Successfully uploaded: {uploaded_file.name}")
    
    if "vectorstore_cache" not in st.session_state:
        st.session_state.vectorstore_cache = {}

    if ("vectorstore" not in st.session_state or is_new_upload) and uploaded_file.name in st.session_state.vectorstore_cache:
         cached = st.session_state.vectorstore_cache[uploaded_file.name]
         st.session_state.vectorstore = cached["vectorstore"]
         st.session_state.current_file = cached["file_name"]
         st.session_state.doc_stats = cached["doc_stats"]
         if is_new_upload:
             st.info(f"⚡ Restored **{uploaded_file.name}** from memory cache! (Saved API quota)")

    # Process the PDF if the VectorStore isn't created OR if a new file was uploaded
    if "vectorstore" not in st.session_state or (is_new_upload and uploaded_file.name not in st.session_state.vectorstore_cache):
        # Create an animated status dashboard
        with st.status("🚀 **RAG Pipeline Dashboard: Processing PDF...**", expanded=True) as status:
            progress_bar = st.progress(0)
            time.sleep(0.5)
            
            st.write("📄 **Phase 1: Document Ingestion** - Extracting text from PDF...")
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            st.write(f"✅ Extracted {len(docs)} pages.")
            progress_bar.progress(33)
            time.sleep(0.5)

            st.write("✂️ **Phase 2: Text Chunking** - Breaking document into smaller segments...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200,
                length_function=len
            )
            splits = text_splitter.split_documents(docs)
            st.write(f"✅ Split document into {len(splits)} chunks.")
            progress_bar.progress(66)
            time.sleep(0.5)

            st.write("🧠 **Phase 3: Vector Indexing** - Generating embeddings and storing in database...")
            # Create an SKLearn vector database (bypasses FAISS compilation bugs and ChromaDB bugs!)
            vectorstore = SKLearnVectorStore.from_documents(
                documents=splits, 
                embedding=embeddings
            )
            st.write("✅ Vector database fully indexed.")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Save current chat to history BEFORE overwriting the global session state!
            if "messages" in st.session_state and len(st.session_state.messages) > 0 and st.session_state.get("current_session_idx", -1) == -1:
                title = st.session_state.messages[0]["content"][:25] + "..."
                st.session_state.setdefault("past_sessions", []).insert(0, {
                    "title": title, 
                    "messages": list(st.session_state.messages),
                    "vectorstore": st.session_state.get("vectorstore"),
                    "file_name": st.session_state.get("current_file"),
                    "doc_stats": st.session_state.get("doc_stats")
                })
            st.session_state.messages = [] 
            st.session_state.current_session_idx = -1

            # NOW we can safely store the new document in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.current_file = uploaded_file.name
            
            st.session_state.doc_stats = {
                "pages": len(docs),
                "chunks": len(splits)
            }
            
            # Save to cache so we don't hit the API rate limit if they click "New Chat" with the same file
            st.session_state.vectorstore_cache[uploaded_file.name] = {
                "vectorstore": vectorstore,
                "file_name": uploaded_file.name,
                "doc_stats": st.session_state.doc_stats
            }
            
            time.sleep(0.5)
            progress_bar.empty()
            status.update(label="✨ **Pipeline Complete! Your document is indexed and ready.**", state="complete", expanded=False)

    vectorstore = st.session_state.vectorstore
    
    # Display Document Statistics Dashboard
    if "doc_stats" in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("📄 Pages Extracted", st.session_state.doc_stats["pages"])
        col2.metric("✂️ Data Chunks", st.session_state.doc_stats["chunks"])
        col3.metric("🗄️ Index Status", "Active")
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Configure the Retriever
    k_value = min(50, st.session_state.doc_stats["chunks"]) if "doc_stats" in st.session_state else 50
    retriever = vectorstore.as_retriever(search_kwargs={"k": k_value}) # Retrieve up to 50 most similar chunks

    # 4. RAG Pipeline Setup
    # Create the prompt template
    system_prompt_template = (
        "You are a highly intelligent and helpful document assistant. Use the following pieces of retrieved context "
        "to answer the user's question. \n"
        "If the document contains specific data, facts, or figures, and the user asks for an interpretation or analysis "
        "of that data (e.g., 'is this normal?', 'what does this mean?', 'is this a good value?'), you MUST use your "
        "general knowledge across all fields (finance, science, medical, general, etc.) to explain it. Clearly mention "
        "that your interpretation is based on general knowledge and is not explicitly stated in the document.\n"
        "If the core factual answer is completely missing from the context, say that you don't know based on the document.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}\n\n"
        "Answer:"
    )

    prompt = PromptTemplate.from_template(system_prompt_template)

    st.divider()

    # Initialize chat history arrays if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "past_sessions" not in st.session_state:
        st.session_state.past_sessions = []
    if "current_session_idx" not in st.session_state:
        st.session_state.current_session_idx = -1

    # Sidebar controls
    with st.sidebar:
        st.header("💬 Chat History")
        
        if st.button("➕ New Chat", use_container_width=True):
            if len(st.session_state.messages) > 0 and st.session_state.current_session_idx == -1:
                title = st.session_state.messages[0]["content"][:25] + "..."
                st.session_state.past_sessions.insert(0, {
                    "title": title, 
                    "messages": list(st.session_state.messages),
                    "vectorstore": st.session_state.get("vectorstore"),
                    "file_name": st.session_state.get("current_file"),
                    "doc_stats": st.session_state.get("doc_stats")
                })
            st.session_state.messages = []
            st.session_state.current_session_idx = -1
            
            # Reset active vectorstore to force loading the current file in the dropzone
            if "vectorstore" in st.session_state:
                del st.session_state.vectorstore
            if "current_file" in st.session_state:
                del st.session_state.current_file
            st.rerun()
            
        if len(st.session_state.past_sessions) > 0:
            st.markdown("### Past Chats")
            for idx, session in enumerate(st.session_state.past_sessions):
                if st.button(f"📄 {session['title']}", key=f"session_{idx}", use_container_width=True):
                    # Save current if active
                    if len(st.session_state.messages) > 0 and st.session_state.current_session_idx == -1:
                        title = st.session_state.messages[0]["content"][:25] + "..."
                        st.session_state.past_sessions.insert(0, {
                            "title": title, 
                            "messages": list(st.session_state.messages),
                            "vectorstore": st.session_state.get("vectorstore"),
                            "file_name": st.session_state.get("current_file"),
                            "doc_stats": st.session_state.get("doc_stats")
                        })
                        st.session_state.current_session_idx = idx + 1
                    else:
                        st.session_state.current_session_idx = idx
                        
                    session_data = st.session_state.past_sessions[st.session_state.current_session_idx]
                    st.session_state.messages = list(session_data["messages"])
                    if session_data.get("vectorstore"):
                        st.session_state.vectorstore = session_data["vectorstore"]
                        st.session_state.current_file = session_data["file_name"]
                        st.session_state.doc_stats = session_data["doc_stats"]
                    st.rerun()

        st.divider()
        st.header("⚙️ Settings")
        dark_mode = st.toggle("🌙 Dark Mode", value=True)
        
        if dark_mode:
            st.markdown("""
            <style>
                [data-testid="stAppViewContainer"] { background-color: #0e1117; color: #fafafa; }
                [data-testid="stSidebar"] { background-color: #262730; }
                [data-testid="stHeader"] { background-color: #0e1117; }
                .stMarkdown, .stText, h1, h2, h3, h4, label { color: #fafafa !important; }
                [data-testid="stChatMessage"] { background-color: #262730 !important; }
            </style>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <style>
                [data-testid="stAppViewContainer"] { background-color: #ffffff; color: #31333F; }
                [data-testid="stSidebar"] { background-color: #f0f2f6; }
                [data-testid="stHeader"] { background-color: #ffffff; }
                .stMarkdown, .stText, h1, h2, h3, h4, label { color: #31333F !important; }
                [data-testid="stChatMessage"] { background-color: #f0f2f6 !important; }
            </style>
            """, unsafe_allow_html=True)

    # (Chat arrays initialized above)
    # Display existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "context" in message and message["context"]:
                with st.expander("View Source Context"):
                    for i, doc in enumerate(message["context"]):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**\n> {doc.page_content}")

    # Query Input
    user_query = st.chat_input("Ask a question about the uploaded document...")

    if user_query:
        # Save user message to history and display it
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document and generating answer..."):
                # Pure Python RAG implementation (bypassing Langchain's broken imports!)
                source_documents = retriever.invoke(user_query)
                context_text = "\n\n".join([f"Source (Page {doc.metadata.get('page', 'Unknown')}):\n{doc.page_content}" for doc in source_documents])
                
                final_prompt = prompt.format(context=context_text, input=user_query)
                answer = llm.invoke(final_prompt).content

                st.write(answer)
                
                with st.expander("View Source Context"):
                    st.write("The following chunks were retrieved from the document to generate this answer:")
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**\n> {doc.page_content}")
                
        # Save assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "context": source_documents
        })
        
        # If we are viewing a past session, update the session's message list directly!
        if st.session_state.get("current_session_idx", -1) != -1:
            st.session_state.past_sessions[st.session_state.current_session_idx]["messages"] = list(st.session_state.messages)
