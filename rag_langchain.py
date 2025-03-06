import tempfile
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict

# Page Configuration
st.set_page_config(
    page_title="RAG ChatBox", 
    page_icon="🧸", 
    layout="wide"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stTextInput>div>div>input {
    border-radius: 20px;
    border: 1.5px solid #e0e0e0;
}
.stButton>button {
    border-radius: 20px;
    background-color: #4CAF50;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📚 RAG ChatBox: Smart Document Interaction")

# Load environment variables (For local run)
# load_dotenv()
# api_key = os.environ.get("GOOGLE_API_KEY")

# if not api_key:
#     st.error("GOOGLE_API_KEY is not set in the environment variables")
#     st.stop()
    
# Check for Google API Key in Streamlit Secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key not found in Streamlit Secrets. Please configure it.")
    st.stop()    

# Initialize session state
if 'messages' not in st.session_state:                 # Store chat history for display
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:              # Store document embeddings
    st.session_state.vectorstore = None
    

# Sidebar for document upload
with st.sidebar:
    st.header("📤 Document Upload")
    input_mode = st.radio(
        "Select Input Mode",
        ["Website URL", "File Upload"],
        horizontal=True
    )

    documents = []
    
    if input_mode == "Website URL":
        url = st.text_input(
            "Enter a URL:", 
            value="https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/"
        )
        
        if st.button("Process From URL"):
            with st.spinner("Loading web document..."):
                if url:
                    loader = WebBaseLoader(url)
                    documents.extend(loader.load())
                else:
                    st.error("Please provide a valid URL.")
                    st.stop()

    else:
        uploaded_file = st.file_uploader(
            "Upload a file", 
            type=["txt", "pdf"],
            help="Supports .txt and .pdf files"
        )
        
        if st.button("Process From File"):
            with st.spinner("Processing document..."):
                if uploaded_file is not None:
                    file_type = uploaded_file.name.split('.')[-1]
                    
                    if file_type == 'txt':
                        documents.append(uploaded_file.read().decode("utf-8"))
                    
                    elif file_type == 'pdf':
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name

                        try:
                            loader = PyPDFLoader(temp_path)
                            documents.extend(loader.load())
                        finally:
                            os.unlink(temp_path)
                else:
                    st.error("Please upload a file.")
                    st.stop()

    # Document Processing
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        st.session_state.vectorstore = FAISS.from_documents(
            chunks, 
            embeddings
        )
        st.success("RAG system processed successfully!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if st.session_state.vectorstore is not None:
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant that answers questions based on the provided document context. If unsure, say 'I don't know'."),
        ("user", "Question: {question}\nContext: {context}")
    ])
    
    chain = prompt | llm

    if prompt := st.chat_input("Ask me anything about your document..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                retriever = st.session_state.vectorstore.as_retriever()
                docs = retriever.invoke(prompt)
                
                response = chain.invoke({
                    "question": prompt,
                    "context": docs
                })
                
                # Display assistant response
                st.markdown(response.content)
                
                # Optional: Show context expansion
                with st.expander("See Document Context"):
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.markdown(doc.page_content)
                        st.markdown("---")
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response.content
                })
else:
    st.info("📌 Please upload a document in the sidebar to start chatting")

st.markdown('</div>', unsafe_allow_html=True)