import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDF2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Set it in Streamlit Cloud > Secrets.")
    st.stop()

# Function to build the FAISS index from the PDF
def build_index():
    loader = PyPDF2Loader("data/Maja Bridgesysteem.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    return vectorstore

# Load or build the FAISS vector store
@st.cache_resource
def load_vector_store():
    if not os.path.exists("faiss_index"):
        return None
    return FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=api_key))

vector_store = load_vector_store()

if not vector_store:
    st.warning("⚠️ FAISS index not found. Click the button below to build it.")
    if st.button("🚀 Build FAISS Index"):
        with st.spinner("Building index from PDF..."):
            vector_store = build_index()
        st.success("Index built! Please reload the app.")
        st.stop()
else:
    # Create retriever and QA system
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=api_key), retriever=retriever)

    # Chat UI
    query = st.text_input("Ask me something about the Maja Bridge System:")
    if query:
        result = qa.run(query)
        st.markdown("**Answer:**")
        st.write(result)
