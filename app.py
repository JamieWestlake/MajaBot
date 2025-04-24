import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# ✅ Use the same embeddings class used to build your FAISS index
INDEX_PATH = "data/faiss_index"

# ✅ Debug info to verify file presence (optional, safe to remove later)
st.markdown("### 🔍 Debug Info")
st.text(f"Does 'data/faiss_index' exist? {os.path.exists('data/faiss_index')}")
if os.path.exists("data"):
    st.text(f"Files in /data/: {os.listdir('data')}")
if os.path.exists("data/faiss_index"):
    st.text(f"Files in /data/faiss_index/: {os.listdir('data/faiss_index')}")

# ✅ Load FAISS index using HuggingFaceEmbeddings
@st.cache_resource
def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        st.error("❌ FAISS index not found. Please upload it to data/faiss_index.")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # required on Streamlit Cloud
    )

    return FAISS.load_local(INDEX_PATH, embeddings), embeddings

vector_store, embeddings = load_vector_store()

# ✅ Chatbot
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

query = st.text_input("Ask me something about the Maja Bridge System:")
if query:
    result = qa.run(query)
    st.markdown("**Answer:**")
    st.write(result)
