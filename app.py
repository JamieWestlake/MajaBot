import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# ✅ TF-IDF wrapper for LangChain compatibility
class TfidfEmbedding:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.fitted = False

    def embed_documents(self, texts):
        if not self.fitted:
            self.vectorizer.fit(texts)
            self.fitted = True
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0]

# ✅ Define persistent FAISS path
INDEX_PATH = "data/faiss_index"

# ✅ Load the existing FAISS index
@st.cache_resource
def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        st.error("❌ FAISS index not found. Please upload it to data/faiss_index.")
        st.stop()
    embeddings = TfidfEmbedding()
    return FAISS.load_local(INDEX_PATH, embeddings), embeddings

vector_store, embeddings = load_vector_store()

# ✅ Chat logic
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

query = st.text_input("Ask me something about the Maja Bridge System:")
if query:
    result = qa.run(query)
    st.markdown("**Answer:**")
    st.write(result)
