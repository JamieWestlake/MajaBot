import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import zipfile
import io

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# ✅ TF-IDF wrapper compatible with LangChain
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

INDEX_PATH = "data/faiss_index"

# ✅ Load and chunk PDF into LangChain documents
def load_pdf(path):
    reader = PdfReader(path)
    return [Document(page_content=page.extract_text()) for page in reader.pages]

# ✅ Build the FAISS index using TF-IDF embeddings
def build_index():
    docs = load_pdf("data/Maja Bridgesysteem.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    embeddings = TfidfEmbedding()
    faiss_index = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    faiss_index.save_local(INDEX_PATH)
    return faiss_index, embeddings

# ✅ Load the index from disk, if it exists
@st.cache_resource
def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        return None, None
    embeddings = TfidfEmbedding()
    return FAISS.load_local(INDEX_PATH, embeddings), embeddings

# ✅ Load or build index
vector_store, embeddings = load_vector_store()

if not vector_store:
    st.warning("⚠️ FAISS index not found. Click below to build it.")
    if st.button("🚀 Build FAISS Index"):
        with st.spinner("Processing PDF and building index..."):
            vector_store, embeddings = build_index()
        st.success("Index built! Please refresh the app.")
        st.stop()
else:
    # ✅ Chat interface
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

    query = st.text_input("Ask me something about the Maja Bridge System:")
    if query:
        result = qa.run(query)
        st.markdown("**Answer:**")
        st.write(result)

    # ✅ Download FAISS index to persist it
    def zip_faiss_index(folder_path):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, start=folder_path)
                    zip_file.write(filepath, arcname)
        return zip_buffer.getvalue()

    if os.path.exists(INDEX_PATH):
        zip_bytes = zip_faiss_index(INDEX_PATH)
        st.download_button(
            label="⬇️ Download FAISS Index (ZIP)",
            data=zip_bytes,
            file_name="faiss_index.zip",
            mime="application/zip"
        )
