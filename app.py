import streamlit as st
import os
import time
import asyncio
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["STREAMLIT_ENV"] = "production"

# 🔧 Fix PyTorch async loop error on reload
asyncio.set_event_loop(asyncio.new_event_loop())

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# 🔁 Auto-reload after index build
if st.session_state.get("index_built"):
    st.session_state.index_built = False
    st.experimental_rerun()

# ✅ Define path to persist FAISS index
INDEX_PATH = "data/faiss_index"

# ✅ Load PDF and convert to LangChain Documents
def load_pdf(path):
    reader = PdfReader(path)
    return [Document(page_content=page.extract_text()) for page in reader.pages]

# ✅ Build FAISS Index with local embeddings (safe device setting)
def build_index():
    docs = load_pdf("data/Maja Bridgesysteem.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]

    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore

# ✅ Load or fallback to index build prompt (model loaded inside call)
@st.cache_resource
def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        return None
    return FAISS.load_local(
        INDEX_PATH,
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
    )

vector_store = load_vector_store()

if not vector_store:
    st.warning("⚠️ FAISS index not found. Click the button below to build it.")
    if st.button("🚀 Build FAISS Index"):
        with st.spinner("Building index from PDF using local model..."):
            vector_store = build_index()
        st.success("Index built! Reloading now...")
        st.session_state.index_built = True
        st.stop()
else:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

    query = st.text_input("Ask me something about the Maja Bridge System:")
    if query:
        result = qa.run(query)
        st.markdown("**Answer:**")
        st.write(result)
