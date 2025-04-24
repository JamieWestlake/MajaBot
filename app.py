import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import traceback

st.set_page_config(page_title="Bridge Index Builder", layout="wide")
st.title("📚 Build FAISS Index from PDF (No OpenAI)")

INDEX_PATH = "data/faiss_index"
PDF_PATH = "data/Maja Bridgesysteem.pdf"
VECTORIZER_PATH = os.path.join(INDEX_PATH, "vectorizer.pkl")

# Check if PDF exists
if not os.path.exists(PDF_PATH):
    st.error("❌ PDF not found at `data/Maja Bridgesysteem.pdf`. Please upload it to your repo.")
    st.stop()

# Button to trigger index building
if st.button("🚀 Build FAISS Index"):
    with st.spinner("Reading PDF and building index..."):
        try:
            # Load and split PDF
            reader = PdfReader(PDF_PATH)
            docs = [Document(page_content=p.extract_text()) for p in reader.pages if p.extract_text()]
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            texts = [doc.page_content for doc in chunks]
            metadatas = [doc.metadata for doc in chunks]

            # Fit TF-IDF vectorizer
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(texts).toarray()

            # Build FAISS index
            embedding = lambda text: vectorizer.transform([text]).toarray()[0]
            faiss_index = FAISS.from_embeddings(list(zip(texts, X)), embedding)

            # Save all files
            os.makedirs(INDEX_PATH, exist_ok=True)
            faiss_index.save_local(INDEX_PATH)
            joblib.dump(vectorizer, VECTORIZER_PATH)

            st.success("✅ Index built and saved to `data/faiss_index/`. You can now commit the files via GitHub.")
            st.info("📂 Files saved: index.faiss, index.pkl, vectorizer.pkl")

        except Exception as e:
            st.error("❌ Failed to build index.")
            st.code(traceback.format_exc())
