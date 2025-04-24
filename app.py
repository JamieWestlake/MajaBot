import streamlit as st
import os
import zipfile
import joblib
import shutil
import tempfile
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Build & Download FAISS Index", layout="wide")
st.title("📚 Build FAISS Index from PDF (TF-IDF, no OpenAI)")

INDEX_DIR = "data/faiss_index"
ZIP_OUTPUT = "faiss_index_bundle.zip"
PDF_PATH = "data/Maja Bridgesysteem.pdf"

if not os.path.exists(PDF_PATH):
    st.error("❌ PDF not found at `data/Maja Bridgesysteem.pdf`")
    st.stop()

if st.button("🚀 Build and Export Index"):
    with st.spinner("Reading and processing PDF..."):
        reader = PdfReader(PDF_PATH)
        docs = [Document(page_content=p.extract_text()) for p in reader.pages if p.extract_text()]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        texts = [doc.page_content for doc in chunks]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts).toarray()
        embed = lambda text: vectorizer.transform([text]).toarray()[0]
        faiss_index = FAISS.from_embeddings(list(zip(texts, X)), embed)

        # Save files
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss_index.save_local(INDEX_DIR)
        joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.pkl"))

        # Zip for download
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, ZIP_OUTPUT)
            with zipfile.ZipFile(zip_path, "w") as zipf:
                for f in ["index.faiss", "index.pkl", "vectorizer.pkl"]:
                    file_path = os.path.join(INDEX_DIR, f)
                    zipf.write(file_path, arcname=f)

            with open(zip_path, "rb") as f:
                st.success("✅ Index built successfully!")
                st.download_button(
                    label="⬇️ Download FAISS Index Bundle (ZIP)",
                    data=f,
                    file_name=ZIP_OUTPUT,
                    mime="application/zip"
                )
