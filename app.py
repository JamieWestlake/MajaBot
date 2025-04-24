import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import zipfile
import io

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# ✅ Free TF-IDF embedding class (with __call__)
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

    def __call__(self, text):
        return self.embed_query(text)

INDEX_PATH = "data/faiss_index"

# ✅ Load and chunk PDF
def load_pdf(path):
    reader = PdfReader(path)
    return [Document(page_content=page.extract_text()) for page in reader.pages]

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

@st.cache_resource
def load_vector_store():
    if not os.path.exists(INDEX_PATH):
        return None, None
    embeddings = TfidfEmbedding()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True), embeddings

vector_store, embeddings = load_vector_store()

if not vector_store:
    st.warning("⚠️ FAISS index not found. Click below to build it.")
    if st.button("🚀 Build FAISS Index"):
        with st.spinner("Building FAISS index..."):
            vector_store, embeddings = build_index()
        st.success("✅ Index built! Download it and upload to GitHub to make it permanent.")

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
        st.stop()

else:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # ✅ Minimal dummy chain to combine retrieved chunks
    class DummyCombineDocumentsChain(BaseCombineDocumentsChain):
        def combine_docs(self, docs, **kwargs):
            return {"output_text": "\n\n".join(doc.page_content for doc in docs)}

        async def acombine_docs(self, docs, **kwargs):
            return {"output_text": "\n\n".join(doc.page_content for doc in docs)}

        @property
        def input_keys(self):
            return ["documents"]

        @property
        def output_keys(self):
            return ["output_text"]

    qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=DummyCombineDocumentsChain(),
        retriever=retriever
    )

    query = st.text_input("Ask me something about the Maja Bridge System:")
    if query:
        result = qa.invoke({"question": query})
        st.markdown("**Answer:**")
        st.write(result["answer"] or "No relevant answer found.")

        if result.get("sources"):
            st.markdown("---")
            st.markdown("**Sources:**")
            st.write(result["sources"])
