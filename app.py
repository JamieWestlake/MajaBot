import streamlit as st
import os
import joblib
import traceback
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

# UI setup
st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# TF-IDF embedding class that inherits from LangChain Embeddings
class TfidfEmbedding(Embeddings):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0]

    def __call__(self, text):
        return self.embed_query(text)

# Load paths
INDEX_PATH = "data/faiss_index"
VECTORIZER_PATH = os.path.join(INDEX_PATH, "vectorizer.pkl")

@st.cache_resource
def load_vector_store():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    vectorizer = joblib.load(VECTORIZER_PATH)
    embedding = TfidfEmbedding(vectorizer)
    vector_store = FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True)
    return vector_store, embedding

vector_store, embeddings = load_vector_store()

if not vector_store:
    st.error("❌ FAISS index not found. Please upload all 3 files to `data/faiss_index/`.")
    st.stop()

# Dummy document combiner (simple string join)
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

# Create QA chain
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQAWithSourcesChain(
    combine_documents_chain=DummyCombineDocumentsChain(),
    retriever=retriever
)

# Chat UI
query = st.text_input("Ask me something about the Maja Bridge System:")
if query:
    try:
        result = qa.invoke({"question": query})
        st.markdown("**Answer:**")
        st.write(result["answer"] or "No relevant answer found.")
        if result.get("sources"):
            st.markdown("---")
            st.markdown("**Sources:**")
            st.write(result["sources"])
    except Exception as e:
        st.error("💥 Something went wrong:")
        st.code(traceback.format_exc())
