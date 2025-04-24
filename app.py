import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import traceback

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# ✅ TF-IDF embedding class
class TfidfEmbedding:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.fitted = True

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0]

    def __call__(self, text):
        return self.embed_query(text)

# ✅ Paths
INDEX_PATH = "data/faiss_index"
VECTORIZER_PATH = os.path.join(INDEX_PATH, "vectorizer.pkl")

# ✅ Load everything
@st.cache_resource
def load_vector_store():
    if not all([
        os.path.exists(INDEX_PATH),
        os.path.exists(os.path.join(INDEX_PATH, "index.faiss")),
        os.path.exists(os.path.join(INDEX_PATH, "index.pkl")),
        os.path.exists(VECTORIZER_PATH)
    ]):
        return None, None
    vectorizer = joblib.load(VECTORIZER_PATH)
    embedding = TfidfEmbedding(vectorizer)
    return FAISS.load_local(INDEX_PATH, embedding, allow_dangerous_deserialization=True), embedding

vector_store, embeddings = load_vector_store()

if not vector_store:
    st.error("❌ FAISS index not found. Please upload it to data/faiss_index.")
    st.stop()

# ✅ Chat setup
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

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

# ✅ Chat UI
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
        st.error("💥 An error occurred:")
        st.code(traceback.format_exc())
