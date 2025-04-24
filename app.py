import streamlit as st
import os
import joblib
import traceback
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from sklearn.feature_extraction.text import TfidfVectorizer

# UI setup
st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# Embedding class using TF-IDF vectorizer
class TfidfEmbedding(Embeddings):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0]

    def __call__(self, text):
        return self.embed_query(text)

# Simple combiner that joins chunks
class DummyCombineDocumentsChain(BaseCombineDocumentsChain):
    def combine_docs(self, docs, **kwargs):
        if not docs:
            return {"output_text": "No relevant documents found."}
        return {"output_text": "\n\n".join(doc.page_content for doc in docs)}

    async def acombine_docs(self, docs, **kwargs):
        if not docs:
            return {"output_text": "No relevant documents found."}
        return {"output_text": "\n\n".join(doc.page_content for doc in docs)}

    @property
    def input_keys(self):
        return ["documents"]

    @property
    def output_keys(self):
        return ["output_text"]

    def __call__(self, inputs, **kwargs):
        documents = inputs.get("documents")
        return self.combine_docs(documents, **kwargs)

# File paths
INDEX_PATH = "data/faiss_index"
VECTORIZER_PATH = os.path.join(INDEX_PATH, "vectorizer.pkl")

# Load the FAISS vector store and TF-IDF vectorizer
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
    st.error("❌ FAISS index not found. Please upload `index.faiss`, `index.pkl`, and `vectorizer.pkl` to `data/faiss_index/`.")
    st.stop()

retriever = vector_store.as_retriever(search_kwargs={"k": 4})
combiner = DummyCombineDocumentsChain()

# Chat UI
query = st.text_input("Ask me something about the Maja Bridge System:")
if query:
    try:
        docs = retriever.get_relevant_documents(query)
        result = combiner(inputs={"documents": docs})

        st.markdown("**Answer:**")
        st.write(result["output_text"])

        # Show retrieved context (optional)
        if docs:
            st.markdown("---")
            st.markdown("**Matched Snippets:**")
            for i, doc in enumerate(docs):
                st.markdown(f"**Match {i+1}:**")
                st.write(doc.page_content[:300])

    except Exception as e:
        st.error("💥 Something went wrong while answering your question:")
        st.code(traceback.format_exc())

# Optional: Debug section to test search without UI
with st.expander("🧪 Debug: Test FAISS index"):
    if st.button("Run test query"):
        try:
            docs = vector_store.similarity_search("opening bid with 5-card major", k=3)
            st.markdown("**Top Matches:**")
            for i, doc in enumerate(docs):
                st.markdown(f"**Match {i+1}:**")
                st.write(doc.page_content[:300])
        except Exception as e:
            st.error("❌ Test query failed.")
            st.code(traceback.format_exc())
