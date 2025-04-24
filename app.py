import streamlit as st
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import os

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

# Load vector store
@st.cache_resource
def load_vector_store():
    return FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")))

vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

# Chat UI
query = st.text_input("Ask me something about the Maja Bridge System:")
if query:
    result = qa.run(query)
    st.markdown("**Answer:**")
    st.write(result)
