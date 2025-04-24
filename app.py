import streamlit as st
import os
import time
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Bridge Chatbot", layout="wide")
st.title("💬 Chat with Maja Bridge System")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Set it in Streamlit Cloud > Secrets.")
    st.stop()

# ✅ Load PDF and convert to LangChain Documents
def load_pdf(path):
    reader = PdfReader(path)
    return [Document(page_content=page.extract_text()) for page in reader.pages]

# ✅ Build FAISS Index with batch embedding + delay
def build_index():
    docs = load_pdf("data/Maja Bridgesysteem.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    
    all_embeddings = []
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_embeddings = embeddings.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"⚠️ Embedding failed on batch {i//batch_size + 1}: {str(e)}")
            time.sleep(10)
            continue
        time.sleep(1)  # avoid rate limit

    vectorstore = FAISS.from_embeddings(all_embeddings, texts, metadatas)
    vectorstore.save_local("faiss_index")
    return vectorstore

# ✅ Load or fallback to index build prompt
@st.cache_resource
def load_vector_store():
    if not os.path.exists("faiss_index"):
        return None
    return FAISS.load_local("faiss_index", OpenAIEmbeddings(openai_api_key=api_key))

vector_store = load_vector_store()

if not vector_store:
    st.warning("⚠️ FAISS index not found. Click the button below to build it.")
    if st.button("🚀 Build FAISS Index"):
        with st.spinner("Building index from PDF..."):
            vector_store = build_index()
        st.success("Index built! Please reload the app.")
        st.stop()
else:
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=api_key), retriever=retriever)

    query = st.text_input("Ask me something about the Maja Bridge System:")
    if query:
        result = qa.run(query)
        st.markdown("**Answer:**")
        st.write(result)
