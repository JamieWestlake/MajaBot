import streamlit as st
import os
import zipfile

st.set_page_config(page_title="Download FAISS Files", layout="wide")
st.title("📦 Download FAISS Index Bundle")

ZIP_PATH = "data/faiss_index_bundle.zip"
INDEX_PATH = "data/faiss_index"

# Re-zip the index folder if needed
if os.path.exists(INDEX_PATH):
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(INDEX_PATH):
            file_path = os.path.join(INDEX_PATH, filename)
            zipf.write(file_path, arcname=filename)

    with open(ZIP_PATH, "rb") as f:
        st.download_button(
            label="⬇️ Download FAISS Index Bundle (ZIP)",
            data=f,
            file_name="faiss_index_bundle.zip",
            mime="application/zip"
        )
else:
    st.error("❌ FAISS index folder not found. Build it first.")
