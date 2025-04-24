# (everything above stays the same...)

# ✅ Download FAISS index as ZIP to upload to GitHub
import zipfile
import io

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
