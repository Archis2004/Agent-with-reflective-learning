# doc_processor.py
import os
import tempfile
from typing import List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def prepare_text_docs(uploaded_file) -> Optional[List[Document]]:
    """
    Load a PDF or DOCX, split it into 1,000-character chunks with a 200-character overlap,
    and return a list of Document objects.
    """
    if not uploaded_file:
        return None

    # Get the file extension
    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
    if ext not in ("pdf", "docx"):
        return None

    # Read the file content
    data = uploaded_file.getvalue()
    
    # Create a temporary file to store the uploaded content because LangChain loaders need a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        # Select the appropriate loader based on the file extension
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        else:  # for docx
            loader = Docx2txtLoader(tmp_path)

        docs = loader.load()
        if not docs:
            return None

        # Initialize the text splitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # Split the documents into smaller chunks
        return splitter.split_documents(docs)

    except Exception as e:
        # Print any errors that occur during document processing
        print(f"[prepare_text_docs] Error: {e}")
        return None

    finally:
        # Clean up and remove the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
