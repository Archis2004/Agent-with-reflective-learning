# chatbot.py
import os
import tempfile
from typing import List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def prepare_text_docs(uploaded_file) -> Optional[List[Document]]:
    """
    Load a PDF or DOCX, split into 1,000-char chunks with 200-char overlap.
    """
    if not uploaded_file:
        return None

    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
    if ext not in ("pdf", "docx"):
        return None

    data = uploaded_file.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = Docx2txtLoader(tmp_path)

        docs = loader.load()
        if not docs:
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    except Exception as e:
        print(f"[prepare_text_docs] Error: {e}")
        return None

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)