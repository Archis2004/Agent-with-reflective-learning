import os
import tempfile
from typing import List, Optional

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.llms.base import LLM
import google.generativeai as genai
import google.ai.generativelanguage as glm
from pydantic import Field


class GeminiChat(LLM):
    api_key: str = Field(...)
    model_name: str = Field(...)

    def __init__(self, **data):
        super().__init__(**data)
        genai.configure(api_key=self.api_key)

    @property
    def _llm_type(self) -> str:
        return "gemini-chat"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        model = genai.GenerativeModel(self.model_name)

        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            return f"I couldn't get a valid response. The prompt might have been blocked. Details: {e}"

    async def _acall(self, prompt: str, stop: List[str] = None) -> str:
        return self._call(prompt, stop)


def prepare_text_docs(uploaded_file) -> Optional[List[Document]]:
    
    if uploaded_file is None:
        return None

    ext = uploaded_file.name.lower().split(".")[-1]
    
    if ext not in ["pdf", "docx"]:
        return None

    data = uploaded_file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext == "docx":
            loader = Docx2txtLoader(tmp_path)
        
        docs = loader.load()
        if not docs:
            return None

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    except Exception as e:
        print(f"Error processing '{uploaded_file.name}': {e}")
        return None

    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)