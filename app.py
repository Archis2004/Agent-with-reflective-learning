# app.py
import io
import streamlit as st
from PIL import Image

from chatbot import prepare_text_docs
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

import google.generativeai as genai

st.set_page_config(page_title="Multimodal Chatbot", layout="wide")
st.title("Multimodal Chatbot (PDF/DOCX RAG & Image Chat)")
st.caption("Upload a text file for RAG or an image for direct Q&A")

# ─── Configure Google Generative AI ──────────────────────────────────────────
try:
    gemini_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=gemini_key)
except Exception:
    st.error("Could not find GOOGLE_API_KEY in .streamlit/secrets.toml")
    st.stop()

# ─── Sidebar: model choice & upload ───────────────────────────────────────────
with st.sidebar:
    st.header("Upload & Configuration")
    chat_model_name = st.selectbox(
        "Gemini Model",
        ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"]
    )
    uploaded_file = st.file_uploader(
        "PDF, DOCX, PNG, JPG or JPEG",
        type=["pdf", "docx", "png", "jpg", "jpeg"]
    )

# ─── Session state defaults ──────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "image_ref" not in st.session_state:
    st.session_state.image_ref = None

# ─── Handle file upload ──────────────────────────────────────────────────────
if uploaded_file:
    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]

    if ext in ("pdf", "docx"):
        # prepare RAG chain
        docs = prepare_text_docs(uploaded_file)
        if not docs:
            st.sidebar.error("Failed to parse document.")
            st.session_state.rag_chain = None
        else:
            emb = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", google_api_key=gemini_key
            )
            vs = FAISS.from_documents(docs, emb)
            llm = ChatGoogleGenerativeAI(
                model=chat_model_name,
                api_key=gemini_key
            )
            st.session_state.rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vs.as_retriever(search_kwargs={"k": 3})
            )
            st.sidebar.success("Document ready for RAG!")
        st.session_state.image_ref = None

    else:
        # prepare image for multimodal Q&A
        img = Image.open(uploaded_file).convert("RGB")
        st.sidebar.image(img, caption="Loaded image")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        # upload into Gemini
        image_ref = genai.upload_image(buf.read())
        st.session_state.image_ref = image_ref
        st.session_state.rag_chain = None
        st.sidebar.success("Image ready for Q&A!")

else:
    st.session_state.rag_chain = None
    st.session_state.image_ref = None

# ─── Render chat history ─────────────────────────────────────────────────────
for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)

# ─── Handle new user question ────────────────────────────────────────────────
if user_q := st.chat_input("Ask a question..."):
    st.session_state.history.append(("user", user_q))
    st.chat_message("user").markdown(user_q)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            # 1) RAG over text
            if st.session_state.rag_chain:
                answer = st.session_state.rag_chain.run(user_q)

            # 2) Multimodal: image + text
            elif st.session_state.image_ref:
                resp = genai.GenerativeModel(chat_model_name).generate_content([
                    st.session_state.image_ref,
                    user_q
                ])
                answer = resp.text

            # 3) Plain chat
            else:
                resp = genai.GenerativeModel(chat_model_name).generate_content(user_q)
                answer = resp.text

        except Exception as e:
            answer = f"Error: {e}"

        st.markdown(answer)
        st.session_state.history.append(("assistant", answer))
