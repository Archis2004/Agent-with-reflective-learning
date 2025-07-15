import streamlit as st
from chatbot import prepare_text_docs, GeminiChat
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import google.generativeai as genai
from PIL import Image

st.set_page_config(page_title="Multimodal Chatbot", layout="wide")
st.title("Multimodal Chatbot (PDF/DOCX RAG & Image Chat)")
st.caption("Upload a text file for RAG or an image for direct Q&A")

try:
    gemini_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=gemini_key)
except (KeyError, FileNotFoundError):
    st.error("GOOGLE_API_KEY not found in .streamlit/secrets.toml")
    st.stop()

with st.sidebar:
    st.header("Upload & Configuration")
    
    chat_model_name = st.selectbox(
        "Choose your Gemini model:",
        [
            "gemini-1.5-flash-latest", 
            "gemini-1.5-pro-latest",   
        ]
    )
    
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, PNG, JPG or JPEG",
        type=["pdf", "docx", "png", "jpg", "jpeg"]
    )

if "history" not in st.session_state:
    st.session_state.history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "image_for_chat" not in st.session_state:
    st.session_state.image_for_chat = None

if uploaded_file:
    file_ext = uploaded_file.name.lower().split(".")[-1]
    
    if file_ext in ["pdf", "docx"]:
        st.session_state.image_for_chat = None
        
        st.sidebar.info(f"Processing '{uploaded_file.name}' for RAG...")
        docs = prepare_text_docs(uploaded_file)
        if docs:
            emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
            vectorstore = FAISS.from_documents(docs, emb)
            llm = GeminiChat(api_key=gemini_key, model_name=chat_model_name)
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )
            st.session_state.rag_chain = rag_chain
            st.sidebar.success("Text document processed for RAG!")
        else:
            st.sidebar.error("Failed to process the text document.")
            st.session_state.rag_chain = None

    elif file_ext in ["png", "jpg", "jpeg"]:
        st.session_state.rag_chain = None
        
        st.sidebar.info(f"Processing '{uploaded_file.name}' for chat...")
        image = Image.open(uploaded_file)
        st.session_state.image_for_chat = image
        st.sidebar.image(image, caption="Ready to chat about this image.")
        st.sidebar.success("Image loaded!")

else:
    st.session_state.rag_chain = None
    st.session_state.image_for_chat = None

for role, text in st.session_state.history:
    st.chat_message(role).markdown(text)

if user_q := st.chat_input("Ask a question..."):
    st.session_state.history.append(("user", user_q))
    st.chat_message("user").markdown(user_q)

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        model = genai.GenerativeModel(chat_model_name)
        answer = ""

        if st.session_state.rag_chain:
            try:
                answer = st.session_state.rag_chain.run(user_q)
            except Exception as e:
                answer = f"Error with RAG chain: {e}"
        
        elif st.session_state.image_for_chat:
            try:
                response = model.generate_content([user_q, st.session_state.image_for_chat])
                answer = response.text
            except Exception as e:
                answer = f"Error generating content from image: {e}"

        else:
            try:
                response = model.generate_content(user_q)
                answer = response.text
            except Exception as e:
                answer = f"Error during generation: {e}"
        
        st.markdown(answer)

    st.session_state.history.append(("assistant", answer))