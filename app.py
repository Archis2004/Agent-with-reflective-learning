import os
import streamlit as st
from chatbot import prepare_file_data, chat_with_gemini
import google.generativeai as genai 

st.set_page_config(page_title="Multimodal Chatbot", layout="wide")
st.title("Multimodal Chatbot")
st.caption("Upload PDF, DOCX, JPG, PNG, or JPEG files and chat!")

PREFERRED_MODELS = [
    "models/gemini-1.5-flash-latest",
    "models/gemini-1.5-pro-latest",   
    "models/gemini-pro-vision",       
    "models/gemini-pro"               
]

if "gemini_api_configured" not in st.session_state:
    st.session_state.gemini_api_configured = False
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = None
if "available_models_info" not in st.session_state:
    st.session_state.available_models_info = "Not checked yet."

def configure_gemini_and_select_model():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        st.session_state.gemini_api_configured = True
        
        available_gemini_models = []
        model_info_str_parts = ["Available models supporting 'generateContent':"]
        found_preferred = False
        
        all_listed_models = list(genai.list_models()) 

        for preferred_model_name in PREFERRED_MODELS:
            for m in all_listed_models:
                if m.name == preferred_model_name and 'generateContent' in m.supported_generation_methods:
                    st.session_state.selected_model_name = m.name
                    model_info_str_parts.append(f"  -  {m.name} (Selected)")
                    found_preferred = True
                    break

            if found_preferred:
                break
        
        if not st.session_state.selected_model_name:
            for m in all_listed_models:
                if 'generateContent' in m.supported_generation_methods:
                    st.session_state.selected_model_name = m.name
                    model_info_str_parts.append(f"  -  {m.name} (Selected as fallback)")
                    break
        
        for m in all_listed_models:
            if m.name != st.session_state.selected_model_name and 'generateContent' in m.supported_generation_methods:
                 model_info_str_parts.append(f"  - {m.name}")

        st.session_state.available_models_info = "\n".join(model_info_str_parts)

        if not st.session_state.selected_model_name:
            st.session_state.available_models_info = "No suitable Gemini models supporting 'generateContent' found with your API key."
            st.error(st.session_state.available_models_info)
            return False
        return True

    except KeyError:
        st.error("GOOGLE_API_KEY not found in Streamlit secrets! Create .streamlit/secrets.toml")
        st.code("[secrets]\nGOOGLE_API_KEY = \"YOUR_API_KEY_HERE\"", language="toml")
        st.session_state.gemini_api_configured = False
        return False
    except Exception as e:
        st.error(f"Error configuring Gemini or listing models: {e}")
        st.session_state.gemini_api_configured = False
        return False

if not st.session_state.gemini_api_configured:
    if not configure_gemini_and_select_model():
        st.stop()

with st.sidebar:
    st.header("Configuration")
    if st.session_state.selected_model_name:
        st.success(f"Using Model: `{st.session_state.selected_model_name}`")
    else:
        st.warning("No model selected. Check API key and model availability.")
    
    with st.expander("Available Models Log", expanded=False):
        st.text_area("Models", st.session_state.available_models_info, height=200)

    st.header("Upload File")
    uploaded_file = st.file_uploader(
        "PDF, DOCX, JPG, PNG, JPEG",
        type=["pdf", "docx", "jpg", "jpeg", "png"],
        key="file_uploader"
    )

if "messages" not in st.session_state:
    st.session_state.messages = []  
if "current_file_data" not in st.session_state:
    st.session_state.current_file_data = None
if "current_file_id" not in st.session_state: 
    st.session_state.current_file_id = None


if uploaded_file:
    if st.session_state.current_file_id != uploaded_file.file_id:
        st.session_state.current_file_id = uploaded_file.file_id
        with st.spinner(f"Processing '{uploaded_file.name}'..."):
            file_data_result = prepare_file_data(uploaded_file)
        
        if file_data_result.get('error') == True: 
            st.sidebar.error(file_data_result['content'])
            st.session_state.current_file_data = None
        elif file_data_result.get('error') == 'warning': 
            st.sidebar.warning(file_data_result['content'])
            st.session_state.current_file_data = file_data_result 
            st.session_state.messages.append({
                'author': 'system_message_ui_only', 
                'content': f"File '{file_data_result['name']}' processed with warning: {file_data_result['content']}. Context may be partial."
            })
        else: # Success
            st.session_state.current_file_data = file_data_result
            st.sidebar.success(f"'{file_data_result['name']}' processed and ready for chat.")
            st.session_state.messages.append({
                'author': 'system_message_ui_only', 
                'content': f"Context from '{file_data_result['name']}' is now active. Ask questions about it!"
            })
            if file_data_result['type'] == 'image':
                st.sidebar.image(file_data_result['content'], caption=f"Uploaded: {file_data_result['name']}", use_column_width=True)
            elif file_data_result['type'] == 'text' and file_data_result['content']:
                 with st.sidebar.expander("Extracted Text"):
                    st.text(file_data_result['content'][:300] + "...")


elif not uploaded_file and st.session_state.current_file_id is not None:
    if st.session_state.current_file_data: 
         st.session_state.messages.append({
            'author': 'system_message_ui_only', 
            'content': f"File '{st.session_state.current_file_data['name']}' removed. Context cleared."
        })
    st.sidebar.info("File uploader is empty. Upload a file to chat about its content.")
    st.session_state.current_file_data = None
    st.session_state.current_file_id = None



for msg in st.session_state.messages:
    with st.chat_message(msg["author"] if msg["author"] != "system_message_ui_only" else "info"):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about the file or chat generally..."):
    if not st.session_state.selected_model_name:
        st.error("Cannot chat: No Gemini model is selected. Check API key and configuration.")
    else:
        st.session_state.messages.append({"author": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)


        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("One second..."):
                can_handle_file = True
                model_is_vision = "vision" in st.session_state.selected_model_name or "1.5" in st.session_state.selected_model_name # Heuristic
                
                temp_file_data_for_llm = st.session_state.current_file_data
                
                if st.session_state.current_file_data and st.session_state.current_file_data['type'] == 'image':
                    if not model_is_vision:
                        st.warning(f"Model '{st.session_state.selected_model_name}' may not support direct image input. Sending filename as context instead.")
                        temp_file_data_for_llm = {
                            'type': 'text', 
                            'content': f"User uploaded an image named '{st.session_state.current_file_data['name']}'. The user's query is about this image.",
                            'name': st.session_state.current_file_data['name'],
                            'error': None
                        }

                assistant_response = chat_with_gemini(
                    messages_history=st.session_state.messages,
                    current_file_data=temp_file_data_for_llm,
                    model_name_to_use=st.session_state.selected_model_name
                )
            message_placeholder.markdown(assistant_response)
        
        st.session_state.messages.append({"author": "assistant", "content": assistant_response})