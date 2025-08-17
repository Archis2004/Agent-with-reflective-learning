import io
import os
import sys
import re
import streamlit as st
from PIL import Image
import google.generativeai as genai

from chatbot import prepare_text_docs

from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


st.set_page_config(page_title="Multimodal Chatbot", layout="wide")
st.title("Multimodal Chatbot with Tools")
st.caption("Upload a file for RAG, an image for analysis, or just ask a question to use the tools.")


try:
    gemini_key = st.secrets["GOOGLE_API_KEY"]
    tavily_key = st.secrets["TAVILY_API_KEY"]
    os.environ["TAVILY_API_KEY"] = tavily_key 
    genai.configure(api_key=gemini_key)
except Exception as e:
    st.error(f"Could not find API keys in .streamlit/secrets.toml. Please ensure GOOGLE_API_KEY and TAVILY_API_KEY are set. Error: {e}")
    st.stop()


def run_python_code(code: str) -> str:
    """Executes Python code and returns the output."""
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    try:
        exec(code, globals())
        output = captured_output.getvalue()
        return output if output else "Code executed successfully with no output."
    except Exception as e:
        return f"Error executing Python code: {e}"
    finally:
        sys.stdout = old_stdout


def process_tool_output(question: str, tool_name: str, tool_output: str, model_name: str, generated_code: str = ""):
    """
    Uses the Gemini model to synthesize a final answer from raw tool output.
    If the user asks for the code, it returns the generated code instead.
    """
    model = genai.GenerativeModel(model_name)
    
    if tool_name == "Python Code Runner" and generated_code and ("code" in question.lower() or "script" in question.lower()):
        return f"Here is the code that was generated and successfully executed:\n```python\n{generated_code}\n```"

    prompt = f"""You are an expert at interpreting tool outputs.
Based on the user's question and the output from a tool, formulate a concise, natural language answer.

User's Question: "{question}"

The "{tool_name}" tool returned the following information:
---
{str(tool_output)}
---

Based on the tool's successful output, what is the final answer to the user's question?
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while processing the tool's output: {e}"


def clean_generated_code(code: str) -> str:
    """Removes markdown backticks and 'python' language identifier."""
    return re.sub(r'^```python\s*|\s*```$', '', code, flags=re.MULTILINE).strip()


with st.sidebar:
    st.header("Upload & Configuration")
    chat_model_name = st.selectbox("Gemini Model", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest"])
    uploaded_file = st.file_uploader("PDF, DOCX, PNG, JPG or JPEG", type=["pdf", "docx", "png", "jpg", "jpeg"])


if "history" not in st.session_state: st.session_state.history = []
if "agent_executor" not in st.session_state: st.session_state.agent_executor = None
if "rag_chain" not in st.session_state: st.session_state.rag_chain = None
if "image_ref" not in st.session_state: st.session_state.image_ref = None
if 'selected_tool' not in st.session_state: st.session_state.selected_tool = 'auto'


tavily_tool_instance = TavilySearchResults(max_results=3)
wikipedia_tool_instance = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
python_tool_instance = Tool(name="PythonCodeRunner", func=run_python_code, description="A Python code interpreter.")
tools = [tavily_tool_instance, wikipedia_tool_instance, python_tool_instance]


def initialize_agent():
    llm = ChatGoogleGenerativeAI(model=chat_model_name, api_key=gemini_key, convert_system_message_to_human=True)
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."), ("user", "{input}"), ("placeholder", "{agent_scratchpad}")])
    agent = create_tool_calling_agent(llm, tools, prompt)
    st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
if not st.session_state.agent_executor: initialize_agent()


if uploaded_file:
    ext = uploaded_file.name.lower().rsplit(".", 1)[-1]
    is_text_file = ext in ("pdf", "docx")
    is_image_file = ext in ("png", "jpg", "jpeg")
    with st.spinner("Processing file..."):
        if is_text_file:
            docs = prepare_text_docs(uploaded_file)
            if docs:
                emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_key)
                vs = FAISS.from_documents(docs, emb)
                retriever = vs.as_retriever(search_kwargs={"k": 3})
                llm = ChatGoogleGenerativeAI(model=chat_model_name, api_key=gemini_key)
                system_prompt = ("You are an assistant for question-answering tasks. Use the retrieved context to answer the question. If you don't know the answer, just say that you don't know. Keep the answer concise.\n\n{context}")
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                st.session_state.image_ref = None
                st.sidebar.success("Document ready for RAG!")
            else:
                st.sidebar.error("Failed to parse document.")
                st.session_state.rag_chain = None
        elif is_image_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.sidebar.image(img, caption="Loaded image")
            st.session_state.image_ref = img
            st.session_state.rag_chain = None
            st.sidebar.success("Image ready for Q&A!")
else:
    st.session_state.rag_chain = None
    st.session_state.image_ref = None
    if not st.session_state.agent_executor: initialize_agent()


st.write("### Choose Your Tool")
is_file_loaded = bool(st.session_state.rag_chain or st.session_state.image_ref)
def set_tool_mode(mode): st.session_state.selected_tool = mode
col1, col2, col3, col4 = st.columns(4)
with col1: st.button("Auto (AI Agent)", on_click=set_tool_mode, args=('auto',), use_container_width=True, disabled=is_file_loaded, type="primary" if st.session_state.selected_tool == 'auto' and not is_file_loaded else "secondary")
with col2: st.button("Web Search", on_click=set_tool_mode, args=('web_search',), use_container_width=True, disabled=is_file_loaded, type="primary" if st.session_state.selected_tool == 'web_search' and not is_file_loaded else "secondary")
with col3: st.button("Wikipedia", on_click=set_tool_mode, args=('wikipedia',), use_container_width=True, disabled=is_file_loaded, type="primary" if st.session_state.selected_tool == 'wikipedia' and not is_file_loaded else "secondary")
with col4: st.button("Python Runner", on_click=set_tool_mode, args=('python_runner',), use_container_width=True, disabled=is_file_loaded, type="primary" if st.session_state.selected_tool == 'python_runner' and not is_file_loaded else "secondary")


for role, msg in st.session_state.history:
    st.chat_message(role).markdown(msg)


placeholder_prompts = {'auto': "Ask a question and let the AI decide...", 'web_search': "Enter a query for Web Search...", 'wikipedia': "Enter a topic for Wikipedia...", 'python_runner': "Describe a task for the AI to code and run..."}
if is_file_loaded:
    placeholder = "Ask a question about the loaded file..."
else:
    placeholder = placeholder_prompts.get(st.session_state.selected_tool, "Ask a question...")
if user_q := st.chat_input(placeholder):
    st.session_state.history.append(("user", user_q))
    st.chat_message("user").markdown(user_q)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            answer = ""
            if st.session_state.rag_chain:
                response = st.session_state.rag_chain.invoke({"input": user_q})
                answer = response.get("answer", "No answer found in the document.")
            elif st.session_state.image_ref:
                model = genai.GenerativeModel(chat_model_name)
                response = model.generate_content([user_q, st.session_state.image_ref])
                answer = response.text
            else:
                tool_mode = st.session_state.get('selected_tool', 'auto')
                if tool_mode == 'python_runner':
                    model = genai.GenerativeModel(chat_model_name)
                    code_gen_prompt = f"Generate a short, runnable Python code snippet to accomplish the following task. The code must produce a final output using the print() function. Do not explain the code, just provide the raw code itself, without markdown backticks. Task: {user_q}"
                    response = model.generate_content(code_gen_prompt)
                    generated_code = clean_generated_code(response.text)
                    final_code_to_run = generated_code
                    
                    tool_output = python_tool_instance.invoke(generated_code)

                    if "error executing python code" in tool_output.lower():
                        st.write("Initial code failed. Attempting to debug...")
                        debug_prompt = f"The following Python code, generated for the task '{user_q}', failed with an error.\n\nCode:\n{generated_code}\n\nError:\n{tool_output}\n\nPlease provide the corrected, runnable Python code. Do not explain the code, just provide the raw corrected code itself, without markdown backticks."
                        response = model.generate_content(debug_prompt)
                        corrected_code = clean_generated_code(response.text)
                        final_code_to_run = corrected_code
                        
                        tool_output = python_tool_instance.invoke(corrected_code)
                    

                    answer = process_tool_output(user_q, "Python Code Runner", tool_output, chat_model_name, generated_code=final_code_to_run)

                elif tool_mode == 'web_search':
                    tool_output = tavily_tool_instance.invoke(user_q)
                    answer = process_tool_output(user_q, "Web Search", tool_output, chat_model_name)
                elif tool_mode == 'wikipedia':
                    tool_output = wikipedia_tool_instance.invoke(user_q)
                    answer = process_tool_output(user_q, "Wikipedia Search", tool_output, chat_model_name)
                elif tool_mode == 'auto':
                    if st.session_state.agent_executor:
                        response = st.session_state.agent_executor.invoke({"input": user_q})
                        answer = response.get("output", "I am not sure how to respond.")
                    else:
                        model = genai.GenerativeModel(chat_model_name)
                        response = model.generate_content(user_q)
                        answer = response.text
        except Exception as e:
            answer = f"An unexpected error occurred: {e}"
        st.markdown(str(answer))
        st.session_state.history.append(("assistant", str(answer)))