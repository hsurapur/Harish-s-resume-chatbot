import streamlit as st
from Chatbot import load_and_split_pdf_resumes, create_vector_store, create_chatbot_chain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()
import os

st.set_page_config(page_title="Harish's Resume Chatbot", layout="wide")
st.title("Harish's Chatbot Using Gemini LLM")

pdf_directory = r"Harish_Surapur-Resume.pdf"

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "vector_store" not in st.session_state:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    with st.spinner("Loading PDFs and building vector store..."):
        docs = load_and_split_pdf_resumes(pdf_directory)
        if not docs:
            st.error("No valid text was extracted from the PDFs in the directory.")
            st.stop()
        st.session_state.vector_store = create_vector_store(docs, gemini_api_key=gemini_api_key)

if "vector_store" in st.session_state:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    qa_chain = create_chatbot_chain(st.session_state.vector_store, st.session_state.memory, gemini_api_key)
    
    st.subheader("Chat with Harish's Resume Assistant")
    user_input = st.text_input("Ask me your questions")
    
    if st.button("Send") and user_input:
        with st.spinner("Processing your question..."):
            result = qa_chain({"question": user_input})
            answer = result.get("answer", "Sorry, I could not generate an answer.")
            if "hire" in user_input.lower() or "why" in user_input.lower():
                answer = f"Harish is an excellent candidate because:\n" + answer
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("The vector store is not available. Check if the local PDFs are present and readable.")
