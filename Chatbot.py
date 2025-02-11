import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Initialize Gemini
def configure_gemini(gemini_api_key: str):
    genai.configure(api_key=gemini_api_key)

class EnhancedChatGemini(ChatGoogleGenerativeAI):
    def __init__(self, temperature=0.2, model_name="gemini-pro", **kwargs):
        super().__init__(temperature=temperature, model=model_name, **kwargs)

def load_and_split_pdf_resumes(directory_path, chunk_size=1000, chunk_overlap=150):
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(".pdf"):
            try:
                with open(os.path.join(directory_path, filename), "rb") as f:
                    text = "\n".join([p.extract_text() for p in PdfReader(f).pages if p.extract_text()])
                    chunks = text_splitter.split_text(text)
                    documents.extend([
                        Document(
                            page_content=chunk,
                            metadata={"source": filename}
                        ) for chunk in chunks
                    ])
            except Exception:
                continue
    
    if documents:
        st.success("Resume data loaded successfully")
    else:
        st.error("No valid resumes found")
    return documents

def create_vector_store(documents, gemini_api_key: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key)
    return FAISS.from_documents(documents=documents,embedding=embeddings)

def create_chatbot_chain(vector_store, memory, gemini_api_key: str):
    hiring_prompt = (
        "When asked about hiring suitability, highlight:\n"
        "- 4+ years data science experience\n"
        "- Cloud platform expertise (GCP, Azure)\n"
        "- ML deployment experience (TensorFlow, PyTorch)\n"
        "- Measurable business impacts achieved\n"
        "- Advanced degrees and certifications\n"
    )
    experience_prompt = (
        "For work experience questions:\n"
        "1. List ALL positions in reverse chronological order\n"
        "2. Include company, location, duration\n"
        "3. Add 2-3 key achievements per role\n"
        "4. Never repeat entries\n"
    )

    custom_prompt = PromptTemplate(
        template=(
            "You are Harish's career assistant. Rules:\n"
            + experience_prompt +
            "1. AGGREGATE INFORMATION FROM ALL SOURCES\n"
            "2. PRIORITIZE WORK EXPERIENCE SECTION\n"
            "3. FOR TOOLS/SKILLS:\n"
            "   - List all mentioned tools in all sections of the resume\n"
            "   - Include parent projects/context\n"
            "4. HIRING QUESTIONS:\n" + hiring_prompt +
            "5. FORMAT:\n"
            "   - Concise natural English\n"
            "   - No markdown or source references\n\n"
            "Context:\n{context}\n\n"
            "History:\n{chat_history}\n\n"
            "Question: {question}\n\n"
            "Answer in English only:"
        ),
        input_variables=["chat_history", "context", "question"]
    )
    retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 20,"lambda_mult": 0.3,}
    )
    
    llm = EnhancedChatGemini(google_api_key=gemini_api_key)
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=False
    )