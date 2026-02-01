import streamlit as st
from faster_whisper import WhisperModel
import torch
import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  

# --- Page Configurations ---
st.set_page_config(page_title="AI Smart Assistant", page_icon="🤖", layout="centered")
st.markdown("""
    <style>
    
    .stChatMessage {
        direction: rtl !important;
        text-align: right !important;
    }
    
    
    .stChatMessage ul, .stChatMessage ol {
        direction: rtl !important;
        padding-right: 1.5rem !important;
        padding-left: 0 !important;
        list-style-position: inside !important;
    }

    
    .stChatMessage li {
        text-align: right !important;
    }

    
    .stChatMessage .st-emotion-cache-1c7n2ka {
        flex-direction: row-reverse !important;
    }
    </style>
    """, unsafe_allow_html=True)

# API Key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error(" اعمل ليهو اضافة GROQ_API_KEY في الإعدادات!")
# --- Load Models (Cached for Performance) ---
@st.cache_resource
def load_all():
    print("Loading Models...")
    # 1. Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")#بيدعم 50 لغة م بحتاج ترجمه
    # 2. Vector DB
    vectordb = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    # 3. Whisper (Using int8 to save VRAM)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper = WhisperModel("turbo", device="cpu", compute_type="int8")#STT
    return vectordb.as_retriever(search_kwargs={"k": 3}), whisper

retriever, whisper_model = load_all()

# --- Groq Function ---
def ask_groq(prompt):
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile", 
                "messages": [{"role": "user", "content": prompt}], 
                "temperature": 0.3
            },
            timeout=15
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"Error connecting to Groq: {str(e)}"

# --- UI Interface ---
st.title("🤖 AI Technical Assistant")
st.markdown("---")

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Input Area (Sidebar for Audio, Bottom for Text) ---
with st.sidebar:
    st.header("Input Options")
    # The magical audio input widget
    audio_file = st.audio_input("Record your question")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

user_query = st.chat_input("Or type your question here...")

# --- Processing Logic ---
final_query = None

# 1. If Audio is recorded
if audio_file:
    with st.spinner("⏳ Transcribing your voice..."):
        with open("temp.wav", "wb") as f:
            f.write(audio_file.read())
        segments, _ = whisper_model.transcribe("temp.wav", beam_size=5)
        final_query = " ".join([s.text for s in segments])
        st.sidebar.success(f"🎤 Heard: {final_query}")

# 2. If Text is typed
if user_query:
    final_query = user_query

# --- RAG Execution ---
if final_query:
    # Add User Message to UI
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"):
        st.markdown(final_query)

    with st.spinner("🔍 Searching documents & thinking..."):
        # Retrieve Context
        docs = retriever.invoke(final_query)
        context = "\n".join([d.page_content for d in docs])
        
        # Professional Prompt Engineering
        # الـ Prompt (Strict Mode)
        full_prompt = f"""
You are a specialized technical assistant. Your knowledge is strictly limited to the provided context.

### Instructions:
1. Answer the user's question using ONLY the provided context below.
2. If the answer is NOT found in the context, or if the question is outside the scope of the documents, strictly reply with: 
   "I'm sorry, this is outside my current scope as it is not mentioned in the provided documents."
3. Do NOT use your general knowledge to answer questions.
4. Use Markdown (bullet points, headers, bold text) for structure.
5. The response should be in Arabic if the user asks in Arabic.

### Context:
{context}

### Question:
{final_query}

### Answer:"""

        answer = ask_groq(full_prompt)
        
        # Display Assistant Response
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})