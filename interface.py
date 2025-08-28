# interface.py

import os
import json
import streamlit as st
from supabase import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai

# ----------------- Setup -----------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials in .env file")

if not GOOGLE_API_KEY:
    raise ValueError("❌ Missing Google Gemini API key in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure Google Gemini client
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")  # Adjust model name if needed

# Use the same embedding model as your uploader (768-dim)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ----------------- RAG + Synthesis Function -----------------
def query_rag(user_input: str, k: int = 3) -> str:
    """
    Query Supabase for similar chunks and synthesize natural language response using Google Gemini.
    """
    # Embed user query with 768-dim model
    query_vector = embeddings.embed_query(user_input)

    # Call Supabase RPC for similarity search (expects: query_embedding, match_count)
    response = supabase.rpc(
        "match_manual_chunks",
        {
            "query_embedding": query_vector,
            "match_count": k,
        },
    ).execute()

    matches = response.data
    if not matches:
        return "Sorry, I couldn’t find anything in the manual. Can you rephrase?"

    # Build context from retrieved docs
    context = "\n\n".join([f"({m['manual_name']}) {m['content']}" for m in matches])

    # Construct prompt for Gemini
    prompt = f"""
You are Spanner AI, a helpful and concise car manual assistant.
Use the following context from car manuals to answer the question:

{context}

Question: {user_input}

Answer:"""

    # Call Google Gemini to generate answer
    gemini_response = gemini_model.generate_content(prompt)
    
    return gemini_response.text.strip()

# ----------------- Streamlit UI -----------------
st.title("🚗 Spanner AI - Car Manual Assistant")
st.caption("Ask me car-related questions. I’ll search the manual and help you out!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How may I help you today?"}
    ]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new input
if prompt := st.chat_input("Describe your problem..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Assistant response
    with st.chat_message("assistant"):
        response = query_rag(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response
    })