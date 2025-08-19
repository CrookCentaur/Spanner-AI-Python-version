# interface.py
# Streamlit interface for Spanner AI with FAISS RAG + Gemini (no async/grpc)

import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

# ---------------- Setup ----------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Put it in your .env file.")
    st.stop()

# Configure the official Google Generative AI SDK (HTTP only, no event loop)
genai.configure(api_key=GOOGLE_API_KEY)

# Load FAISS index (built previously via chunker.py)
INDEX_PATH = "toyota_manual_index"  # <— ensure this folder exists beside interface.py
try:
    # We load without an embedding function because we'll embed queries ourselves
    db = FAISS.load_local(INDEX_PATH, embeddings=None, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"❌ Failed to load FAISS index at '{INDEX_PATH}'.\n{e}")
    st.stop()

# ---------------- Helpers ----------------
def embed_query(text: str) -> list[float]:
    """Return embedding vector for a query using Gemini embeddings API."""
    # task_type="retrieval_query" is recommended for queries
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_query",
    )
    return res["embedding"]

def answer_with_gemini(context: str, question: str) -> str:
    """Call Gemini text model via REST to generate a concise, friendly answer."""
    prompt = f"""
You are Spanner AI, a concise and friendly automotive assistant.
Use the provided manual excerpts when applicable. If they are insufficient,
use your general automotive knowledge. Keep answers short, clear, and human.

Context:
{context}

Question:
{question}

Answer:
"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GOOGLE_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 500},
    }
    r = requests.post(url, json=payload, timeout=60)
    if not r.ok:
        return f"⚠️ Gemini API error: {r.text}"
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return "⚠️ Sorry, I couldn't generate a response."

def query_rag(user_input: str, k: int = 3):
    """Embed the query, search FAISS by vector, and ask Gemini with context."""
    try:
        qvec = embed_query(user_input)
    except Exception as e:
        return f"⚠️ Embedding error: {e}", []

    # Use FAISS by-vector search (no embedding function needed on the store)
    try:
        docs = db.similarity_search_by_vector(qvec, k=k)
    except Exception as e:
        return f"⚠️ FAISS search error: {e}", []

    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    answer = answer_with_gemini(context, user_input)
    return answer, docs

# ---------------- Streamlit UI ----------------
st.title("🚗 Spanner AI - Car Manual Assistant")
st.caption("Ask car or motorcycle questions. I’ll search your manual and answer concisely.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How may I help you today?"}]

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input
if prompt := st.chat_input("Describe the issue and include your make/model..."):
    # Show user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant reply
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking… ▌")

        answer, docs = query_rag(prompt)

        # Simulate a tiny typing effect (optional)
        typed = ""
        for word in answer.split():
            typed += word + " "
            time.sleep(0.01)
            placeholder.markdown(typed + "▌")
        placeholder.markdown(typed.strip())

        # Optional: show sources (first 120 chars of each)
        if docs:
            with st.expander("📚 Sources (top matches)"):
                for i, d in enumerate(docs, 1):
                    src = d.metadata.get("source", "manual")
                    pg = d.metadata.get("page", None)
                    head = (d.page_content or "").strip().replace("\n", " ")
                    snippet = head[:160] + ("…" if len(head) > 160 else "")
                    st.markdown(f"**{i}.** {src}{f' (p.{pg})' if pg is not None else ''}\n\n> {snippet}")

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": answer})
