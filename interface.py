import os
import json
import streamlit as st
from supabase import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai
from rapidfuzz import fuzz, process


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
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Use HuggingFace embeddings (768-dim)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# ----------------- Utility: Manual Detection -----------------
def normalize_name(name: str) -> str:
    """Normalize car names by stripping brand and punctuation."""
    name = name.lower()
    for brand in ["toyota", "honda", "yamaha", "mazda"]:
        name = name.replace(brand, "")
    return "".join(ch for ch in name if ch.isalnum() or ch.isspace()).strip()

def detect_manual(user_input: str) -> tuple[str | None, int]:
    """
    Detect if the user mentions a known manual (by fuzzy matching).
    Returns (manual_name, score).
    """
    manuals_resp = supabase.table("manual_chunks").select("manual_name").execute()
    all_manuals = [m["manual_name"] for m in manuals_resp.data]

    if not all_manuals:
        return None, 0

    query_norm = normalize_name(user_input)

    # Build forward + reverse lookup maps
    manual_norms = {manual: normalize_name(manual) for manual in all_manuals}
    reverse_map = {v: k for k, v in manual_norms.items()}  # norm → original

    # Run fuzzy match on normalized names
    result = process.extractOne(query_norm, list(manual_norms.values()), scorer=fuzz.token_sort_ratio)

    if result:
        best_match, score, _ = result  # RapidFuzz v3 returns (match, score, index)
        original_manual = reverse_map.get(best_match)
        if original_manual:
            return original_manual, score

    return None, 0


# ----------------- RAG + Synthesis Function -----------------
def query_rag(user_input: str, k: int = 3, manual_name: str = None) -> str:
    # Handle simple greetings / non-questions
    if len(user_input.strip().split()) < 2:
        return "👋 Hello! How can I help you with your car manual today?"

    query_vector = embeddings.embed_query(user_input)

    rpc_params = {"query_embedding": query_vector, "match_count": k}
    if manual_name:
        rpc_params["manual_filter"] = manual_name

    response = supabase.rpc("match_manual_chunks", rpc_params).execute()
    matches = response.data

    if not matches:
        return "Sorry, I couldn’t find anything in the manuals. Can you rephrase?"

    context = "\n\n".join([f"({m['manual_name']}) {m['content']}" for m in matches])

    prompt = f"""
You are Spanner AI, a helpful and concise car manual assistant.
Use the following context from car manuals to answer the question:

{context}

Question: {user_input}

Answer:"""

    gemini_response = gemini_model.generate_content(prompt)
    return gemini_response.text.strip()


# ----------------- Streamlit UI -----------------
st.title("🔧 Spanner AI - Your Car Manual Assistant")
st.caption("Ask me any car-related questions and mention the make & model. I’ll search the manual to help you out!")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How may I help you today?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Describe your problem..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    manual_name = None
    if len(prompt.strip().split()) >= 2:  # skip greetings
        detected_manual, score = detect_manual(prompt)

        if detected_manual and score >= 80:
            manual_name = detected_manual
            st.info(f"🔎 Searching in **{manual_name}** manual (confidence {score}%)")

        elif detected_manual and 60 <= score < 80:
            st.warning(f"❓ Did you mean **{detected_manual}**? (confidence {score}%)")
            # keep manual_name = None so we still search all manuals

        else:
            st.info("🔎 No specific manual detected — searching across all available manuals")

    # Get assistant response
    with st.chat_message("assistant"):
        response = query_rag(prompt, manual_name=manual_name)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
