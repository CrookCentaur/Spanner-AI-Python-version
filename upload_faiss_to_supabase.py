import os
import json
from supabase import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load env
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ Define all manuals to upload
# Key = manual name, Value = folder where FAISS index is stored
MANUALS = {
    "Toyota Corolla 2020": "toyota_manual_index",
    # Add more like:
    # "Honda Civic 2021": "honda_manual_index",
    # "Yamaha R15": "yamaha_r15_manual_index",
}

# Use HuggingFace embeddings (no quotas 🚀)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

for manual_name, index_folder in MANUALS.items():
    print(f"\n📖 Processing manual: {manual_name}")

    # Load FAISS index
    db = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS index from {index_folder}")

    # Upload docs + embeddings
    for i, doc in enumerate(db.docstore._dict.values()):
        emb = embeddings.embed_query(doc.page_content)
        record = {
            "id": f"{manual_name}_{i}",  # unique per manual
            "manual_name": manual_name,
            "content": doc.page_content,
            "metadata": json.dumps(doc.metadata) if doc.metadata else "{}",
            "embedding": emb,  # pgvector
        }
        supabase.table("manual_chunks").upsert(record).execute()

    print(f"✅ Uploaded chunks for {manual_name} into Supabase")
