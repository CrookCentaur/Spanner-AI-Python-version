# upload_faiss_to_supabase.py
# Auto-detect FAISS indexes and upload chunks + embeddings to Supabase
# With batching + skip already uploaded manuals

import os
import json
import time
from supabase import create_client, Client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from tqdm import tqdm  # progress bar

# ----------------- Setup -----------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Use HuggingFace embeddings (768-dim)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Folder where chunker.py saved all FAISS indexes
INDEX_DIR = "C:\\Users\\soham\\OneDrive\\Desktop\\Spanner AI\\code\\car-manual-rag\\data\\FAISS Indexes"

# ----------------- Upload Process -----------------
BATCH_SIZE = 50  # upload 50 rows per request

for index_folder in os.listdir(INDEX_DIR):
    index_path = os.path.join(INDEX_DIR, index_folder)
    if not os.path.isdir(index_path):
        continue

    manual_name = index_folder.replace("_index", "")

    # ✅ Skip manuals already uploaded
    check = supabase.table("manual_chunks").select("id").eq("manual_name", manual_name).limit(1).execute()
    if check.data:
        print(f"⏭️  Skipping {manual_name} (already uploaded)")
        continue

    print(f"\n📖 Processing manual: {manual_name}")

    # Load FAISS index
    db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    print(f"✅ Loaded FAISS index from {index_path}")

    # Prepare records
    records = []
    for i, doc in enumerate(db.docstore._dict.values()):
        emb = embeddings.embed_query(doc.page_content)
        record = {
            "id": f"{manual_name}_{i}",  # unique per manual
            "manual_name": manual_name,
            "content": doc.page_content,
            "metadata": json.dumps(doc.metadata) if doc.metadata else "{}",
            "embedding": emb,
        }
        records.append(record)

    # Upload in batches with progress bar
    print(f"⬆️ Uploading {len(records)} chunks to Supabase...")
    for i in tqdm(range(0, len(records), BATCH_SIZE), desc=f"Uploading {manual_name}"):
        batch = records[i : i + BATCH_SIZE]
        supabase.table("manual_chunks").upsert(batch).execute()
        time.sleep(0.2)  # small pause to avoid rate limits

    print(f"✅ Uploaded chunks for {manual_name} into Supabase")

print("\n🎉 All FAISS indexes uploaded to Supabase.")
