# chunker.py
# Create chunks + embeddings for multiple manuals and save FAISS indexes
# Now with fallback loader for encrypted/problematic PDFs

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------- Setup -----------------
load_dotenv()

DATA_DIR = "C:\\Users\\soham\\OneDrive\\Desktop\\Spanner AI\\code\\car-manual-rag\\data\\manuals"      # Folder containing all manuals (PDFs)
OUTPUT_DIR = "C:\\Users\\soham\\OneDrive\\Desktop\\Spanner AI\\code\\car-manual-rag\\data\\FAISS Indexes" # Folder to save FAISS indexes
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use HuggingFace 768-dim model (consistent with uploader + interface)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Chunking strategy
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # ~500 words
    chunk_overlap=100,    # context overlap
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ----------------- Process Manuals -----------------
for file in os.listdir(DATA_DIR):
    if not file.endswith(".pdf"):
        continue

    manual_name = os.path.splitext(file)[0]  # e.g. "Toyota_Corolla_2020"
    pdf_path = os.path.join(DATA_DIR, file)

    print(f"\n📖 Processing manual: {manual_name}")

    # 1. Load PDF (try PyPDF first, then fallback to PDFPlumber)
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"⚠️ PyPDFLoader failed for {manual_name} ({e}), trying PDFPlumberLoader...")
        try:
            loader = PDFPlumberLoader(pdf_path)
            documents = loader.load()
        except Exception as e2:
            print(f"❌ Failed to load {manual_name} with both loaders ({e2}). Skipping.")
            continue

    # 2. Split into chunks
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")

    # 3. Create FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 4. Save FAISS index
    out_path = os.path.join(OUTPUT_DIR, f"{manual_name}_index")
    vectorstore.save_local(out_path)
    print(f"✅ Saved FAISS index to {out_path}")

print("\n🎉 All manuals processed and FAISS indexes saved.")
