# code to create chunks + embeddings and saves FAISS index

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Set your Google Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# 1. Load PDF
pdf_path = "C:\\Users\\soham\\OneDrive\\Desktop\\Spanner AI\\code\\car-manual-rag\\data\\toyota_manual.pdf"  # Must be in the same folder
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
chunks = splitter.split_documents(documents)

print(f"✅ Loaded {len(chunks)} chunks from the manual.")

# 3. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# 4. Store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Save the vectorstore locally
vectorstore.save_local("toyota_manual_index")

print("✅ Embeddings created and stored in 'toyota_manual_index'")
