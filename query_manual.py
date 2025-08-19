# code to load FAISS index, set up Gemini LLM, answer questions
# Running this in cmd prompt tests the RAG system

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

# Recreate the same embeddings object used in chunker.py
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Load existing FAISS index
vectorstore = FAISS.load_local(
    "toyota_manual_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = input("Ask a question about the Toyota manual: ")
result = qa_chain.invoke({"query": query})

# Show answer
print("\nAnswer:", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata)
