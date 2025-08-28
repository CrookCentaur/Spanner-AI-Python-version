# Spanner-AI-Python-version
### A python variant of my Spanner AI diagnostic tool project

Uses HuggingFace model to create 786-dimension vector embeddings using local resources
Then pushes those embeddings to Supabase
The frontend then queries Supabase and fetches said embeddings
The embeddings are used to generate a RAG-response
Uses *Google Gemini 2.5 Flash* to generate human-like response for the user
