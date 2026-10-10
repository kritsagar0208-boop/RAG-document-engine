# 📄 RAG-Powered AI Document Q&A Engine

A full-stack Retrieval-Augmented Generation (RAG) engine that allows users to query custom PDF documents using vector search and large language models.

## 🛠️ Tech Stack
- **Backend:** FastAPI, Python
- **Frontend:** Streamlit
- **Vector Database:** ChromaDB
- **Embeddings & LLM:** OpenAI `text-embedding-3-small` and `gpt-4o-mini`

## 🚀 System Architecture
1. **Ingestion & Embedding:** Converts PDF text into 1,000-character chunks and embeds them as mathematical vectors using OpenAI's `text-embedding-3-small` into a persistent ChromaDB instance.
2. **Semantic Search:** Converts user queries into vectors and uses cosine similarity to retrieve the top matching document chunks.
3. **Context Grounding:** Passes retrieved context into `gpt-4o-mini` with strict system guardrails to deliver accurate, hallucination-free answers.

## 🏎️ Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/rag-document-engine.git](https://github.com/YOUR_GITHUB_USERNAME/rag-document-engine.git)
   cd rag-document-engine