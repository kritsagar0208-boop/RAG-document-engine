import os
from dotenv import load_dotenv
from fastapi import FastAPI
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from openai import OpenAI

# 1. Load environment variables and initialize tools
load_dotenv()
app = FastAPI()
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 2. Connect to your existing Chroma database
chroma_client = chromadb.PersistentClient(path="./chroma_db")
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Use get_collection instead of create (since we already built it!)
collection = chroma_client.get_collection(
    name="document_collection",
    embedding_function=openai_ef
)

# 3. Create the Q&A Endpoint
@app.get("/ask")
async def ask_question(query: str):
    print(f"Searching database for: {query}")
    
    # Step A: Convert the user's question into a vector and find the 3 closest matches
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    
    # Step B: Stitch those 3 document chunks together into one large text block
    retrieved_context = "\n\n".join(results["documents"][0])
    
    # Step C: Give OpenAI strict instructions to ONLY use the retrieved context
    prompt = f"""
    Answer the user's question using ONLY the provided context below. 
    If you cannot find the answer in the context, politely state that you do not know.
    
    Context from Document:
    {retrieved_context}
    
    User Question: {query}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise, helpful document assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return {"question": query, "answer": response.choices[0].message.content}