import os
import glob
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma_db")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="document_collection",
    embedding_function=openai_ef
)

def process_all_pdfs():
    # Find all .pdf files in your folder
    pdf_files = glob.glob("*.pdf")
    if not pdf_files:
        print("No PDF files found!")
        return

    print(f"Found {len(pdf_files)} PDF file(s): {pdf_files}")
    
    all_chunks = []
    all_ids = []
    
    for pdf_path in pdf_files:
        print(f"Reading {pdf_path}...")
        reader = PdfReader(pdf_path)
        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
        
        # Break text into chunks
        chunks = [raw_text[i:i+1000] for i in range(0, len(raw_text), 1000)]
        
        # Create unique IDs for each chunk using the filename
        filename = os.path.basename(pdf_path).replace(" ", "_")
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_chunk_{idx}")
            
    if all_chunks:
        print(f"Saving {len(all_chunks)} total chunks across all PDFs to ChromaDB...")
        collection.add(
            documents=all_chunks,
            ids=all_ids
        )
        print("Database update complete!")

process_all_pdfs()