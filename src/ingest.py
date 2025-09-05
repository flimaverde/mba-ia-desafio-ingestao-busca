import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

def add_documents_in_batches(store, documents, ids, batch_size=5, delay=30):
    total = len(documents)
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        try:
            store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"Processados {min(i + batch_size, total)}/{total} documentos")
            
            if i + batch_size < total:
                time.sleep(delay)
                
        except Exception as e:
            if "429" in str(e):
                print("Rate limit atingido. Aguardando...")
                time.sleep(delay * 2)
                store.add_documents(documents=batch_docs, ids=batch_ids)
            else:
                raise e

PDF_PATH = os.getenv("PDF_PATH")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()    
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)


enriched = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
    )
    for d in chunks
]    

ids = [f"doc-{i}" for i in range(len(enriched))]

embeddings = GoogleGenerativeAIEmbeddings(task_type="RETRIEVAL_DOCUMENT", model="gemini-embedding-001", request_timeout=60)

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
    connection=os.getenv("DATABASE_URL"),
    use_jsonb=True,
)

add_documents_in_batches(store, enriched, ids, batch_size=20, delay=5)

