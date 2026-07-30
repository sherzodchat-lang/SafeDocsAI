import asyncio
from app.services.rag_service import RAGService

async def peek_chroma():
    rag = RAGService()
    if not rag.collection:
        print("Collection not found.")
        return
    
    results = rag.collection.get()
    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    
    print(f"Total documents in ChromaDB: {len(ids)}")
    for i in range(len(ids)):
        print(f"--- Chroma ID: {ids[i]} ---")
        print(f"Metadata: {metadatas[i]}")
        print(f"Text Snippet: {documents[i][:100]}...")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(peek_chroma())
