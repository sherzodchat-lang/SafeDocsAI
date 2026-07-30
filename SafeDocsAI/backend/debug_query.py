import asyncio
import json
import re
from app.services.rag_service import RAGService
from app.api.endpoints.chat import _select_relevant_chunks

async def debug_query(question: str):
    print(f"--- Debugging Question: {question} ---")
    rag_service = RAGService()
    normalized_question = rag_service.normalize_query(question)
    print(f"Normalized: {normalized_question}")
    
    results = rag_service.query_documents(normalized_question, n_results=5)
    documents = results.get("documents", [])
    chunk_ids = results.get("ids", [])
    metadatas = results.get("metadatas", [])
    distances = results.get("distances", [])
    
    context = documents[0] if documents else []
    context_chunk_ids = chunk_ids[0] if chunk_ids else []
    context_metadatas = metadatas[0] if metadatas else []
    context_distances = distances[0] if distances else []
    
    print(f"Found {len(context)} chunks in ChromaDB")
    for i in range(len(context)):
        print(f"Chunk {context_chunk_ids[i]}: Distance={context_distances[i]:.4f}")
        print(f"Text Snippet: {context[i][:100]}...")

    selected = _select_relevant_chunks(
        context=context,
        context_chunk_ids=context_chunk_ids,
        context_metadatas=context_metadatas,
        context_distances=context_distances
    )
    print(f"\nSelected {len(selected)} chunks after filtering")
    for s in selected:
        print(f"Selected: {s['chunk_id']} (Distance={s['distance']:.4f})")

if __name__ == "__main__":
    import sys
    q = "Какова стандартная ставка налога на прыбыль?"
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    asyncio.run(debug_query(q))
