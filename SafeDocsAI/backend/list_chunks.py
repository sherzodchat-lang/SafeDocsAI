import asyncio
from sqlmodel import select
from app.core.database import get_session
from app.models.models import Chunk

async def list_chunks():
    async for session in get_session():
        statement = select(Chunk)
        result = await session.exec(statement)
        chunks = result.all()
        for chunk in chunks:
            print(f"--- Chunk {chunk.id} (Doc ID {chunk.doc_id}) ---")
            print(chunk.text)
            print("-" * 30)

if __name__ == "__main__":
    asyncio.run(list_chunks())
