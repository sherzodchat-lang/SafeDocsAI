import asyncio
from app.services.hybrid_chunker import HybridChunker
from app.services.source_service import SourceService

async def main():
    with open("test.txt", "w") as f:
        f.write("2023-10-01\n2023-10-02\n")

    chunker = HybridChunker()
    chunk_results = await asyncio.to_thread(
        SourceService.extract_and_chunk, "test.txt", ".txt", chunker
    )
    print(len(chunk_results))
    print(chunk_results)

if __name__ == "__main__":
    asyncio.run(main())
