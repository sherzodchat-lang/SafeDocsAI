import asyncio
from app.services.hybrid_chunker import HybridChunker, TextBlock

chunker = HybridChunker()
blocks = [TextBlock(text="2023-10-01", page=1, order=0)]
chunks = chunker.chunk(blocks)
print(chunks)
