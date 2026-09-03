import sys; sys.path.insert(0, ".")
from src.chunking import ThaiLegalChunker, chunks_to_dicts
from src.indexer import Indexer

chunker = ThaiLegalChunker()
chunks = chunker.chunk_corpus()

sizes = [c.n_tokens for c in chunks]
print(f"Total chunks: {len(chunks)} | avg tokens: {sum(sizes)/len(sizes):.0f} "
      f"| min: {min(sizes)} | max: {max(sizes)}")

Indexer().build(chunks_to_dicts(chunks))
