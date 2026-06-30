from rag_pipeline.document_loader import DocumentLoader
from rag_pipeline.chunker import TextChunker
from rag_pipeline.embedder import Embedder
from rag_pipeline.vector_store import VectorStore

PDF_PATH = "data/Sample.pdf"

print("=" * 60)
print("INDEXING DOCUMENT")
print("=" * 60)

# Load PDF
loader = DocumentLoader(PDF_PATH)
text = loader.load()

# Chunk Text
chunker = TextChunker()
chunks = chunker.split_text(text)

print(f"\nChunks Created : {len(chunks)}")

# Generate Embeddings
embedder = Embedder()
embeddings = embedder.generate_embeddings(chunks)

print("Embeddings Generated")

# Store in ChromaDB
db = VectorStore()

# Remove previous database
db.clear_collection()

db.add_documents(chunks, embeddings)

print("\nIndexing Completed Successfully!")