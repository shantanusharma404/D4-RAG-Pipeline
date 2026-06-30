import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(
        self,
        db_path="chroma_db",
        collection_name="rag_documents",
        model_name="all-MiniLM-L6-v2",
    ):
        # Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

        # Load the SAME embedding model used during indexing
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Retriever ready.\n")

    def retrieve(self, query, top_k=3):
        # Convert user query into an embedding
        query_embedding = self.model.encode(query).tolist()

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results