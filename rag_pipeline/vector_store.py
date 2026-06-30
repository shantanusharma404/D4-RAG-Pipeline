import chromadb


class VectorStore:
    def __init__(self, db_path="chroma_db", collection_name="rag_documents"):
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def clear_collection(self):
        """
        Delete all existing documents from the collection.
        """

        try:
            self.client.delete_collection("rag_documents")
        except:
            pass

        self.collection = self.client.get_or_create_collection(
            name="rag_documents"
        )

        print("Old collection cleared.")

    def add_documents(self, chunks, embeddings):

        ids = [f"doc_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist()
        )

        print(f"Stored {len(chunks)} chunks successfully.")