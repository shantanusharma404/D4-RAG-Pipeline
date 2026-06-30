from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the embedding model.
        """
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.\n")

    def generate_embeddings(self, chunks):
        """
        Generates embeddings for a list of text chunks.

        Args:
            chunks (list): List of text chunks.

        Returns:
            embeddings (numpy.ndarray)
        """
        embeddings = self.model.encode(chunks)

        return embeddings