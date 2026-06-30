from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initializes the text splitter.

        Parameters:
            chunk_size (int): Maximum characters per chunk.
            chunk_overlap (int): Number of overlapping characters.
        """

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def split_text(self, text):
        """
        Splits raw text into chunks.

        Returns:
            List[str]
        """
        return self.splitter.split_text(text)