from rag_pipeline.retriever import Retriever
from rag_pipeline.llm import GeminiLLM


class RAGPipeline:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = GeminiLLM()

    def ask(self, question):

        # Retrieve relevant chunks
        results = self.retriever.retrieve(question)

        # Merge retrieved chunks
        context = "\n\n".join(results["documents"][0])

        # Ask Gemini
        answer = self.llm.generate_answer(
            question=question,
            context=context
        )

        return answer