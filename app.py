from rag_pipeline.pipeline import RAGPipeline

print("=" * 60)
print("          BASIC RAG PIPELINE")
print("=" * 60)

rag = RAGPipeline()

while True:

    question = input("\nAsk a question (or type 'exit'): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer = rag.ask(question)

    print("\nAnswer:\n")
    print(answer)