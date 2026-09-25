from rag.pipeline import RAGPipeline
from llm.gemini import GeminiClient


def main():

    llm = GeminiClient()

    rag = RAGPipeline(llm)

    count = rag.ingest_pdf(
        "docs/STTL-AapleSarkar-CR-1.1 (002) (1) (1).pdf"
    )

    print(f"Indexed {count} chunks")

    for index, (chunk, embedding) in enumerate(
        zip(rag.chunks, rag.embeddings),
        start=1
    ):
        print(f"\n--- Chunk {index} ---")
        print(chunk)
        print("Embedding:")
        print(embedding.tolist())

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            break

        answer = rag.ask(question)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()