from rag.pipeline import RAGPipeline
from llm.gemini import GeminiClient


def main():

    llm = GeminiClient()

    rag = RAGPipeline(llm)

    count = rag.ingest_pdf(
        "docs/company_policy.pdf"
    )

    print(f"Indexed {count} chunks")

    while True:

        question = input("\nQuestion: ")

        if question.lower() == "exit":
            break

        answer = rag.ask(question)

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()