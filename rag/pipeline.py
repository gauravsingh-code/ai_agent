from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.prompt import build_rag_prompt


class RAGPipeline:

    def __init__(self, llm):
        self.llm = llm
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def ingest_pdf(self, file_path: str):

        text = load_pdf(file_path)

        chunks = chunk_text(text)

        embeddings = self.embedding_model.embed(chunks)

        ids = [
            f"chunk-{i}"
            for i in range(len(chunks))
        ]

        self.vector_store.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5):

        query_embedding = self.embedding_model.embed(
            [query]
        )[0]

        results = self.vector_store.search(
            query_embedding,
            top_k
        )

        return results["documents"][0]

    def ask(self, question: str):

        contexts = self.retrieve(question)

        prompt = build_rag_prompt(
            question,
            contexts
        )

        return self.llm.generate(prompt)